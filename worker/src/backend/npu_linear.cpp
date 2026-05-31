#include "backend/npu_linear.h"
#include "ops/op_cast.h"
#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <sys/resource.h>

namespace {

void ensure_nofile_limit() {
    static bool done = false;
    if (done) return;
    done = true;

    constexpr rlim_t kTargetNoFile = 4096;
    struct rlimit lim {};
    if (getrlimit(RLIMIT_NOFILE, &lim) != 0) {
        return;
    }
    if (lim.rlim_cur >= kTargetNoFile) {
        return;
    }

    rlim_t new_soft = kTargetNoFile;
    if (lim.rlim_max != RLIM_INFINITY && lim.rlim_max < new_soft) {
        new_soft = lim.rlim_max;
    }
    if (new_soft <= lim.rlim_cur) {
        std::fprintf(stderr,
                     "[NpuLinear] RLIMIT_NOFILE soft=%llu hard=%llu; "
                     "consider `ulimit -n 4096` before running\n",
                     (unsigned long long)lim.rlim_cur,
                     (unsigned long long)lim.rlim_max);
        return;
    }

    struct rlimit updated = lim;
    updated.rlim_cur = new_soft;
    if (setrlimit(RLIMIT_NOFILE, &updated) == 0) {
        std::fprintf(stderr,
                     "[NpuLinear] raised RLIMIT_NOFILE soft limit to %llu\n",
                     (unsigned long long)new_soft);
    }
}

int dynamic_m_limit() {
    constexpr int kDefaultBatchRows = 1;
    const char* batch = std::getenv("RKLLM_LINEAR_BATCH");
    if (!batch || batch[0] == '\0') {
        return kDefaultBatchRows;
    }

    char* end = nullptr;
    long parsed = std::strtol(batch, &end, 10);
    if (end == batch || parsed <= 1) {
        return 1;
    }
    return std::min<long>(parsed, 512);
}

}  // namespace

static inline uint16_t f16_order_key(uint16_t v) {
    return (v & 0x8000u) ? (uint16_t)~v : (uint16_t)(v ^ 0x8000u);
}

bool NpuLinear::init(int K, int N, const uint16_t* weight_kn) {
    destroy();

    ensure_nofile_limit();

    K_ = K; N_ = N;

    // 创建 matmul 上下文。M>1 必须使用 RKNN dynamic shape；
    // 在静态 M=1 context 上只改 tensor attr 会 silent wrong output。
    rknn_matmul_info info{};
    info.M             = 1;
    info.K             = K_;
    info.N             = N_;
    info.type          = RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT16;
    info.B_layout      = 1;  // native layout（性能更好）
    info.AC_layout     = 0;  // normal layout
    info.B_quant_type  = 0;
    info.AC_quant_type = 0;

    int ret = 0;
    dynamic_max_m_ = (has_core_mask_ || N_ > 32768) ? 1 : dynamic_m_limit();
    if (dynamic_max_m_ > 1) {
        dynamic_shapes_.resize(2);
        dynamic_io_attrs_.resize(2);
        dynamic_shapes_[0] = rknn_matmul_shape{1, K_, N_};
        dynamic_shapes_[1] = rknn_matmul_shape{dynamic_max_m_, K_, N_};
        ret = rknn_matmul_create_dynamic_shape(
            &ctx_, &info, (int)dynamic_shapes_.size(),
            dynamic_shapes_.data(), dynamic_io_attrs_.data());
        if (ret < 0) {
            std::fprintf(stderr,
                         "[NpuLinear] dynamic shape create failed: %d (K=%d N=%d maxM=%d), fallback static M=1\n",
                         ret, K_, N_, dynamic_max_m_);
            ctx_ = 0;
            dynamic_shapes_.clear();
            dynamic_io_attrs_.clear();
            dynamic_m_ = false;
            dynamic_max_m_ = 1;
        } else {
            dynamic_m_ = true;
            io_attr_ = dynamic_io_attrs_[0];
        }
    }

    if (!ctx_) {
        ret = rknn_matmul_create(&ctx_, &info, &io_attr_);
        if (ret < 0) {
            std::fprintf(stderr, "[NpuLinear] rknn_matmul_create failed: %d (K=%d N=%d)\n", ret, K_, N_);
            K_ = 0;
            N_ = 0;
            return false;
        }
    }

    auto fail = [this](const char* msg, int code) {
        if (code < 0) {
            std::fprintf(stderr, "[NpuLinear] %s failed: %d\n", msg, code);
        } else {
            std::fprintf(stderr, "[NpuLinear] %s failed\n", msg);
        }
        destroy();
        return false;
    };

    if (has_core_mask_) {
        ret = rknn_matmul_set_core_mask(ctx_, core_mask_);
        if (ret < 0) {
            return fail("rknn_matmul_set_core_mask", ret);
        }
    }

    // 分配 B 内存
    const rknn_matmul_tensor_attr& B_attr = dynamic_m_
        ? dynamic_io_attrs_[0].B
        : io_attr_.B;
    B_mem_ = rknn_create_mem(ctx_, B_attr.size);
    if (!B_mem_) {
        return fail("rknn_create_mem(B)", 0);
    }

    // normal layout 权重 -> native layout
    ret = rknn_B_normal_layout_to_native_layout(
        (void*)weight_kn, B_mem_->virt_addr, K_, N_, &info);
    if (ret < 0) {
        return fail("rknn_B_normal_layout_to_native_layout", ret);
    }

    ret = rknn_matmul_set_io_mem(ctx_, B_mem_, const_cast<rknn_matmul_tensor_attr*>(&B_attr));
    if (ret < 0) {
        return fail("rknn_matmul_set_io_mem(B)", ret);
    }

    // A/C are activation buffers. Do not allocate them during load:
    // hundreds of Linear contexts would otherwise keep hundreds of dmabuf fds
    // open before inference even starts. forward() allocates lazily and reuses.
    return true;
}

bool NpuLinear::rebuild_ac(int M) {
    release_ac();

    uint32_t A_size = 0;
    uint32_t C_size = 0;
    if (!ac_sizes(M, &A_size, &C_size)) {
        return false;
    }

    A_mem_ = rknn_create_mem(ctx_, A_size);
    C_mem_ = rknn_create_mem(ctx_, C_size);
    auto cleanup_ac = [this]() {
        if (A_mem_) { rknn_destroy_mem(ctx_, A_mem_); A_mem_ = nullptr; }
        if (C_mem_) { rknn_destroy_mem(ctx_, C_mem_); C_mem_ = nullptr; }
        cur_M_ = 0;
        alloc_M_ = 0;
    };
    if (!A_mem_ || !C_mem_) {
        std::fprintf(stderr, "[NpuLinear] rknn_create_mem(A/C) failed M=%d\n", M);
        cleanup_ac();
        return false;
    }

    alloc_M_ = M;

    if (!bind_ac(M)) {
        cleanup_ac();
        return false;
    }

    return true;
}

bool NpuLinear::bind_ac(int M, bool quiet) {
    if (dynamic_m_) {
        const int shape_idx = dynamic_index_for_m(M);
        if (shape_idx < 0) {
            if (!quiet) {
                std::fprintf(stderr,
                             "[NpuLinear] unsupported dynamic M=%d maxM=%d\n",
                             M, dynamic_max_m_);
            }
            return false;
        }
        rknn_matmul_shape shape = dynamic_shapes_[(size_t)shape_idx];
        int ret = rknn_matmul_set_dynamic_shape(ctx_, &shape);
        if (ret < 0) {
            if (!quiet) {
                std::fprintf(stderr, "[NpuLinear] set dynamic shape M=%d failed: %d\n", M, ret);
            }
            return false;
        }
    } else if (M != 1) {
        return false;
    }

    const int shape_idx = dynamic_m_ ? dynamic_index_for_m(M) : -1;
    rknn_matmul_tensor_attr A_attr = dynamic_m_
        ? dynamic_io_attrs_[(size_t)shape_idx].A
        : io_attr_.A;
    rknn_matmul_tensor_attr C_attr = dynamic_m_
        ? dynamic_io_attrs_[(size_t)shape_idx].C
        : io_attr_.C;

    int ret = rknn_matmul_set_io_mem(ctx_, A_mem_, &A_attr);
    if (ret < 0) {
        if (!quiet) {
            std::fprintf(stderr, "[NpuLinear] set A failed: %d\n", ret);
        }
        return false;
    }

    ret = rknn_matmul_set_io_mem(ctx_, C_mem_, &C_attr);
    if (ret < 0) {
        if (!quiet) {
            std::fprintf(stderr, "[NpuLinear] set C failed: %d\n", ret);
        }
        return false;
    }

    cur_M_ = M;
    return true;
}

bool NpuLinear::ac_sizes(int M, uint32_t* A_size, uint32_t* C_size) const {
    if (!A_size || !C_size || M <= 0) {
        return false;
    }
    if (dynamic_m_) {
        const int shape_idx = dynamic_index_for_m(M);
        if (shape_idx < 0) {
            return false;
        }
        *A_size = dynamic_io_attrs_[(size_t)shape_idx].A.size;
        *C_size = dynamic_io_attrs_[(size_t)shape_idx].C.size;
        return true;
    }
    if (M != 1) {
        return false;
    }
    *A_size = io_attr_.A.size;
    *C_size = io_attr_.C.size;
    return true;
}

int NpuLinear::dynamic_index_for_m(int M) const {
    if (!dynamic_m_) {
        return -1;
    }
    if (M == 1) {
        return 0;
    }
    if (M == dynamic_max_m_ && dynamic_io_attrs_.size() >= 2) {
        return 1;
    }
    return -1;
}

bool NpuLinear::ensure_ac(int M) {
    if (!A_mem_ || !C_mem_ || M > alloc_M_) {
        return rebuild_ac(M);
    }
    if (M != cur_M_) {
        if (bind_ac(M, /*quiet=*/true)) {
            return true;
        }
        return rebuild_ac(M);
    }
    return true;
}

void NpuLinear::release_ac() {
    if (A_mem_) { rknn_destroy_mem(ctx_, A_mem_); A_mem_ = nullptr; }
    if (C_mem_) { rknn_destroy_mem(ctx_, C_mem_); C_mem_ = nullptr; }
    cur_M_ = 0;
    alloc_M_ = 0;
}

bool NpuLinear::forward(const uint16_t* input_f16, int M, uint16_t* output_f16) {
    if (!ctx_ || !input_f16 || !output_f16 || K_ <= 0 || N_ <= 0 || M <= 0) {
        std::fprintf(stderr, "[NpuLinear] invalid forward args M=%d K=%d N=%d\n", M, K_, N_);
        return false;
    }

    if (!ensure_ac(M)) {
        return false;
    }

    std::memcpy(A_mem_->virt_addr, input_f16, (size_t)M * K_ * 2);

    int ret = rknn_matmul_run(ctx_);
    if (ret < 0) {
        std::fprintf(stderr, "[NpuLinear] rknn_matmul_run failed: %d\n", ret);
        release_ac();
        return false;
    }

    std::memcpy(output_f16, C_mem_->virt_addr, (size_t)M * N_ * 2);
    return true;
}

bool NpuLinear::forward_accumulate(const uint16_t* input_f16, int M, float* accum_f32) {
    if (!ctx_ || !input_f16 || !accum_f32 || K_ <= 0 || N_ <= 0 || M <= 0) {
        return false;
    }
    if (!ensure_ac(M)) {
        return false;
    }

    std::memcpy(A_mem_->virt_addr, input_f16, (size_t)M * K_ * sizeof(uint16_t));
    int ret = rknn_matmul_run(ctx_);
    if (ret < 0) {
        std::fprintf(stderr, "[NpuLinear] rknn_matmul_run failed: %d\n", ret);
        release_ac();
        return false;
    }

    op_add_f16_to_f32_inplace(
        accum_f32, reinterpret_cast<const uint16_t*>(C_mem_->virt_addr), M * N_);
    return true;
}

bool NpuLinear::forward_f32_accumulate(const float* input_f32, int M, float* accum_f32) {
    if (!ctx_ || !input_f32 || !accum_f32 || K_ <= 0 || N_ <= 0 || M <= 0) {
        return false;
    }
    if (!ensure_ac(M)) {
        return false;
    }

    op_f32_to_f16(input_f32, reinterpret_cast<uint16_t*>(A_mem_->virt_addr), M * K_);
    int ret = rknn_matmul_run(ctx_);
    if (ret < 0) {
        std::fprintf(stderr, "[NpuLinear] rknn_matmul_run failed: %d\n", ret);
        release_ac();
        return false;
    }

    op_add_f16_to_f32_inplace(
        accum_f32, reinterpret_cast<const uint16_t*>(C_mem_->virt_addr), M * N_);
    return true;
}

bool NpuLinear::supports_batch(int M) const {
    return M == 1 || dynamic_index_for_m(M) >= 0;
}

bool NpuLinear::forward_argmax(const uint16_t* input_f16, int M,
                               int* argmax_id, uint16_t* argmax_value) {
    if (!argmax_id || M != 1 || !ctx_ || !input_f16 || K_ <= 0 || N_ <= 0) {
        return false;
    }
    if (!ensure_ac(M)) {
        return false;
    }

    std::memcpy(A_mem_->virt_addr, input_f16, (size_t)K_ * 2);

    int ret = rknn_matmul_run(ctx_);
    if (ret < 0) {
        std::fprintf(stderr, "[NpuLinear] rknn_matmul_run failed: %d\n", ret);
        release_ac();
        return false;
    }

    const uint16_t* out = reinterpret_cast<const uint16_t*>(C_mem_->virt_addr);
    int best = 0;
    uint16_t best_val = out[0];
    uint16_t best_key = f16_order_key(out[0]);
    for (int i = 1; i < N_; ++i) {
        const uint16_t key = f16_order_key(out[i]);
        if (key > best_key) {
            best_key = key;
            best_val = out[i];
            best = i;
        }
    }
    *argmax_id = best;
    if (argmax_value) {
        *argmax_value = best_val;
    }
    return true;
}

void NpuLinear::destroy() {
    release_ac();
    if (B_mem_) { rknn_destroy_mem(ctx_, B_mem_); B_mem_ = nullptr; }
    if (ctx_)   { rknn_matmul_destroy(ctx_);      ctx_    = 0;     }
    K_ = 0;
    N_ = 0;
    cur_M_ = 0;
    alloc_M_ = 0;
    dynamic_shapes_.clear();
    dynamic_io_attrs_.clear();
    dynamic_m_ = false;
    dynamic_max_m_ = 1;
}

void NpuLinear::set_core_mask(rknn_core_mask mask) {
    core_mask_ = mask;
    has_core_mask_ = true;
}
