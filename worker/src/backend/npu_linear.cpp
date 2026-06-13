#include "backend/npu_linear.h"
#include "backend/npu_weight_cache.h"
#include "ops/op_cast.h"
#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <sys/resource.h>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

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

std::vector<int> dynamic_m_values(int max_m) {
    std::vector<int> values;
    values.push_back(1);
    if (max_m > 1) {
        values.push_back(max_m);
    }

    const char* spec = std::getenv("RKLLM_LINEAR_EXTRA_M");
    if (spec && spec[0] != '\0') {
        const char* p = spec;
        while (*p) {
            while (*p == ',' || *p == ' ' || *p == '\t') {
                ++p;
            }
            if (!*p) {
                break;
            }
            char* end = nullptr;
            long parsed = std::strtol(p, &end, 10);
            if (end == p) {
                while (*p && *p != ',') {
                    ++p;
                }
                continue;
            }
            if (parsed > 1 && parsed < max_m) {
                values.push_back((int)parsed);
            }
            p = end;
        }
    }

    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
    constexpr size_t kMaxDynamicShapes = 4;
    if (values.size() > kMaxDynamicShapes) {
        std::fprintf(stderr,
                     "[NpuLinear] RKLLM_LINEAR_EXTRA_M creates %zu dynamic shapes; keep first %zu to avoid RKNN fd pressure\n",
                     values.size(), kMaxDynamicShapes);
        values.resize(kMaxDynamicShapes);
        if (values.back() != max_m) {
            values.back() = max_m;
            std::sort(values.begin(), values.end());
            values.erase(std::unique(values.begin(), values.end()), values.end());
        }
    }
    return values;
}

bool shard_dynamic_m_enabled() {
    const char* v = std::getenv("RKLLM_SHARD_DYNAMIC_M");
    return v && v[0] != '\0' &&
           std::strcmp(v, "0") != 0 &&
           std::strcmp(v, "false") != 0 &&
           std::strcmp(v, "FALSE") != 0 &&
           std::strcmp(v, "off") != 0 &&
           std::strcmp(v, "OFF") != 0;
}

uint32_t fp16_cache_flags_for_dynamic_m(bool dynamic_m, int max_m, bool output_f32 = false) {
    uint32_t flags = output_f32 ? 0x2u : 0;
    if (!dynamic_m) {
        return flags;
    }
    const uint32_t capped_m = (uint32_t)std::min(std::max(max_m, 1), 0xffff);
    uint32_t extra_sig = 0;
    for (int m : dynamic_m_values(max_m)) {
        if (m > 1 && m < max_m) {
            extra_sig = (extra_sig * 131u) ^ (uint32_t)m;
        }
    }
    flags |= 0x1u | (capped_m << 8);
    flags |= (extra_sig & 0xffu) << 24;
    return flags;
}

uint32_t predicted_fp16_cache_flags(bool has_core_mask, int N, bool output_f32 = false) {
    const int max_m = dynamic_m_limit();
    const bool dynamic_m = max_m > 1 &&
        (!has_core_mask || shard_dynamic_m_enabled()) &&
        N <= 32768;
    return fp16_cache_flags_for_dynamic_m(dynamic_m, max_m, output_f32);
}

void add_f32_inplace(float* dst, const float* src, int n) {
    int i = 0;
#if defined(__aarch64__)
    for (; i + 4 <= n; i += 4) {
        float32x4_t d = vld1q_f32(dst + i);
        float32x4_t s = vld1q_f32(src + i);
        vst1q_f32(dst + i, vaddq_f32(d, s));
    }
#endif
    for (; i < n; ++i) {
        dst[i] += src[i];
    }
}

}  // namespace

static inline uint16_t f16_order_key(uint16_t v) {
    return (v & 0x8000u) ? (uint16_t)~v : (uint16_t)(v ^ 0x8000u);
}

static bool argmax_f16_row(const uint16_t* out, int N,
                           int* argmax_id, uint16_t* argmax_value) {
    if (!out || N <= 0 || !argmax_id) {
        return false;
    }

    int best = 0;
    uint16_t best_val = out[0];
    uint16_t best_key = f16_order_key(out[0]);
    for (int i = 1; i < N; ++i) {
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

bool NpuLinear::init(int K, int N, const uint16_t* weight_kn) {
    if (!weight_kn) {
        std::fprintf(stderr, "[NpuLinear] invalid init args K=%d N=%d\n", K, N);
        return false;
    }

    rknn_matmul_info info{};
    if (!create_context_and_b(K, N, &info)) {
        return false;
    }
    const rknn_matmul_tensor_attr B_attr = current_b_attr();

    // 冷加载路径：weight_kn 是普通 [K, N] row-major FP16。RKNN native B
    // 是设备偏好的内部布局，不能直接用 memcpy 得到，必须调用转换 API。
    int ret = rknn_B_normal_layout_to_native_layout(
        (void*)weight_kn, B_mem_->virt_addr, K_, N_, &info);
    if (ret < 0) {
        std::fprintf(stderr,
                     "[NpuLinear] rknn_B_normal_layout_to_native_layout failed: %d\n",
                     ret);
        destroy();
        return false;
    }

    if (!cache_key_.empty()) {
        // 转换完成后把 B_mem_ 里的 native layout 原样写入缓存。下次加载
        // 可以直接读回 B_mem_，不再读取/转置/转换原始权重。
        npu_weight_cache::CacheSpec spec;
        spec.key = cache_key_;
        spec.kind = npu_weight_cache::CACHE_KIND_FP16_NATIVE;
        spec.K = K_;
        spec.N = N_;
        spec.K_matmul = K_;
        spec.flags = fp16_cache_flags_for_dynamic_m(dynamic_m_, dynamic_max_m_, output_f32_);
        spec.packed_bytes = B_attr.size;
        npu_weight_cache::write(spec, B_mem_->virt_addr, B_attr.size);
    }

    if (!bind_b_mem(B_attr)) {
        return false;
    }

    // A/C are activation buffers. Do not allocate them during load:
    // hundreds of Linear contexts would otherwise keep hundreds of dmabuf fds
    // open before inference even starts. forward() allocates lazily and reuses.
    return true;
}

bool NpuLinear::init_from_cache(int K, int N) {
    if (cache_key_.empty() || !npu_weight_cache::enabled()) {
        return false;
    }

    npu_weight_cache::CacheSpec spec;
    spec.key = cache_key_;
    spec.kind = npu_weight_cache::CACHE_KIND_FP16_NATIVE;
    spec.K = K;
    spec.N = N;
    spec.K_matmul = K;
    spec.flags = predicted_fp16_cache_flags(has_core_mask_, N, output_f32_);
    // 先只根据文件名探测，避免 cold miss 时创建 RKNN context/B buffer。
    if (!npu_weight_cache::exists(spec)) {
        return false;
    }

    // 确认缓存存在后才创建 context，并用实际 B_attr.size 完成 header 校验。
    rknn_matmul_info info{};
    if (!create_context_and_b(K, N, &info)) {
        return false;
    }
    const rknn_matmul_tensor_attr B_attr = current_b_attr();
    spec.K = K_;
    spec.N = N_;
    spec.K_matmul = K_;
    const uint32_t actual_flags =
        fp16_cache_flags_for_dynamic_m(dynamic_m_, dynamic_max_m_, output_f32_);
    if (actual_flags != spec.flags) {
        destroy();
        return false;
    }
    spec.packed_bytes = B_attr.size;

    if (!npu_weight_cache::read(spec, B_mem_->virt_addr, B_attr.size)) {
        destroy();
        return false;
    }
    if (!bind_b_mem(B_attr)) {
        return false;
    }
    return true;
}

bool NpuLinear::create_context_and_b(int K, int N, rknn_matmul_info* info_out) {
    destroy();
    ensure_nofile_limit();

    if (K <= 0 || N <= 0 || !info_out) {
        std::fprintf(stderr, "[NpuLinear] invalid init args K=%d N=%d\n", K, N);
        return false;
    }

    K_ = K;
    N_ = N;

    // 创建 matmul 上下文。M>1 必须使用 RKNN dynamic shape；
    // 在静态 M=1 context 上只改 tensor attr 会 silent wrong output。
    // B_layout=1 表示 B 使用 native layout；A/C 仍是 normal layout，
    // 因为运行时激活需要 CPU 直接读写。
    rknn_matmul_info info{};
    info.M             = 1;
    info.K             = K_;
    info.N             = N_;
    info.type          = output_f32_
        ? RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32
        : RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT16;
    info.B_layout      = 1;  // native layout（性能更好）
    info.AC_layout     = 0;  // normal layout
    info.B_quant_type  = 0;
    info.AC_quant_type = 0;

    int ret = 0;
    const bool allow_dynamic_m = (!has_core_mask_ || shard_dynamic_m_enabled()) && N_ <= 32768;
    dynamic_max_m_ = allow_dynamic_m ? dynamic_m_limit() : 1;
    if (dynamic_max_m_ > 1) {
        dynamic_ms_ = dynamic_m_values(dynamic_max_m_);
        dynamic_shapes_.resize(dynamic_ms_.size());
        dynamic_io_attrs_.resize(dynamic_ms_.size());
        for (size_t i = 0; i < dynamic_ms_.size(); ++i) {
            dynamic_shapes_[i] = rknn_matmul_shape{dynamic_ms_[i], K_, N_};
        }
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
            dynamic_ms_.clear();
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
            std::fprintf(stderr,
                         "[NpuLinear] rknn_matmul_create failed: %d (K=%d N=%d)\n",
                         ret, K_, N_);
            K_ = 0;
            N_ = 0;
            return false;
        }
    }

    if (has_core_mask_) {
        ret = rknn_matmul_set_core_mask(ctx_, core_mask_);
        if (ret < 0) {
            if (dynamic_m_) {
                std::fprintf(stderr,
                             "[NpuLinear] dynamic shape set_core_mask failed: %d "
                             "(K=%d N=%d maxM=%d), fallback static M=1\n",
                             ret, K_, N_, dynamic_max_m_);
                rknn_matmul_destroy(ctx_);
                ctx_ = 0;
                io_attr_ = {};
                dynamic_shapes_.clear();
                dynamic_io_attrs_.clear();
                dynamic_ms_.clear();
                dynamic_m_ = false;
                dynamic_max_m_ = 1;

                ret = rknn_matmul_create(&ctx_, &info, &io_attr_);
                if (ret < 0) {
                    std::fprintf(stderr,
                                 "[NpuLinear] fallback rknn_matmul_create failed: %d "
                                 "(K=%d N=%d)\n",
                                 ret, K_, N_);
                    K_ = 0;
                    N_ = 0;
                    return false;
                }
                ret = rknn_matmul_set_core_mask(ctx_, core_mask_);
            }
            if (ret < 0) {
                std::fprintf(stderr, "[NpuLinear] rknn_matmul_set_core_mask failed: %d\n", ret);
                destroy();
                return false;
            }
        }
    }

    const rknn_matmul_tensor_attr B_attr = current_b_attr();
    B_mem_ = rknn_create_mem(ctx_, B_attr.size);
    if (!B_mem_) {
        std::fprintf(stderr, "[NpuLinear] rknn_create_mem(B) failed\n");
        destroy();
        return false;
    }

    *info_out = info;
    return true;
}

rknn_matmul_tensor_attr NpuLinear::current_b_attr() const {
    return dynamic_m_ ? dynamic_io_attrs_[0].B : io_attr_.B;
}

bool NpuLinear::bind_b_mem(const rknn_matmul_tensor_attr& B_attr) {
    int ret = rknn_matmul_set_io_mem(
        ctx_, B_mem_, const_cast<rknn_matmul_tensor_attr*>(&B_attr));
    if (ret < 0) {
        std::fprintf(stderr, "[NpuLinear] rknn_matmul_set_io_mem(B) failed: %d\n", ret);
        destroy();
        return false;
    }
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
        A_mem_external_ = false;
        cur_M_ = 0;
        alloc_M_ = 0;
    };
    if (!A_mem_ || !C_mem_) {
        std::fprintf(stderr, "[NpuLinear] rknn_create_mem(A/C) failed M=%d\n", M);
        cleanup_ac();
        return false;
    }

    alloc_M_ = M;
    A_mem_external_ = false;

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
    for (size_t i = 0; i < dynamic_ms_.size(); ++i) {
        if (dynamic_ms_[i] == M && i < dynamic_io_attrs_.size()) {
            return (int)i;
        }
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
    A_mem_external_ = false;
    external_A_fd_ = -1;
    external_A_virt_addr_ = nullptr;
    external_A_offset_ = 0;
    cur_M_ = 0;
    alloc_M_ = 0;
}

bool NpuLinear::forward(const uint16_t* input_f16, int M, uint16_t* output_f16) {
    if (!ctx_ || !input_f16 || !output_f16 || K_ <= 0 || N_ <= 0 || M <= 0) {
        std::fprintf(stderr, "[NpuLinear] invalid forward args M=%d K=%d N=%d\n", M, K_, N_);
        return false;
    }

    uint16_t* prepared = prepare_input_f16(M);
    if (!prepared) {
        return false;
    }

    std::memcpy(prepared, input_f16, (size_t)M * K_ * sizeof(uint16_t));
    return forward_prepared(output_f16);
}

bool NpuLinear::forward_accumulate(const uint16_t* input_f16, int M, float* accum_f32) {
    if (!ctx_ || !input_f16 || !accum_f32 || K_ <= 0 || N_ <= 0 || M <= 0) {
        return false;
    }

    uint16_t* prepared = prepare_input_f16(M);
    if (!prepared) {
        return false;
    }

    std::memcpy(prepared, input_f16, (size_t)M * K_ * sizeof(uint16_t));
    return forward_prepared_accumulate(accum_f32);
}

bool NpuLinear::forward_f32_accumulate(const float* input_f32, int M, float* accum_f32) {
    if (!ctx_ || !input_f32 || !accum_f32 || K_ <= 0 || N_ <= 0 || M <= 0) {
        return false;
    }

    uint16_t* prepared = prepare_input_f16(M);
    if (!prepared) {
        return false;
    }

    op_f32_to_f16(input_f32, prepared, M * K_);
    return forward_prepared_accumulate(accum_f32);
}

uint16_t* NpuLinear::prepare_input_f16(int M) {
    if (!ctx_ || K_ <= 0 || N_ <= 0 || M <= 0) {
        return nullptr;
    }
    if (A_mem_external_) {
        release_ac();
    }
    if (!ensure_ac(M)) {
        return nullptr;
    }
    return reinterpret_cast<uint16_t*>(A_mem_->virt_addr);
}

const rknn_tensor_mem* NpuLinear::prepared_input_mem() const {
    return A_mem_;
}

bool NpuLinear::bind_external_input_f16(int M, const rknn_tensor_mem* external_mem) {
    if (!ctx_ || !external_mem || external_mem->fd < 0 ||
        !external_mem->virt_addr || K_ <= 0 || N_ <= 0 || M <= 0) {
        return false;
    }

    uint32_t A_size = 0;
    uint32_t C_size = 0;
    if (!ac_sizes(M, &A_size, &C_size) || external_mem->size < A_size) {
        return false;
    }

    if (!A_mem_external_ || !A_mem_ || !C_mem_ || cur_M_ != M ||
        external_A_fd_ != external_mem->fd ||
        external_A_virt_addr_ != external_mem->virt_addr ||
        external_A_offset_ != external_mem->offset) {
        release_ac();

        A_mem_ = rknn_create_mem_from_fd(ctx_, external_mem->fd,
                                         external_mem->virt_addr,
                                         A_size, external_mem->offset);
        C_mem_ = rknn_create_mem(ctx_, C_size);
        if (!A_mem_ || !C_mem_) {
            std::fprintf(stderr,
                         "[NpuLinear] shared rknn_create_mem(A/C) failed M=%d\n",
                         M);
            release_ac();
            return false;
        }

        A_mem_external_ = true;
        external_A_fd_ = external_mem->fd;
        external_A_virt_addr_ = external_mem->virt_addr;
        external_A_offset_ = external_mem->offset;
        alloc_M_ = M;
        if (!bind_ac(M)) {
            release_ac();
            return false;
        }
    }

    return true;
}

bool NpuLinear::forward_prepared(uint16_t* output_f16) {
    if (!ctx_ || !A_mem_ || !C_mem_ || !output_f16 ||
        K_ <= 0 || N_ <= 0 || cur_M_ <= 0) {
        return false;
    }

    const uint16_t* out = forward_prepared_output_f16();
    if (!out) {
        return false;
    }

    std::memcpy(output_f16, out, (size_t)cur_M_ * N_ * sizeof(uint16_t));
    return true;
}

const uint16_t* NpuLinear::forward_prepared_output_f16() {
    if (output_f32_ || !ctx_ || !A_mem_ || !C_mem_ ||
        K_ <= 0 || N_ <= 0 || cur_M_ <= 0) {
        return nullptr;
    }

    int ret = rknn_matmul_run(ctx_);
    if (ret < 0) {
        std::fprintf(stderr, "[NpuLinear] rknn_matmul_run failed: %d\n", ret);
        release_ac();
        return nullptr;
    }

    return reinterpret_cast<const uint16_t*>(C_mem_->virt_addr);
}

const float* NpuLinear::forward_prepared_output_f32() {
    if (!output_f32_ || !ctx_ || !A_mem_ || !C_mem_ ||
        K_ <= 0 || N_ <= 0 || cur_M_ <= 0) {
        return nullptr;
    }

    int ret = rknn_matmul_run(ctx_);
    if (ret < 0) {
        std::fprintf(stderr, "[NpuLinear] rknn_matmul_run f32 failed: %d\n", ret);
        release_ac();
        return nullptr;
    }

    return reinterpret_cast<const float*>(C_mem_->virt_addr);
}

bool NpuLinear::forward_prepared_accumulate(float* accum_f32) {
    if (!ctx_ || !A_mem_ || !C_mem_ || !accum_f32 ||
        K_ <= 0 || N_ <= 0 || cur_M_ <= 0) {
        return false;
    }

    if (output_f32_) {
        return forward_prepared_f32_accumulate(accum_f32);
    }

    const uint16_t* out = forward_prepared_output_f16();
    if (!out) {
        return false;
    }

    op_add_f16_to_f32_inplace(accum_f32, out, cur_M_ * N_);
    return true;
}

bool NpuLinear::forward_prepared_f32_accumulate(float* accum_f32) {
    if (!ctx_ || !A_mem_ || !C_mem_ || !accum_f32 ||
        K_ <= 0 || N_ <= 0 || cur_M_ <= 0) {
        return false;
    }

    const float* out = forward_prepared_output_f32();
    if (!out) {
        return false;
    }

    add_f32_inplace(accum_f32, out, cur_M_ * N_);
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
    uint16_t* prepared = prepare_input_f16(M);
    if (!prepared) {
        return false;
    }

    std::memcpy(prepared, input_f16, (size_t)K_ * sizeof(uint16_t));
    return forward_prepared_argmax(argmax_id, argmax_value);
}

bool NpuLinear::forward_prepared_argmax(int* argmax_id, uint16_t* argmax_value) {
    if (output_f32_ || !argmax_id || !ctx_ || !A_mem_ || !C_mem_ ||
        K_ <= 0 || N_ <= 0 || cur_M_ != 1) {
        return false;
    }

    const uint16_t* out = forward_prepared_output_f16();
    if (!out) {
        return false;
    }

    return argmax_f16_row(out, N_, argmax_id, argmax_value);
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
    dynamic_ms_.clear();
    dynamic_m_ = false;
    dynamic_max_m_ = 1;
}

void NpuLinear::set_core_mask(rknn_core_mask mask) {
    core_mask_ = mask;
    has_core_mask_ = true;
}
