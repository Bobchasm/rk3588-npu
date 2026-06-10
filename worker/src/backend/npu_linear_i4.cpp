#include "backend/npu_linear_i4.h"

#include "backend/npu_weight_cache.h"
#include "core/half.h"
#include "ops/op_cast.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/resource.h>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace {

void ensure_nofile_limit_i4() {
    static bool done = false;
    if (done) return;
    done = true;

    constexpr rlim_t kTargetNoFile = 4096;
    struct rlimit lim {};
    if (getrlimit(RLIMIT_NOFILE, &lim) != 0 || lim.rlim_cur >= kTargetNoFile) {
        return;
    }
    rlim_t new_soft = kTargetNoFile;
    if (lim.rlim_max != RLIM_INFINITY && lim.rlim_max < new_soft) {
        new_soft = lim.rlim_max;
    }
    if (new_soft > lim.rlim_cur) {
        struct rlimit updated = lim;
        updated.rlim_cur = new_soft;
        setrlimit(RLIMIT_NOFILE, &updated);
    }
}

inline int align_up(int v, int align) {
    return ((v + align - 1) / align) * align;
}

inline int8_t clamp_i4(int v) {
    return (int8_t)std::max(-7, std::min(7, v));
}

inline int8_t clamp_i8(int v) {
    return (int8_t)std::max(-127, std::min(127, v));
}

inline void set_i4(uint8_t* data, size_t idx, int8_t value) {
    uint8_t& byte = data[idx / 2];
    const uint8_t packed = (uint8_t)value & 0x0f;
    if ((idx & 1) == 0) {
        byte = (byte & 0xf0u) | packed;
    } else {
        byte = (byte & 0x0fu) | (uint8_t)(packed << 4);
    }
}

inline uint16_t f16_order_key(uint16_t v) {
    return (v & 0x8000u) ? (uint16_t)~v : (uint16_t)(v ^ 0x8000u);
}

int linear_batch_limit_i4() {
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

bool shard_dynamic_m_enabled_i4() {
    const char* v = std::getenv("RKLLM_SHARD_DYNAMIC_M");
    return v && v[0] != '\0' &&
           std::strcmp(v, "0") != 0 &&
           std::strcmp(v, "false") != 0 &&
           std::strcmp(v, "FALSE") != 0 &&
           std::strcmp(v, "off") != 0 &&
           std::strcmp(v, "OFF") != 0;
}

}  // namespace

bool NpuLinearI4::configure_shape(int K, int N) {
    if (K <= 0 || N <= 0) {
        std::fprintf(stderr, "[NpuLinearI4] invalid init args K=%d N=%d\n", K, N);
        return false;
    }
    if ((N % 64) != 0) {
        std::fprintf(stderr, "[NpuLinearI4] N=%d is not aligned to 64 for A4W4\n", N);
        return false;
    }

    K_ = K;
    K_matmul_ = align_up(K_, 32);
    N_ = N;
    return true;
}

bool NpuLinearI4::create_context_and_b() {
    rknn_matmul_info info{};
    info.M = 1;
    info.K = K_matmul_;
    info.N = N_;
    info.type = RKNN_INT8_MM_INT4_TO_INT32;
    info.B_layout = 1;
    info.AC_layout = 0;
    info.B_quant_type = 0;
    info.AC_quant_type = 0;

    int ret = 0;
    const bool allow_dynamic_m = !has_core_mask_ || shard_dynamic_m_enabled_i4();
    dynamic_max_m_ = allow_dynamic_m ? linear_batch_limit_i4() : 1;
    if (dynamic_max_m_ > 1) {
        dynamic_shapes_.resize(2);
        dynamic_io_attrs_.resize(2);
        dynamic_shapes_[0] = rknn_matmul_shape{1, K_matmul_, N_};
        dynamic_shapes_[1] = rknn_matmul_shape{dynamic_max_m_, K_matmul_, N_};
        ret = rknn_matmul_create_dynamic_shape(
            &ctx_, &info, (int)dynamic_shapes_.size(),
            dynamic_shapes_.data(), dynamic_io_attrs_.data());
        if (ret < 0) {
            std::fprintf(stderr,
                         "[NpuLinearI4] dynamic shape create failed: %d (K=%d N=%d maxM=%d), fallback static M=1\n",
                         ret, K_matmul_, N_, dynamic_max_m_);
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
            std::fprintf(stderr, "[NpuLinearI4] rknn_matmul_create failed: %d (K=%d N=%d)\n",
                         ret, K_matmul_, N_);
            destroy();
            return false;
        }
    }

    if (has_core_mask_) {
        ret = rknn_matmul_set_core_mask(ctx_, core_mask_);
        if (ret < 0) {
            std::fprintf(stderr, "[NpuLinearI4] rknn_matmul_set_core_mask failed: %d\n", ret);
            destroy();
            return false;
        }
    }

    const rknn_matmul_tensor_attr B_attr = current_b_attr();
    B_mem_ = rknn_create_mem(ctx_, B_attr.size);
    if (!B_mem_) {
        std::fprintf(stderr, "[NpuLinearI4] rknn_create_mem(B) failed\n");
        destroy();
        return false;
    }
    return true;
}

rknn_matmul_tensor_attr NpuLinearI4::current_b_attr() const {
    return dynamic_m_ ? dynamic_io_attrs_[0].B : io_attr_.B;
}

bool NpuLinearI4::bind_b_mem() {
    rknn_matmul_tensor_attr B_attr = current_b_attr();
    const int ret = rknn_matmul_set_io_mem(ctx_, B_mem_, &B_attr);
    if (ret < 0) {
        std::fprintf(stderr, "[NpuLinearI4] rknn_matmul_set_io_mem(B) failed: %d\n", ret);
        destroy();
        return false;
    }
    return true;
}

uint32_t NpuLinearI4::cache_flags(bool dynamic_m) const {
    // Version bit 0x1 distinguishes this path from the previous
    // INT4xINT4->INT16 implementation. Reusing old native B cache would keep
    // the old accumulator semantics and can look like severe quantization loss.
    uint32_t flags = 0x1u;
    if (dynamic_m) {
        const uint32_t capped_m = (uint32_t)std::min(std::max(dynamic_max_m_, 1), 255);
        flags |= 0x2u | (capped_m << 24);
    }
    return flags;
}

bool NpuLinearI4::init(int K, int N, const uint16_t* weight_kn) {
    destroy();
    ensure_nofile_limit_i4();

    if (!weight_kn) {
        std::fprintf(stderr, "[NpuLinearI4] invalid init args K=%d N=%d\n", K, N);
        return false;
    }
    if (!configure_shape(K, N)) {
        return false;
    }

    const size_t packed_weight_bytes = ((size_t)K_matmul_ * N_ + 1) / 2;
    std::vector<uint8_t> weight_i4(packed_weight_bytes, 0);
    scales_.assign(N_, 1.0f);
    for (int n = 0; n < N_; ++n) {
        float max_abs = 0.0f;
        for (int k = 0; k < K_; ++k) {
            max_abs = std::max(max_abs, std::fabs(f16_to_f32(weight_kn[(size_t)k * N_ + n])));
        }
        const float scale = max_abs > 0.0f ? max_abs / 7.0f : 1.0f;
        scales_[n] = scale;
        const float inv = 1.0f / scale;
        for (int k = 0; k < K_; ++k) {
            int q = (int)std::lrint(f16_to_f32(weight_kn[(size_t)k * N_ + n]) * inv);
            set_i4(weight_i4.data(), (size_t)k * N_ + n, clamp_i4(q));
        }
    }

    if (!create_context_and_b()) {
        return false;
    }

    rknn_matmul_info info{};
    info.M = 1;
    info.K = K_matmul_;
    info.N = N_;
    info.type = RKNN_INT8_MM_INT4_TO_INT32;
    info.B_layout = 1;
    info.AC_layout = 0;
    info.B_quant_type = 0;
    info.AC_quant_type = 0;

    int ret = rknn_B_normal_layout_to_native_layout(weight_i4.data(), B_mem_->virt_addr,
                                                    K_matmul_, N_, &info);
    if (ret < 0) {
        std::fprintf(stderr,
                     "[NpuLinearI4] rknn_B_normal_layout_to_native_layout failed: %d\n",
                     ret);
        destroy();
        return false;
    }

    if (!cache_key_.empty()) {
        npu_weight_cache::CacheSpec spec;
        spec.key = cache_key_;
        spec.kind = npu_weight_cache::CACHE_KIND_A4W4_NATIVE;
        spec.K = K_;
        spec.N = N_;
        spec.K_matmul = K_matmul_;
        spec.flags = cache_flags(dynamic_m_);
        const rknn_matmul_tensor_attr B_attr = current_b_attr();
        spec.packed_bytes = B_attr.size;
        spec.aux_bytes = scales_.size() * sizeof(float);
        npu_weight_cache::write(spec,
                                B_mem_->virt_addr, B_attr.size,
                                scales_.data(), scales_.size() * sizeof(float));
    }

    if (!bind_b_mem()) {
        return false;
    }

    std::fprintf(stderr, "[NpuLinearA4W4] init K=%d", K_);
    if (K_matmul_ != K_) {
        std::fprintf(stderr, "->%d", K_matmul_);
    }
    std::fprintf(stderr, " N=%d\n", N_);
    return true;
}

bool NpuLinearI4::init_from_cache(int K, int N) {
    if (cache_key_.empty() || !npu_weight_cache::enabled()) {
        return false;
    }

    destroy();
    ensure_nofile_limit_i4();
    if (!configure_shape(K, N)) {
        return false;
    }

    npu_weight_cache::CacheSpec spec;
    spec.key = cache_key_;
    spec.kind = npu_weight_cache::CACHE_KIND_A4W4_NATIVE;
    spec.K = K_;
    spec.N = N_;
    spec.K_matmul = K_matmul_;
    const bool predicted_dynamic_m =
        linear_batch_limit_i4() > 1 &&
        (!has_core_mask_ || shard_dynamic_m_enabled_i4());
    dynamic_max_m_ = predicted_dynamic_m ? linear_batch_limit_i4() : 1;
    spec.flags = cache_flags(predicted_dynamic_m);
    if (!npu_weight_cache::exists(spec)) {
        destroy();
        return false;
    }

    scales_.resize((size_t)N_);
    if (!create_context_and_b()) {
        return false;
    }
    const rknn_matmul_tensor_attr B_attr = current_b_attr();
    const uint32_t actual_flags = cache_flags(dynamic_m_);
    if (actual_flags != spec.flags) {
        destroy();
        return false;
    }
    spec.packed_bytes = B_attr.size;
    spec.aux_bytes = scales_.size() * sizeof(float);
    if (!npu_weight_cache::read(spec,
                                B_mem_->virt_addr, B_attr.size,
                                scales_.data(), scales_.size() * sizeof(float))) {
        destroy();
        return false;
    }
    if (!bind_b_mem()) {
        return false;
    }
    std::fprintf(stderr, "[NpuLinearA4W4] init K=%d", K_);
    if (K_matmul_ != K_) {
        std::fprintf(stderr, "->%d", K_matmul_);
    }
    std::fprintf(stderr, " N=%d cache\n", N_);
    return true;
}

bool NpuLinearI4::rebuild_ac(int M) {
    release_ac();

    uint32_t A_size = 0;
    uint32_t C_size = 0;
    if (!ac_sizes(M, &A_size, &C_size)) {
        return false;
    }

    A_mem_ = rknn_create_mem(ctx_, A_size);
    C_mem_ = rknn_create_mem(ctx_, C_size);
    auto cleanup = [this]() {
        if (A_mem_) { rknn_destroy_mem(ctx_, A_mem_); A_mem_ = nullptr; }
        if (C_mem_) { rknn_destroy_mem(ctx_, C_mem_); C_mem_ = nullptr; }
        cur_M_ = 0;
        alloc_M_ = 0;
    };
    if (!A_mem_ || !C_mem_) {
        std::fprintf(stderr, "[NpuLinearI4] rknn_create_mem(A/C) failed M=%d\n", M);
        cleanup();
        return false;
    }
    alloc_M_ = M;
    if (!bind_ac(M)) {
        cleanup();
        return false;
    }
    return true;
}

bool NpuLinearI4::bind_ac(int M, bool quiet) {
    if (dynamic_m_) {
        const int shape_idx = dynamic_index_for_m(M);
        if (shape_idx < 0) {
            if (!quiet) {
                std::fprintf(stderr,
                             "[NpuLinearI4] unsupported dynamic M=%d maxM=%d\n",
                             M, dynamic_max_m_);
            }
            return false;
        }
        rknn_matmul_shape shape = dynamic_shapes_[(size_t)shape_idx];
        int ret = rknn_matmul_set_dynamic_shape(ctx_, &shape);
        if (ret < 0) {
            if (!quiet) {
                std::fprintf(stderr, "[NpuLinearI4] set dynamic shape M=%d failed: %d\n",
                             M, ret);
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
        if (!quiet) std::fprintf(stderr, "[NpuLinearI4] set A failed: %d\n", ret);
        return false;
    }
    ret = rknn_matmul_set_io_mem(ctx_, C_mem_, &C_attr);
    if (ret < 0) {
        if (!quiet) std::fprintf(stderr, "[NpuLinearI4] set C failed: %d\n", ret);
        return false;
    }
    cur_M_ = M;
    return true;
}

bool NpuLinearI4::ac_sizes(int M, uint32_t* A_size, uint32_t* C_size) const {
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

int NpuLinearI4::dynamic_index_for_m(int M) const {
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

bool NpuLinearI4::ensure_ac(int M) {
    if (!A_mem_ || !C_mem_ || M > alloc_M_) {
        return rebuild_ac(M);
    }
    if (M != cur_M_) {
        return bind_ac(M, true) || rebuild_ac(M);
    }
    return true;
}

uint16_t* NpuLinearI4::prepare_input_f16(int M) {
    if (!ctx_ || K_ <= 0 || K_matmul_ <= 0 || N_ <= 0 || M <= 0) {
        return nullptr;
    }
    if (!ensure_ac(M)) {
        return nullptr;
    }
    prepared_input_f16_.resize((size_t)M * K_);
    prepared_M_ = M;
    return prepared_input_f16_.data();
}

float NpuLinearI4::quantize_current_input(const uint16_t* input_f16, int8_t* input_i8) {
    std::memset(input_i8, 0, (size_t)K_matmul_);
    float max_abs = 0.0f;
    for (int k = 0; k < K_; ++k) {
        max_abs = std::max(max_abs, std::fabs(f16_to_f32(input_f16[k])));
    }
    const float scale = max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
    const float inv = 1.0f / scale;
    for (int k = 0; k < K_; ++k) {
        int q = (int)std::lrint(f16_to_f32(input_f16[k]) * inv);
        input_i8[k] = clamp_i8(q);
    }
    return scale;
}

bool NpuLinearI4::run_prepared_raw(std::vector<float>* input_scales) {
    if (!ctx_ || !A_mem_ || !C_mem_ || !input_scales ||
        K_ <= 0 || K_matmul_ <= 0 || N_ <= 0 || prepared_M_ <= 0 ||
        prepared_input_f16_.size() < (size_t)prepared_M_ * K_) {
        return false;
    }

    input_scales->resize((size_t)prepared_M_);
    int8_t* a = reinterpret_cast<int8_t*>(A_mem_->virt_addr);
    for (int m = 0; m < prepared_M_; ++m) {
        (*input_scales)[(size_t)m] = quantize_current_input(
            prepared_input_f16_.data() + (size_t)m * K_,
            a + (size_t)m * K_matmul_);
    }
    const int ret = rknn_matmul_run(ctx_);
    if (ret < 0) {
        std::fprintf(stderr, "[NpuLinearI4] rknn_matmul_run failed: %d\n", ret);
        release_ac();
        return false;
    }
    return true;
}

void NpuLinearI4::scale_output_f16(const int32_t* raw, const float* input_scales,
                                   int M, uint16_t* out) const {
    if (!input_scales) {
        return;
    }
    for (int m = 0; m < M; ++m) {
        const int32_t* src = raw + (size_t)m * N_;
        uint16_t* dst = out + (size_t)m * N_;
        for (int n = 0; n < N_; ++n) {
            dst[n] = f32_to_f16((float)src[n] * input_scales[m] * scales_[n]);
        }
    }
}

bool NpuLinearI4::forward(const uint16_t* input_f16, int M, uint16_t* output_f16) {
    if (!ctx_ || !input_f16 || !output_f16 || K_ <= 0 || K_matmul_ <= 0 ||
        N_ <= 0 || M <= 0) {
        return false;
    }
    uint16_t* prepared = prepare_input_f16(M);
    if (!prepared) {
        return false;
    }
    std::memcpy(prepared, input_f16, (size_t)M * K_ * sizeof(uint16_t));
    return forward_prepared(output_f16);
}

bool NpuLinearI4::forward_accumulate(const uint16_t* input_f16, int M, float* accum_f32) {
    if (!ctx_ || !input_f16 || !accum_f32 || K_ <= 0 || K_matmul_ <= 0 ||
        N_ <= 0 || M <= 0) {
        return false;
    }
    uint16_t* prepared = prepare_input_f16(M);
    if (!prepared) {
        return false;
    }
    std::memcpy(prepared, input_f16, (size_t)M * K_ * sizeof(uint16_t));
    return forward_prepared_accumulate(accum_f32);
}

bool NpuLinearI4::forward_f32_accumulate(const float* input_f32, int M, float* accum_f32) {
    if (!ctx_ || !input_f32 || !accum_f32 || K_ <= 0 || K_matmul_ <= 0 ||
        N_ <= 0 || M <= 0) {
        return false;
    }
    uint16_t* prepared = prepare_input_f16(M);
    if (!prepared) {
        return false;
    }
    op_f32_to_f16(input_f32, prepared, M * K_);
    return forward_prepared_accumulate(accum_f32);
}

bool NpuLinearI4::forward_prepared(uint16_t* output_f16) {
    if (!output_f16) {
        return false;
    }
    const uint16_t* out = forward_prepared_output_f16();
    if (!out) {
        return false;
    }
    std::memcpy(output_f16, out, (size_t)prepared_M_ * N_ * sizeof(uint16_t));
    return true;
}

const uint16_t* NpuLinearI4::forward_prepared_output_f16() {
    if (!run_prepared_raw(&prepared_input_scales_)) {
        return nullptr;
    }

    prepared_output_f16_.resize((size_t)prepared_M_ * N_);
    scale_output_f16(reinterpret_cast<const int32_t*>(C_mem_->virt_addr),
                     prepared_input_scales_.data(), prepared_M_,
                     prepared_output_f16_.data());
    return prepared_output_f16_.data();
}

bool NpuLinearI4::forward_prepared_accumulate(float* accum_f32) {
    if (!accum_f32) {
        return false;
    }
    if (!run_prepared_raw(&prepared_input_scales_)) {
        return false;
    }
    const int32_t* raw = reinterpret_cast<const int32_t*>(C_mem_->virt_addr);
    for (int m = 0; m < prepared_M_; ++m) {
        const int32_t* src = raw + (size_t)m * N_;
        float* dst = accum_f32 + (size_t)m * N_;
        for (int n = 0; n < N_; ++n) {
            dst[n] += (float)src[n] * prepared_input_scales_[(size_t)m] * scales_[n];
        }
    }
    return true;
}

bool NpuLinearI4::supports_batch(int M) const {
    return M == 1 || dynamic_index_for_m(M) >= 0;
}

bool NpuLinearI4::forward_argmax(const uint16_t* input_f16, int M,
                                 int* argmax_id, uint16_t* argmax_value) {
    if (!argmax_id || M != 1 || !ctx_ || !input_f16 || K_ <= 0 ||
        K_matmul_ <= 0 || N_ <= 0) {
        return false;
    }
    uint16_t* prepared = prepare_input_f16(M);
    if (!prepared) {
        return false;
    }
    std::memcpy(prepared, input_f16, (size_t)K_ * sizeof(uint16_t));
    return forward_prepared_argmax(argmax_id, argmax_value);
}

bool NpuLinearI4::forward_prepared_argmax(int* argmax_id, uint16_t* argmax_value) {
    if (!argmax_id || !ctx_ || K_ <= 0 || K_matmul_ <= 0 || N_ <= 0) {
        return false;
    }

    const uint16_t* out = forward_prepared_output_f16();
    if (!out) {
        return false;
    }
    int best = 0;
    uint16_t best_value = out[0];
    uint16_t best_key = f16_order_key(best_value);
    for (int i = 1; i < N_; ++i) {
        const uint16_t key = f16_order_key(out[i]);
        if (key > best_key) {
            best_key = key;
            best_value = out[i];
            best = i;
        }
    }
    *argmax_id = best;
    if (argmax_value) {
        *argmax_value = best_value;
    }
    return true;
}

void NpuLinearI4::release_ac() {
    if (A_mem_) { rknn_destroy_mem(ctx_, A_mem_); A_mem_ = nullptr; }
    if (C_mem_) { rknn_destroy_mem(ctx_, C_mem_); C_mem_ = nullptr; }
    cur_M_ = 0;
    alloc_M_ = 0;
    prepared_M_ = 0;
}

void NpuLinearI4::destroy() {
    release_ac();
    if (B_mem_) { rknn_destroy_mem(ctx_, B_mem_); B_mem_ = nullptr; }
    if (ctx_) { rknn_matmul_destroy(ctx_); ctx_ = 0; }
    K_ = 0;
    K_matmul_ = 0;
    N_ = 0;
    prepared_M_ = 0;
    dynamic_shapes_.clear();
    dynamic_io_attrs_.clear();
    dynamic_m_ = false;
    dynamic_max_m_ = 1;
    scales_.clear();
    prepared_input_scales_.clear();
    prepared_input_f16_.clear();
    prepared_output_f16_.clear();
}

void NpuLinearI4::set_core_mask(rknn_core_mask mask) {
    core_mask_ = mask;
    has_core_mask_ = true;
}
