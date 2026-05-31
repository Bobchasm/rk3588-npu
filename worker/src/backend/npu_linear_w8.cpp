#include "backend/npu_linear_w8.h"

#include "core/half.h"

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

void ensure_nofile_limit_w8() {
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

inline uint16_t f16_order_key(uint16_t v) {
    return (v & 0x8000u) ? (uint16_t)~v : (uint16_t)(v ^ 0x8000u);
}

bool env_flag_enabled(const char* v) {
    return v && (std::strcmp(v, "1") == 0 ||
                 std::strcmp(v, "true") == 0 ||
                 std::strcmp(v, "TRUE") == 0 ||
                 std::strcmp(v, "on") == 0 ||
                 std::strcmp(v, "ON") == 0);
}

bool a8w8_neon_quant_enabled() {
    return env_flag_enabled(std::getenv("RKLLM_A8W8_NEON_QUANT"));
}

bool a8w8_hadamard_enabled() {
    return env_flag_enabled(std::getenv("RKLLM_A8W8_HADAMARD"));
}

bool is_power_of_two(int v) {
    return v > 0 && (v & (v - 1)) == 0;
}

int a8w8_hadamard_block(int K) {
    const char* v = std::getenv("RKLLM_A8W8_HADAMARD_BLOCK");
    if (!v || v[0] == '\0') {
        return 0;
    }
    char* end = nullptr;
    long parsed = std::strtol(v, &end, 10);
    if (end == v || *end != '\0' || parsed <= 0 || parsed > K ||
        !is_power_of_two((int)parsed) || (K % (int)parsed) != 0) {
        std::fprintf(stderr,
                     "[NpuLinearA8W8] invalid RKLLM_A8W8_HADAMARD_BLOCK=%s for K=%d, use full\n",
                     v, K);
        return 0;
    }
    return (int)parsed;
}

int next_power_of_two(int v) {
    int out = 1;
    while (out < v) out <<= 1;
    return out;
}

float hadamard_sign(int k, int K, int N) {
    uint32_t x = (uint32_t)k * 0x9e3779b1u ^
                 (uint32_t)K * 0x85ebca6bu ^
                 (uint32_t)N * 0xc2b2ae35u;
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return (x & 1u) ? -1.0f : 1.0f;
}

void fwht(float* data, int n) {
    for (int len = 1; len < n; len <<= 1) {
        for (int i = 0; i < n; i += len << 1) {
            for (int j = 0; j < len; ++j) {
                const float a = data[i + j];
                const float b = data[i + j + len];
                data[i + j] = a + b;
                data[i + j + len] = a - b;
            }
        }
    }
}

void fwht_matrix_k(float* data, int K, int N) {
    for (int len = 1; len < K; len <<= 1) {
        for (int i = 0; i < K; i += len << 1) {
            for (int j = 0; j < len; ++j) {
                float* a = data + (size_t)(i + j) * N;
                float* b = data + (size_t)(i + j + len) * N;
                int n = 0;
#if defined(__aarch64__)
                for (; n + 4 <= N; n += 4) {
                    float32x4_t va = vld1q_f32(a + n);
                    float32x4_t vb = vld1q_f32(b + n);
                    vst1q_f32(a + n, vaddq_f32(va, vb));
                    vst1q_f32(b + n, vsubq_f32(va, vb));
                }
#endif
                for (; n < N; ++n) {
                    const float va = a[n];
                    const float vb = b[n];
                    a[n] = va + vb;
                    b[n] = va - vb;
                }
            }
        }
    }
}

}  // namespace

bool NpuLinearW8::init(int K, int N, const uint16_t* weight_kn) {
    destroy();
    ensure_nofile_limit_w8();

    if (K <= 0 || N <= 0 || !weight_kn) {
        std::fprintf(stderr, "[NpuLinearW8] invalid init args K=%d N=%d\n", K, N);
        return false;
    }
    if ((N % 32) != 0) {
        std::fprintf(stderr, "[NpuLinearW8] N=%d is not aligned to 32 for A8W8\n", N);
        return false;
    }

    K_ = K;
    use_hadamard_ = a8w8_hadamard_enabled();
    hadamard_block_ = 0;
    if (use_hadamard_) {
        hadamard_block_ = a8w8_hadamard_block(K_);
        K_matmul_ = hadamard_block_ > 0 ? K_ : next_power_of_two(K_);
        if (hadamard_block_ == 0) {
            hadamard_block_ = K_matmul_;
        }
    } else {
        K_matmul_ = K_;
    }
    N_ = N;

    std::vector<int8_t> weight_i8((size_t)K_matmul_ * N_);
    scales_.assign(N_, 1.0f);
    if (use_hadamard_) {
        std::vector<float> transformed((size_t)K_matmul_ * N_, 0.0f);
        for (int k = 0; k < K_; ++k) {
            const float sign = hadamard_sign(k, K_, N_);
            float* dst = transformed.data() + (size_t)k * N_;
            const uint16_t* src = weight_kn + (size_t)k * N_;
            for (int n = 0; n < N_; ++n) {
                dst[n] = f16_to_f32(src[n]) * sign;
            }
        }
        for (int base = 0; base < K_matmul_; base += hadamard_block_) {
            fwht_matrix_k(transformed.data() + (size_t)base * N_,
                          hadamard_block_, N_);
        }
        for (int n = 0; n < N_; ++n) {
            float max_abs = 0.0f;
            for (int k = 0; k < K_matmul_; ++k) {
                max_abs = std::max(max_abs, std::fabs(transformed[(size_t)k * N_ + n]));
            }
            const float scale = max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
            scales_[n] = scale;
            const float inv = 1.0f / scale;
            for (int k = 0; k < K_matmul_; ++k) {
                int q = (int)std::lrint(transformed[(size_t)k * N_ + n] * inv);
                q = std::max(-127, std::min(127, q));
                weight_i8[(size_t)k * N_ + n] = (int8_t)q;
            }
        }
    } else {
        for (int n = 0; n < N_; ++n) {
            float max_abs = 0.0f;
            for (int k = 0; k < K_; ++k) {
                const float w = f16_to_f32(weight_kn[(size_t)k * N_ + n]);
                max_abs = std::max(max_abs, std::fabs(w));
            }
            const float scale = max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
            scales_[n] = scale;
            const float inv = 1.0f / scale;
            for (int k = 0; k < K_; ++k) {
                int q = (int)std::lrint(f16_to_f32(weight_kn[(size_t)k * N_ + n]) * inv);
                q = std::max(-127, std::min(127, q));
                weight_i8[(size_t)k * N_ + n] = (int8_t)q;
            }
        }
    }

    rknn_matmul_info info{};
    info.M = 1;
    info.K = K_matmul_;
    info.N = N_;
    info.type = RKNN_INT8_MM_INT8_TO_INT32;
    info.B_layout = 1;
    info.AC_layout = 0;
    info.B_quant_type = 0;
    info.AC_quant_type = 0;

    int ret = rknn_matmul_create(&ctx_, &info, &io_attr_);
    if (ret < 0) {
        std::fprintf(stderr, "[NpuLinearW8] rknn_matmul_create failed: %d (K=%d N=%d)\n",
                     ret, K_matmul_, N_);
        destroy();
        return false;
    }

    auto fail = [this](const char* msg, int code) {
        if (code < 0) {
            std::fprintf(stderr, "[NpuLinearW8] %s failed: %d\n", msg, code);
        } else {
            std::fprintf(stderr, "[NpuLinearW8] %s failed\n", msg);
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

    B_mem_ = rknn_create_mem(ctx_, io_attr_.B.size);
    if (!B_mem_) {
        return fail("rknn_create_mem(B)", 0);
    }
    ret = rknn_B_normal_layout_to_native_layout(weight_i8.data(), B_mem_->virt_addr,
                                                K_matmul_, N_, &info);
    if (ret < 0) {
        return fail("rknn_B_normal_layout_to_native_layout", ret);
    }
    ret = rknn_matmul_set_io_mem(ctx_, B_mem_, &io_attr_.B);
    if (ret < 0) {
        return fail("rknn_matmul_set_io_mem(B)", ret);
    }

    std::fprintf(stderr, "[NpuLinearA8W8] init K=%d", K_);
    if (use_hadamard_) {
        if (K_matmul_ != K_) {
            std::fprintf(stderr, "->%d hadamard", K_matmul_);
        } else {
            std::fprintf(stderr, " hadamard_block=%d", hadamard_block_);
        }
    }
    std::fprintf(stderr, " N=%d\n", N_);
    return true;
}

bool NpuLinearW8::rebuild_ac(int M) {
    release_ac();
    if (M != 1) return false;

    A_mem_ = rknn_create_mem(ctx_, io_attr_.A.size);
    C_mem_ = rknn_create_mem(ctx_, io_attr_.C.size);
    auto cleanup = [this]() {
        if (A_mem_) { rknn_destroy_mem(ctx_, A_mem_); A_mem_ = nullptr; }
        if (C_mem_) { rknn_destroy_mem(ctx_, C_mem_); C_mem_ = nullptr; }
        cur_M_ = 0;
        alloc_M_ = 0;
    };
    if (!A_mem_ || !C_mem_) {
        std::fprintf(stderr, "[NpuLinearW8] rknn_create_mem(A/C) failed M=%d\n", M);
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

bool NpuLinearW8::bind_ac(int M, bool quiet) {
    if (M != 1) return false;
    int ret = rknn_matmul_set_io_mem(ctx_, A_mem_, &io_attr_.A);
    if (ret < 0) {
        if (!quiet) std::fprintf(stderr, "[NpuLinearW8] set A failed: %d\n", ret);
        return false;
    }
    ret = rknn_matmul_set_io_mem(ctx_, C_mem_, &io_attr_.C);
    if (ret < 0) {
        if (!quiet) std::fprintf(stderr, "[NpuLinearW8] set C failed: %d\n", ret);
        return false;
    }
    cur_M_ = M;
    return true;
}

bool NpuLinearW8::ensure_ac(int M) {
    if (!A_mem_ || !C_mem_ || M > alloc_M_) {
        return rebuild_ac(M);
    }
    if (M != cur_M_) {
        return bind_ac(M, true) || rebuild_ac(M);
    }
    return true;
}

float NpuLinearW8::quantize_input_row(int K, const uint16_t* input_f16, int8_t* input_i8) {
#if defined(__aarch64__)
    if (a8w8_neon_quant_enabled()) {
        int k = 0;
        float32x4_t max0 = vdupq_n_f32(0.0f);
        float32x4_t max1 = vdupq_n_f32(0.0f);
        for (; k + 8 <= K; k += 8) {
            const float16x8_t h = vreinterpretq_f16_u16(vld1q_u16(input_f16 + k));
            const float32x4_t lo = vcvt_f32_f16(vget_low_f16(h));
            const float32x4_t hi = vcvt_f32_f16(vget_high_f16(h));
            max0 = vmaxq_f32(max0, vabsq_f32(lo));
            max1 = vmaxq_f32(max1, vabsq_f32(hi));
        }

        alignas(16) float max_vals[4];
        vst1q_f32(max_vals, vmaxq_f32(max0, max1));
        float max_abs = std::max(std::max(max_vals[0], max_vals[1]),
                                 std::max(max_vals[2], max_vals[3]));
        for (; k < K; ++k) {
            max_abs = std::max(max_abs, std::fabs(f16_to_f32(input_f16[k])));
        }

        const float scale = max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
        const float inv = 1.0f / scale;
        const float32x4_t inv_v = vdupq_n_f32(inv);
        const int32x4_t min_v = vdupq_n_s32(-127);
        const int32x4_t max_v = vdupq_n_s32(127);
        k = 0;
        for (; k + 8 <= K; k += 8) {
            const float16x8_t h = vreinterpretq_f16_u16(vld1q_u16(input_f16 + k));
            const float32x4_t lo = vmulq_f32(vcvt_f32_f16(vget_low_f16(h)), inv_v);
            const float32x4_t hi = vmulq_f32(vcvt_f32_f16(vget_high_f16(h)), inv_v);
            int32x4_t qi_lo = vmaxq_s32(min_v, vminq_s32(max_v, vcvtnq_s32_f32(lo)));
            int32x4_t qi_hi = vmaxq_s32(min_v, vminq_s32(max_v, vcvtnq_s32_f32(hi)));
            const int16x8_t q16 = vcombine_s16(vmovn_s32(qi_lo), vmovn_s32(qi_hi));
            vst1_s8(input_i8 + k, vmovn_s16(q16));
        }
        for (; k < K; ++k) {
            int q = (int)std::lrint(f16_to_f32(input_f16[k]) * inv);
            q = std::max(-127, std::min(127, q));
            input_i8[k] = (int8_t)q;
        }
        return scale;
    }
#endif

    float max_abs = 0.0f;
    for (int k = 0; k < K; ++k) {
        max_abs = std::max(max_abs, std::fabs(f16_to_f32(input_f16[k])));
    }
    const float scale = max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
    const float inv = 1.0f / scale;
    for (int k = 0; k < K; ++k) {
        int q = (int)std::lrint(f16_to_f32(input_f16[k]) * inv);
        q = std::max(-127, std::min(127, q));
        input_i8[k] = (int8_t)q;
    }
    return scale;
}

float NpuLinearW8::quantize_float_row(int K, const float* input, int8_t* input_i8) {
    float max_abs = 0.0f;
    for (int k = 0; k < K; ++k) {
        max_abs = std::max(max_abs, std::fabs(input[k]));
    }
    const float scale = max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
    const float inv = 1.0f / scale;
    for (int k = 0; k < K; ++k) {
        int q = (int)std::lrint(input[k] * inv);
        q = std::max(-127, std::min(127, q));
        input_i8[k] = (int8_t)q;
    }
    return scale;
}

float NpuLinearW8::quantize_current_input(const uint16_t* input_f16, int8_t* input_i8) {
    if (!use_hadamard_) {
        return quantize_input_row(K_, input_f16, input_i8);
    }
    hadamard_buf_.assign((size_t)K_matmul_, 0.0f);
    for (int k = 0; k < K_; ++k) {
        hadamard_buf_[(size_t)k] = f16_to_f32(input_f16[k]) * hadamard_sign(k, K_, N_);
    }
    for (int base = 0; base < K_matmul_; base += hadamard_block_) {
        fwht(hadamard_buf_.data() + base, hadamard_block_);
    }
    return quantize_float_row(K_matmul_, hadamard_buf_.data(), input_i8);
}

void NpuLinearW8::scale_output_f16(const int32_t* raw, float input_scale,
                                   int M, uint16_t* out) const {
    const float divisor = use_hadamard_ ? (float)hadamard_block_ : 1.0f;
    for (int m = 0; m < M; ++m) {
        const int32_t* src = raw + (size_t)m * N_;
        uint16_t* dst = out + (size_t)m * N_;
        int n = 0;
#if defined(__aarch64__)
        const float* scales = scales_.data();
        const float32x4_t input_scale_v = vdupq_n_f32(input_scale / divisor);
        for (; n + 4 <= N_; n += 4) {
            float32x4_t v = vcvtq_f32_s32(vld1q_s32(src + n));
            v = vmulq_f32(v, vld1q_f32(scales + n));
            v = vmulq_f32(v, input_scale_v);
            const float16x4_t h = vcvt_f16_f32(v);
            vst1_u16(dst + n, vreinterpret_u16_f16(h));
        }
#endif
        for (; n < N_; ++n) {
            dst[n] = f32_to_f16((float)src[n] * input_scale * scales_[n] / divisor);
        }
    }
}

bool NpuLinearW8::forward(const uint16_t* input_f16, int M, uint16_t* output_f16) {
    if (!ctx_ || !input_f16 || !output_f16 || K_ <= 0 || K_matmul_ <= 0 ||
        N_ <= 0 || M != 1) {
        return false;
    }
    if (!ensure_ac(M)) {
        return false;
    }
    const float input_scale = quantize_current_input(
        input_f16, reinterpret_cast<int8_t*>(A_mem_->virt_addr));
    const int ret = rknn_matmul_run(ctx_);
    if (ret < 0) {
        std::fprintf(stderr, "[NpuLinearW8] rknn_matmul_run failed: %d\n", ret);
        release_ac();
        return false;
    }

    scale_output_f16(reinterpret_cast<const int32_t*>(C_mem_->virt_addr),
                     input_scale, M, output_f16);
    return true;
}

bool NpuLinearW8::forward_argmax(const uint16_t* input_f16, int M,
                                 int* argmax_id, uint16_t* argmax_value) {
    if (!argmax_id || M != 1 || !ctx_ || !input_f16 || K_ <= 0 ||
        K_matmul_ <= 0 || N_ <= 0) {
        return false;
    }
    if (!ensure_ac(M)) {
        return false;
    }

    const float input_scale = quantize_current_input(
        input_f16, reinterpret_cast<int8_t*>(A_mem_->virt_addr));
    int ret = rknn_matmul_run(ctx_);
    if (ret < 0) {
        std::fprintf(stderr, "[NpuLinearW8] rknn_matmul_run failed: %d\n", ret);
        release_ac();
        return false;
    }

    const int32_t* raw = reinterpret_cast<const int32_t*>(C_mem_->virt_addr);
    int best = 0;
    float best_score = (float)raw[0] * scales_[0];
    for (int i = 1; i < N_; ++i) {
        const float score = (float)raw[i] * scales_[i];
        if (score > best_score) {
            best_score = score;
            best = i;
        }
    }
    *argmax_id = best;
    if (argmax_value) {
        const float divisor = use_hadamard_ ? (float)hadamard_block_ : 1.0f;
        *argmax_value = f32_to_f16(best_score * input_scale / divisor);
    }
    return true;
}

void NpuLinearW8::release_ac() {
    if (A_mem_) { rknn_destroy_mem(ctx_, A_mem_); A_mem_ = nullptr; }
    if (C_mem_) { rknn_destroy_mem(ctx_, C_mem_); C_mem_ = nullptr; }
    cur_M_ = 0;
    alloc_M_ = 0;
}

void NpuLinearW8::destroy() {
    release_ac();
    if (B_mem_) { rknn_destroy_mem(ctx_, B_mem_); B_mem_ = nullptr; }
    if (ctx_) { rknn_matmul_destroy(ctx_); ctx_ = 0; }
    K_ = 0;
    K_matmul_ = 0;
    N_ = 0;
    use_hadamard_ = false;
    hadamard_block_ = 0;
    scales_.clear();
    hadamard_buf_.clear();
}

void NpuLinearW8::set_core_mask(rknn_core_mask mask) {
    core_mask_ = mask;
    has_core_mask_ = true;
}
