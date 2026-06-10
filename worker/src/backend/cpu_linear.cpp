#include "backend/cpu_linear.h"
#include "core/half.h"
#include "ops/op_cast.h"

#include <algorithm>
#include <cstdio>
#include <cstring>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

bool CpuLinear::init(int K, int N, const uint16_t* weight_kn) {
    destroy();

    if (K <= 0 || N <= 0 || !weight_kn) {
        std::fprintf(stderr, "[CpuLinear] invalid init args K=%d N=%d\n", K, N);
        return false;
    }

    K_ = K;
    N_ = N;
    if (fast_f32_weight_) {
        weight_f32_.resize((size_t)K_ * N_);
        for (size_t i = 0; i < (size_t)K_ * N_; ++i) {
            weight_f32_[i] = f16_to_f32(weight_kn[i]);
        }
        std::vector<uint16_t>().swap(weight_);
    } else {
        weight_.assign(weight_kn, weight_kn + (size_t)K_ * N_);
        std::vector<float>().swap(weight_f32_);
    }
    scratch_.resize(N_);
    return true;
}

bool CpuLinear::forward(const uint16_t* input_f16, int M, uint16_t* output_f16) {
    if (!input_f16 || !output_f16 || K_ <= 0 || N_ <= 0 || M <= 0) {
        std::fprintf(stderr, "[CpuLinear] invalid forward args M=%d K=%d N=%d\n", M, K_, N_);
        return false;
    }

    for (int m = 0; m < M; ++m) {
        std::fill(scratch_.begin(), scratch_.end(), 0.0f);

        const uint16_t* in_row = input_f16 + (size_t)m * K_;
        if (!weight_f32_.empty()) {
            for (int k = 0; k < K_; ++k) {
                const float x = f16_to_f32(in_row[k]);
                const float* w_row = weight_f32_.data() + (size_t)k * N_;
                int n = 0;
#if defined(__aarch64__)
                const float32x4_t xv = vdupq_n_f32(x);
                for (; n + 4 <= N_; n += 4) {
                    float32x4_t acc = vld1q_f32(scratch_.data() + n);
                    float32x4_t w = vld1q_f32(w_row + n);
                    acc = vfmaq_f32(acc, xv, w);
                    vst1q_f32(scratch_.data() + n, acc);
                }
#endif
                for (; n < N_; ++n) {
                    scratch_[n] += x * w_row[n];
                }
            }
        } else {
            for (int k = 0; k < K_; ++k) {
                const float x = f16_to_f32(in_row[k]);
                const uint16_t* w_row = weight_.data() + (size_t)k * N_;
                int n = 0;
#if defined(__aarch64__)
                const float32x4_t xv = vdupq_n_f32(x);
                for (; n + 4 <= N_; n += 4) {
                    float32x4_t acc = vld1q_f32(scratch_.data() + n);
                    float16x4_t wh = vld1_f16(reinterpret_cast<const float16_t*>(w_row + n));
                    float32x4_t w = vcvt_f32_f16(wh);
                    acc = vfmaq_f32(acc, xv, w);
                    vst1q_f32(scratch_.data() + n, acc);
                }
#endif
                for (; n < N_; ++n) {
                    scratch_[n] += x * f16_to_f32(w_row[n]);
                }
            }
        }

        uint16_t* out_row = output_f16 + (size_t)m * N_;
        int n = 0;
#if defined(__aarch64__)
        for (; n + 4 <= N_; n += 4) {
            float32x4_t v = vld1q_f32(scratch_.data() + n);
            float16x4_t h = vcvt_f16_f32(v);
            vst1_f16(reinterpret_cast<float16_t*>(out_row + n), h);
        }
#endif
        for (; n < N_; ++n) {
            out_row[n] = f32_to_f16(scratch_[n]);
        }
    }

    return true;
}

uint16_t* CpuLinear::prepare_input_f16(int M) {
    prepared_M_ = 0;
    if (K_ <= 0 || N_ <= 0 || M <= 0) {
        return nullptr;
    }
    prepared_input_.resize((size_t)M * K_);
    prepared_M_ = M;
    return prepared_input_.data();
}

bool CpuLinear::forward_prepared(uint16_t* output_f16) {
    if (!output_f16 || prepared_M_ <= 0 || prepared_input_.empty()) {
        return false;
    }
    return forward(prepared_input_.data(), prepared_M_, output_f16);
}

const uint16_t* CpuLinear::forward_prepared_output_f16() {
    if (prepared_M_ <= 0 || prepared_input_.empty()) {
        return nullptr;
    }
    prepared_output_.resize((size_t)prepared_M_ * N_);
    if (!forward(prepared_input_.data(), prepared_M_, prepared_output_.data())) {
        return nullptr;
    }
    return prepared_output_.data();
}

bool CpuLinear::forward_prepared_accumulate(float* accum_f32) {
    if (!accum_f32) {
        return false;
    }
    const uint16_t* out = forward_prepared_output_f16();
    if (!out) {
        return false;
    }
    op_add_f16_to_f32_inplace(accum_f32, out, prepared_M_ * N_);
    return true;
}

void CpuLinear::destroy() {
    std::vector<uint16_t>().swap(weight_);
    std::vector<float>().swap(weight_f32_);
    std::vector<float>().swap(scratch_);
    std::vector<uint16_t>().swap(prepared_input_);
    std::vector<uint16_t>().swap(prepared_output_);
    prepared_M_ = 0;
    K_ = 0;
    N_ = 0;
}
