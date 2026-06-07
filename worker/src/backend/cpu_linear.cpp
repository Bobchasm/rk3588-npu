#include "backend/cpu_linear.h"
#include "core/half.h"

#include <algorithm>
#include <cstdio>

bool CpuLinear::init(int K, int N, const uint16_t* weight_kn) {
    destroy();

    if (K <= 0 || N <= 0 || !weight_kn) {
        std::fprintf(stderr, "[CpuLinear] invalid init args K=%d N=%d\n", K, N);
        return false;
    }

    K_ = K;
    N_ = N;
    weight_.assign(weight_kn, weight_kn + (size_t)K_ * N_);
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
        for (int k = 0; k < K_; ++k) {
            const float x = f16_to_f32(in_row[k]);
            const uint16_t* w_row = weight_.data() + (size_t)k * N_;
            for (int n = 0; n < N_; ++n) {
                scratch_[n] += x * f16_to_f32(w_row[n]);
            }
        }

        uint16_t* out_row = output_f16 + (size_t)m * N_;
        for (int n = 0; n < N_; ++n) {
            out_row[n] = f32_to_f16(scratch_[n]);
        }
    }

    return true;
}

void CpuLinear::destroy() {
    std::vector<uint16_t>().swap(weight_);
    std::vector<float>().swap(scratch_);
    K_ = 0;
    N_ = 0;
}
