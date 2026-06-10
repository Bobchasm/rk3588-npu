#include "ops/op_rmsnorm.h"
#include "core/half.h"
#include <cmath>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace {

inline float sum_squares_f32(const float* x, int n) {
    int i = 0;
    float sum = 0.0f;
#if defined(__aarch64__)
    float32x4_t acc = vdupq_n_f32(0.0f);
    for (; i + 4 <= n; i += 4) {
        float32x4_t v = vld1q_f32(x + i);
        acc = vfmaq_f32(acc, v, v);
    }
    sum = vaddvq_f32(acc);
#endif
    for (; i < n; ++i) {
        sum += x[i] * x[i];
    }
    return sum;
}

}  // namespace

void op_rmsnorm(const float* x, const float* weight, float* y,
                int seq, int hidden, float eps)
{
    for (int s = 0; s < seq; ++s) {
        const float* xrow = x + s * hidden;
        float*       yrow = y + s * hidden;

        // rms = sqrt(mean(x^2) + eps)
        float sum = sum_squares_f32(xrow, hidden);
        float rms = std::sqrt(sum / hidden + eps);
        float inv = 1.0f / rms;

        for (int i = 0; i < hidden; ++i)
            yrow[i] = xrow[i] * inv * weight[i];
    }
}

void op_rmsnorm_to_f16(const float* x, const float* weight, uint16_t* y,
                       int seq, int hidden, float eps)
{
    // NPU Linear 的 A 输入是 FP16。这个版本把 RMSNorm 结果直接写成 FP16，
    // 省掉 “FP32 临时输出 -> FP16 NPU 输入” 的一次额外遍历。
    for (int s = 0; s < seq; ++s) {
        const float* xrow = x + s * hidden;
        uint16_t*    yrow = y + s * hidden;

        float sum = sum_squares_f32(xrow, hidden);
        float rms = std::sqrt(sum / hidden + eps);
        float inv = 1.0f / rms;

        int i = 0;
#if defined(__aarch64__)
        const float32x4_t invv = vdupq_n_f32(inv);
        for (; i + 4 <= hidden; i += 4) {
            float32x4_t xv = vld1q_f32(xrow + i);
            float32x4_t wv = vld1q_f32(weight + i);
            float32x4_t yv = vmulq_f32(vmulq_f32(xv, invv), wv);
            float16x4_t hv = vcvt_f16_f32(yv);
            vst1_f16(reinterpret_cast<float16_t*>(yrow + i), hv);
        }
#endif
        for (; i < hidden; ++i) {
            yrow[i] = f32_to_f16(xrow[i] * inv * weight[i]);
        }
    }
}
