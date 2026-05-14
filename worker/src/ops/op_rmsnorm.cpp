#include "ops/op_rmsnorm.h"
#include <cmath>

void op_rmsnorm(const float* x, const float* weight, float* y,
                int seq, int hidden, float eps)
{
    for (int s = 0; s < seq; ++s) {
        const float* xrow = x + s * hidden;
        float*       yrow = y + s * hidden;

        // rms = sqrt(mean(x^2) + eps)
        float sum = 0.0f;
        for (int i = 0; i < hidden; ++i) sum += xrow[i] * xrow[i];
        float rms = std::sqrt(sum / hidden + eps);
        float inv = 1.0f / rms;

        for (int i = 0; i < hidden; ++i)
            yrow[i] = xrow[i] * inv * weight[i];
    }
}
