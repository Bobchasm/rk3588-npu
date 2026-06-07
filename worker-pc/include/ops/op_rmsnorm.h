#pragma once

// ============================================================
// op_rmsnorm: RMSNorm（Root Mean Square LayerNorm）
//   y = x / rms(x) * weight
//   rms(x) = sqrt(mean(x_i^2) + eps)
//
// 形状：
//   x, y : [seq, hidden]
//   weight : [hidden]
// ============================================================

#include <cstdint>

void op_rmsnorm(const float* x, const float* weight, float* y,
                int seq, int hidden, float eps = 1e-6f);

// Same RMSNorm, but stores the result directly as FP16. This is for NPU-bound
// projections where the FP32 normalized activation would only be converted.
void op_rmsnorm_to_f16(const float* x, const float* weight, uint16_t* y,
                       int seq, int hidden, float eps = 1e-6f);
