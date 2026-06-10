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

// 同样的 RMSNorm，但直接输出 FP16。用于后面马上接 NPU Linear 的路径；
// 若输出 FP32，下一步也只是再转成 FP16。
void op_rmsnorm_to_f16(const float* x, const float* weight, uint16_t* y,
                       int seq, int hidden, float eps = 1e-6f);
