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

void op_rmsnorm(const float* x, const float* weight, float* y,
                int seq, int hidden, float eps = 1e-6f);
