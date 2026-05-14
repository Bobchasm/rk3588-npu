#pragma once

// ============================================================
// op_elementwise: 逐元素操作
//   op_vec_add      : dst[i] += src[i]             （残差累加）
//   op_vec_add_bias : x[r, c] += bias[c]           （逐行加 bias）
// ============================================================

void op_vec_add(float* dst, const float* src, int n);
void op_vec_add_bias(float* x, const float* bias, int rows, int cols);
