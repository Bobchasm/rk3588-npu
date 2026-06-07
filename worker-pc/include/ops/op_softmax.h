#pragma once

// ============================================================
// op_softmax: 按行 softmax（in-place）
//   x : [rows, cols]
// 每行独立做：先减行内最大值稳定数值，再做 exp + 归一化
// ============================================================

void op_softmax(float* x, int rows, int cols);
