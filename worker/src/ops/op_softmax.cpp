#include "ops/op_softmax.h"
#include <cmath>
#include <cfloat>
#include <algorithm>

void op_softmax(float* x, int rows, int cols) {
    // 对每一行独立 softmax。先减去最大值提高数值稳定性，
    // attention score 中的 causal mask 会以 -1e9f 参与这里的 exp。
    for (int r = 0; r < rows; ++r) {
        float* row = x + r * cols;
        float  mx  = -FLT_MAX;
        for (int i = 0; i < cols; ++i) mx = std::max(mx, row[i]);
        float sum = 0.0f;
        for (int i = 0; i < cols; ++i) { row[i] = std::exp(row[i] - mx); sum += row[i]; }
        float inv = 1.0f / sum;
        for (int i = 0; i < cols; ++i) row[i] *= inv;
    }
}
