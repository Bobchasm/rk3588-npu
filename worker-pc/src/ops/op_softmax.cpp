#include "ops/op_softmax.h"
#include <cmath>
#include <cfloat>
#include <algorithm>

void op_softmax(float* x, int rows, int cols) {
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
