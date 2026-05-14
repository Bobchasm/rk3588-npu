#include "ops/op_elementwise.h"

void op_vec_add(float* dst, const float* src, int n) {
    for (int i = 0; i < n; ++i) dst[i] += src[i];
}

void op_vec_add_bias(float* x, const float* bias, int rows, int cols) {
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            x[r * cols + c] += bias[c];
}
