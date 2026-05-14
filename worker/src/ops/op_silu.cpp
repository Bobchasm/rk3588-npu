#include "ops/op_silu.h"
#include <cmath>

void op_silu(float* x, int n) {
    for (int i = 0; i < n; ++i)
        x[i] = x[i] / (1.0f + std::exp(-x[i]));
}
