#include "ops/op_rope.h"
#include <cmath>

// 对单个 head 向量（长度 head_dim）原地应用旋转
static void rope_one_head(float* v, int head_dim, int pos, float theta) {
    for (int i = 0; i < head_dim / 2; ++i) {
        float freq  = 1.0f / std::pow(theta, (float)(2 * i) / (float)head_dim);
        float angle = (float)pos * freq;
        float cos_a = std::cos(angle);
        float sin_a = std::sin(angle);
        float v0 = v[2 * i];
        float v1 = v[2 * i + 1];
        v[2 * i]     = v0 * cos_a - v1 * sin_a;
        v[2 * i + 1] = v0 * sin_a + v1 * cos_a;
    }
}

void op_rope(float* q, float* k,
             int n_heads, int n_kv_heads, int head_dim,
             int pos, float theta)
{
    for (int h = 0; h < n_heads; ++h)
        rope_one_head(q + h * head_dim, head_dim, pos, theta);
    for (int h = 0; h < n_kv_heads; ++h)
        rope_one_head(k + h * head_dim, head_dim, pos, theta);
}
