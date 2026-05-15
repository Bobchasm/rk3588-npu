#include "ops/op_rope.h"
#include <cmath>

// 对单个 head 向量（长度 head_dim）原地应用旋转
// Qwen2 使用 GPT-NeoX 半分式（half-split）旋转：
//   旋转对为 (v[i], v[i + head_dim/2])
// 对应 PyTorch transformers 中的 rotate_half + apply_rotary_pos_emb
static void rope_one_head(float* v, int head_dim, int pos, float theta) {
    const int half = head_dim / 2;
    for (int i = 0; i < half; ++i) {
        float freq  = 1.0f / std::pow(theta, (float)(2 * i) / (float)head_dim);
        float angle = (float)pos * freq;
        float cos_a = std::cos(angle);
        float sin_a = std::sin(angle);
        float v0 = v[i];        // 前半部分
        float v1 = v[i + half]; // 后半部分
        v[i]        = v0 * cos_a - v1 * sin_a;
        v[i + half] = v0 * sin_a + v1 * cos_a;
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
