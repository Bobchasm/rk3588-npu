#include "cpu_ops.h"
#include "half.h"

#include <cmath>
#include <cfloat>
#include <algorithm>
#include <cstring>

// RMSNorm
void rms_norm(const float* x, const float* weight, float* y,
              int seq, int hidden, float eps)
{
    for (int s = 0; s < seq; ++s) {
        const float* xrow = x + s * hidden;
        float*       yrow = y + s * hidden;

        // rms = sqrt(mean(x^2) + eps)
        float sum = 0.0f;
        for (int i = 0; i < hidden; ++i) sum += xrow[i] * xrow[i];
        float rms = sqrtf(sum / hidden + eps);
        float inv = 1.0f / rms;

        for (int i = 0; i < hidden; ++i)
            yrow[i] = xrow[i] * inv * weight[i];
    }
}

// RoPE
// 对单个头向量（长度 head_dim）原地应用旋转
static void rope_one_head(float* v, int head_dim, int pos, float theta) {
    for (int i = 0; i < head_dim / 2; ++i) {
        float freq = 1.0f / powf(theta, (float)(2 * i) / (float)head_dim);
        float angle = (float)pos * freq;
        float cos_a = cosf(angle);
        float sin_a = sinf(angle);
        float v0 = v[2 * i];
        float v1 = v[2 * i + 1];
        v[2 * i]     = v0 * cos_a - v1 * sin_a;
        v[2 * i + 1] = v0 * sin_a + v1 * cos_a;
    }
}

void apply_rope(float* q, float* k,
                int n_heads, int n_kv_heads, int head_dim,
                int pos, float theta)
{
    for (int h = 0; h < n_heads; ++h)
        rope_one_head(q + h * head_dim, head_dim, pos, theta);
    for (int h = 0; h < n_kv_heads; ++h)
        rope_one_head(k + h * head_dim, head_dim, pos, theta);
}

// Softmax (in-place, per row)
void softmax(float* x, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        float* row = x + r * cols;
        float  mx  = -FLT_MAX;
        for (int i = 0; i < cols; ++i) mx = std::max(mx, row[i]);
        float sum = 0.0f;
        for (int i = 0; i < cols; ++i) { row[i] = expf(row[i] - mx); sum += row[i]; }
        float inv = 1.0f / sum;
        for (int i = 0; i < cols; ++i) row[i] *= inv;
    }
}

// SiLU (in-place)
void silu(float* x, int n) {
    for (int i = 0; i < n; ++i)
        x[i] = x[i] / (1.0f + expf(-x[i]));
}

// Element-wise add
void vec_add(float* dst, const float* src, int n) {
    for (int i = 0; i < n; ++i) dst[i] += src[i];
}

// 加 bias：x[rows, cols] += bias[cols]
void vec_add_bias(float* x, const float* bias, int rows, int cols) {
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            x[r * cols + c] += bias[c];
}

// FP16 <-> FP32 批量转换
void f16_to_f32_vec(const uint16_t* src, float* dst, int n) {
    for (int i = 0; i < n; ++i) dst[i] = f16_to_f32(src[i]);
}

void f32_to_f16_vec(const float* src, uint16_t* dst, int n) {
    for (int i = 0; i < n; ++i) dst[i] = f32_to_f16(src[i]);
}
