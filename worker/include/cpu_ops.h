#pragma once
#include <cstdint>
#include <vector>

// ============================================================
// 所有 CPU 算子输入/输出均使用 FP32
// 在与 NPU 的边界处由调用方做 FP16 <-> FP32 转换
// ============================================================

// ---------- RMSNorm ----------
// y = x / rms(x) * weight
// x/y: [seq, hidden], weight: [hidden]
void rms_norm(const float* x, const float* weight, float* y,
              int seq, int hidden, float eps = 1e-6f);

// ---------- RoPE（旋转位置编码）----------
// 原地修改 q/k，应用 position=pos 处的旋转
// q: [n_heads, head_dim], k: [n_kv_heads, head_dim]
// 每对相邻维度旋转：(q[2i], q[2i+1]) 乘以 (cos, sin)
void apply_rope(float* q, float* k,
                int n_heads, int n_kv_heads, int head_dim,
                int pos, float theta = 500000.0f);

// ---------- Softmax（in-place）----------
// x: [rows, cols] -> 对每行做 softmax，支持 causal mask（mask到 pos+1）
void softmax(float* x, int rows, int cols);

// ---------- SiLU ----------
// y[i] = x[i] * sigmoid(x[i]) = x[i] / (1 + exp(-x[i]))
void silu(float* x, int n);

// ---------- Element-wise add ----------
void vec_add(float* dst, const float* src, int n);
void vec_add_bias(float* x, const float* bias, int rows, int cols);

// ---------- FP16 <-> FP32 批量转换 ----------
void f16_to_f32_vec(const uint16_t* src, float* dst, int n);
void f32_to_f16_vec(const float* src, uint16_t* dst, int n);
