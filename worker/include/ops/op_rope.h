#pragma once

// ============================================================
// op_rope: Rotary Position Embedding（绝对位置）
//
// 对每个 head 中相邻两维 (v[2i], v[2i+1]) 施加旋转：
//   freq_i = 1 / theta^(2i / head_dim)
//   angle  = pos * freq_i
//   v[2i]   = v0 * cos - v1 * sin
//   v[2i+1] = v0 * sin + v1 * cos
//
// 原地修改 q, k：
//   q : [n_heads,    head_dim]
//   k : [n_kv_heads, head_dim]
// ============================================================

void op_rope(float* q, float* k,
             int n_heads, int n_kv_heads, int head_dim,
             int pos, float theta = 500000.0f);
