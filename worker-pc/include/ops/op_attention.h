#pragma once
#include <cstdint>

// ============================================================
// op_attention: 因果 GQA 多头注意力（CPU 版）
//
// 输入：
//   q        : [seq, n_heads * head_dim]          FP32（已含 RoPE）
//   k_cache  : [total_len, n_kv_heads * head_dim] FP16（到当前层全部历史 + 新写入）
//   v_cache  : [total_len, n_kv_heads * head_dim] FP16
// 输出：
//   out      : [seq, n_heads * head_dim]          FP32
//
// 其他参数：
//   pos_base  : 当前 seq 第 0 个 query 的绝对位置（= kv_cache.cur_pos - seq 之前的值）
// 内部做：scores = q · k^T / sqrt(head_dim) → causal mask → softmax → · v
// ============================================================

void op_attention(
    const float*    q,
    const uint16_t* k_cache,
    const uint16_t* v_cache,
    float*          out,
    int seq, int total_len,
    int n_heads, int n_kv_heads, int head_dim,
    int pos_base);

// Decode 阶段专用快路径：seq 固定为 1，当前位置已经写入 KV cache，
// 因而没有 future token 需要 causal mask。数学结果与
// op_attention(..., seq=1, pos_base=total_len-1) 等价。
void op_attention_decode(
    const float*    q,
    const uint16_t* k_cache,
    const uint16_t* v_cache,
    float*          out,
    int total_len,
    int n_heads, int n_kv_heads, int head_dim);
