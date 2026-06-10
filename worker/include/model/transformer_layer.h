#pragma once
#include "ops/op_linear.h"
#include <memory>
#include <vector>

// ============================================================
// TransformerLayer: 单个 Transformer Block 的权重容器
//   - 纯数据 + 各 Linear 后端实例
//   - 不包含 forward 逻辑（forward 放在 qwen2_model.cpp 统一管理 buffer）
//
// 所有 Linear 均用 ILinearOp 抽象，可方便切换 NPU / CPU / 多核后端
// ============================================================

struct TransformerLayer {
    // --- Attention ---
    std::vector<float>          input_layernorm;   // [hidden]
    // NPU 路径默认使用 qkv_proj，把 q/k/v 三个矩阵合成一次 matmul；
    // CPU 或调试路径可以退回 q_proj/k_proj/v_proj 三个独立 matmul。
    std::unique_ptr<ILinearOp>  qkv_proj;          // optional fused [hidden, hidden + 2 * kv_dim]
    std::unique_ptr<ILinearOp>  q_proj;            // [hidden, hidden]
    std::unique_ptr<ILinearOp>  k_proj;            // [hidden, kv_dim]
    std::unique_ptr<ILinearOp>  v_proj;            // [hidden, kv_dim]
    std::unique_ptr<ILinearOp>  o_proj;            // [hidden, hidden]
    std::vector<float>          q_bias;            // [hidden]
    std::vector<float>          k_bias;            // [kv_dim]
    std::vector<float>          v_bias;            // [kv_dim]

    // --- FFN ---
    std::vector<float>          post_attention_layernorm;  // [hidden]
    // NPU 路径默认使用 gate_up_proj，一次算出 SwiGLU 的 gate 和 up。
    // 后续 CPU SiLU(gate)*up 得到 down_proj 的输入。
    std::unique_ptr<ILinearOp>  gate_up_proj; // optional fused [hidden, 2 * intermediate]
    std::unique_ptr<ILinearOp>  gate_proj;   // [hidden, intermediate]
    std::unique_ptr<ILinearOp>  up_proj;     // [hidden, intermediate]
    std::unique_ptr<ILinearOp>  down_proj;   // [intermediate, hidden]

    TransformerLayer() = default;
    TransformerLayer(const TransformerLayer&) = delete;
    TransformerLayer& operator=(const TransformerLayer&) = delete;
};
