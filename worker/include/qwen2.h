#pragma once
#include "npu_matmul.h"
#include <cstdint>
#include <string>
#include <vector>
#include <memory>

// Qwen2 模型配置（Qwen1.5B）
struct Qwen2Config {
    int hidden_size          = 1536;
    int num_hidden_layers    = 28;
    int num_attention_heads  = 12;
    int num_kv_heads         = 2;
    int head_dim             = 128;   // hidden_size / num_attention_heads
    int intermediate_size    = 8960;
    int vocab_size           = 151936;
    float rms_norm_eps       = 1e-6f;
    float rope_theta         = 500000.0f;
    int max_position         = 512;
};

// 单层权重
struct TransformerLayer {
    // --- Attention ---
    std::vector<float>    input_layernorm;  // [hidden]  CPU RMSNorm weight
    NpuLinear             q_proj;           // [hidden, hidden]
    NpuLinear             k_proj;           // [hidden, kv_dim]
    NpuLinear             v_proj;           // [hidden, kv_dim]
    NpuLinear             o_proj;           // [hidden, hidden]
    std::vector<float>    q_bias;           // [hidden]  q/k/v 有 bias，o 没有
    std::vector<float>    k_bias;           // [kv_dim]
    std::vector<float>    v_bias;           // [kv_dim]

    // --- FFN ---
    std::vector<float>    post_attention_layernorm;  // [hidden]
    NpuLinear             gate_proj;  // [hidden, intermediate]
    NpuLinear             up_proj;    // [hidden, intermediate]
    NpuLinear             down_proj;  // [intermediate, hidden]

    // 禁止拷贝（NpuLinear 不可拷贝）
    TransformerLayer() = default;
    TransformerLayer(const TransformerLayer&) = delete;
    TransformerLayer& operator=(const TransformerLayer&) = delete;
};

// KV Cache（每层存所有历史 K/V，FP16）
struct KVCache {
    // k_cache[layer][pos * kv_dim + d]
    // v_cache[layer][pos * kv_dim + d]
    std::vector<std::vector<uint16_t>> k_cache;
    std::vector<std::vector<uint16_t>> v_cache;
    int capacity = 0;  // 最大 position 数
    int kv_dim   = 0;  // num_kv_heads * head_dim
    int cur_pos  = 0;  // 当前已写入的 position 数（prefill + decode 累加）
};

// 完整 Qwen2 模型
struct Qwen2Model {
    Qwen2Config config;

    // Embedding table（CPU，FP16）
    std::vector<uint16_t> embed_tokens;  // [vocab, hidden]

    // 28 层
    std::vector<std::unique_ptr<TransformerLayer>> layers;

    // 最终 norm
    std::vector<float> norm_weight;  // [hidden]

    // lm_head（与 embed_tokens 共享权重，FP16）
    // 此处复用 embed_tokens 的转置，或单独存储
    NpuLinear lm_head;  // [hidden, vocab]

    // KV Cache
    KVCache kv_cache;

    // 从 safetensors 文件加载全部权重
    bool load(const std::string& model_dir);


    // 推理一步（decode，M=1）
    // tokens 输入 token id 数组
    // 返回：下一个 token 的 logits（长度 vocab_size，FP32）
    std::vector<float> forward(const std::vector<int>& tokens);

    // 重置 KV Cache
    void reset_kv_cache();
};

// 贪心采样
int greedy_sample(const std::vector<float>& logits);
