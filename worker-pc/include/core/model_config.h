#pragma once

// ============================================================
// 模型结构参数
// 目前只描述 Qwen2 系列；新增模型时可在同目录下加 xxx_config.h
// ============================================================

struct Qwen2Config {
    int   hidden_size          = 1536;
    int   num_hidden_layers    = 28;
    int   num_attention_heads  = 12;
    int   num_kv_heads         = 2;
    int   head_dim             = 128;   // hidden_size / num_attention_heads
    int   intermediate_size    = 8960;
    int   vocab_size           = 151936;
    float rms_norm_eps         = 1e-6f;
    float rope_theta           = 1000000.0f;  // config.json: rope_theta=1000000
    int   max_position         = 32768;

    // 便捷派生量
    int kv_dim() const { return num_kv_heads * head_dim; }
};
