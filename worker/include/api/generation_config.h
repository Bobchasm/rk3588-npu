#pragma once
#include <vector>

// ============================================================
// GenerationConfig: 生成参数
//   目前贪心为主，预留后续扩展（temperature / top_k / top_p）
// ============================================================

struct GenerationConfig {
    int  max_new_tokens = 10;
    bool greedy         = true;

    // 停止 token id（遇到即停止）
    // Qwen2 默认：151645 = <|im_end|>，151643 = <|endoftext|>
    std::vector<int> stop_tokens = {151645, 151643};

    // 预留：sampling 参数
    // float temperature = 1.0f;
    // int   top_k       = 0;
    // float top_p       = 1.0f;
};

// 生成结果统计
struct GenerationResult {
    std::vector<int> output_ids;   // 新生成的 token id（不含输入）
    int   prefill_tokens = 0;
    int   decode_tokens  = 0;
    float prefill_ms     = 0.0f;
    float decode_ms      = 0.0f;
    bool  hit_stop       = false;
};
