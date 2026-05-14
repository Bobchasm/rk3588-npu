#pragma once
#include <vector>

// ============================================================
// op_sampling: 采样算子
// 目前只提供贪心；后续可在此文件里加 op_sample_topk / top_p / temperature
// ============================================================

int op_greedy_sample(const std::vector<float>& logits);
