#pragma once
#include "ops/op_linear.h"

#include <cstdint>
#include <vector>

// ============================================================
// CpuLinear: 纯 CPU Linear fallback
//
// 计算：C = A * B
//   A: [M, K] FP16 输入
//   B: [K, N] FP16 权重
//   C: [M, N] FP16 输出
//
// 主要用途：lm_head fallback，避免 467MB 权重占用 NPU/CMA 连续内存。
// ============================================================

class CpuLinear : public ILinearOp {
public:
    CpuLinear() = default;
    ~CpuLinear() override { destroy(); }

    CpuLinear(const CpuLinear&) = delete;
    CpuLinear& operator=(const CpuLinear&) = delete;

    bool init(int K, int N, const uint16_t* weight_kn) override;
    bool forward(const uint16_t* input_f16, int M, uint16_t* output_f16) override;
    bool supports_batch(int M) const override { return M > 0; }
    void destroy() override;

private:
    int K_ = 0;
    int N_ = 0;
    std::vector<uint16_t> weight_;   // [K, N]
    std::vector<float> scratch_;     // [N], reused per row
};
