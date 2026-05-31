#pragma once
#include "backend/npu_linear.h"

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

// ============================================================
// ShardedNpuLinear: 按输出 N 维切分的 NPU Linear
//
// 计算：C = A * B
//   B 被切成 3 个 [K, N_i] 分片，分别绑定 NPU core 0/1/2。
//   forward 时三个分片并发计算，最后按 N 维拼回完整输出。
//
// 主要用途：lm_head。它能把 467MB 的整块 B 内存拆成三块较小 B，
// 同时绕开 rknn_matmul_set_core_mask(0_1_2) 不支持的问题。
// ============================================================

class ShardedNpuLinear : public ILinearOp {
public:
    ShardedNpuLinear() = default;
    ~ShardedNpuLinear() override { destroy(); }

    ShardedNpuLinear(const ShardedNpuLinear&) = delete;
    ShardedNpuLinear& operator=(const ShardedNpuLinear&) = delete;

    bool init(int K, int N, const uint16_t* weight_kn) override;
    bool forward(const uint16_t* input_f16, int M, uint16_t* output_f16) override;
    bool supports_batch(int M) const override;
    bool forward_argmax(const uint16_t* input_f16, int M,
                        int* argmax_id, uint16_t* argmax_value = nullptr) override;
    void destroy() override;

    void set_allow_a8w8(bool allow) { allow_a8w8_ = allow; }

private:
    static constexpr int kNumShards = 3;
    static constexpr int kNAlign = 16;

    int K_ = 0;
    int N_ = 0;
    bool allow_a8w8_ = false;
    std::vector<int> offsets_;
    std::vector<int> sizes_;
    std::vector<std::unique_ptr<ILinearOp>> shards_;
    std::array<std::vector<uint16_t>, kNumShards> shard_outputs_;
};
