#pragma once
#include "backend/npu_linear.h"

#include <array>
#include <cstdint>
#include <memory>
#include <string>
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
    void set_cache_key(const std::string& key) override { cache_key_ = key; }
    bool init_from_cache(int K, int N) override;
    bool forward(const uint16_t* input_f16, int M, uint16_t* output_f16) override;
    uint16_t* prepare_input_f16(int M) override;
    const uint16_t* forward_prepared_output_f16() override;
    bool forward_prepared_output_shards_f16(const uint16_t** outputs,
                                            int* offsets,
                                            int* sizes,
                                            int max_shards,
                                            int* num_shards) override;
    bool prepared_output_shards_are_gate_up_pairs() const override {
        return gate_up_pair_layout_;
    }
    bool forward_prepared_accumulate(float* accum_f32) override;
    bool supports_batch(int M) const override;
    bool forward_argmax(const uint16_t* input_f16, int M,
                        int* argmax_id, uint16_t* argmax_value = nullptr) override;
    bool forward_prepared_argmax(int* argmax_id, uint16_t* argmax_value = nullptr) override;
    void destroy() override;

    void set_allow_a8w8(bool allow) { allow_a8w8_ = allow; }
    void set_allow_a4w4(bool allow) { allow_a4w4_ = allow; }
    void set_gate_up_pair_layout(bool enabled) { gate_up_pair_layout_ = enabled; }
    void set_hybrid_cpu_shard(bool enabled, int percent);
    void set_k_split_accumulate(bool enabled, int chunk_k);
    void set_k_shard_accumulate(bool enabled) { k_shard_accumulate_enabled_ = enabled; }

private:
    static constexpr int kNumNpuShards = 3;
    static constexpr int kMaxShards = 4;
    static constexpr int kNAlign = 16;

    int K_ = 0;
    int N_ = 0;
    bool allow_a8w8_ = false;
    bool allow_a4w4_ = false;
    bool gate_up_pair_layout_ = false;
    bool hybrid_cpu_shard_enabled_ = false;
    bool k_split_accumulate_enabled_ = false;
    bool k_split_accumulate_active_ = false;
    bool k_shard_accumulate_enabled_ = false;
    bool k_shard_accumulate_active_ = false;
    int hybrid_cpu_percent_ = 0;
    int k_split_chunk_k_ = 4096;
    int shard_count_ = kNumNpuShards;
    int cpu_shard_index_ = -1;
    std::string cache_key_;
    std::vector<int> offsets_;
    std::vector<int> sizes_;
    std::vector<int> k_offsets_;
    std::vector<int> k_sizes_;
    std::vector<int> pair_offsets_;
    std::vector<int> pair_sizes_;
    std::vector<std::unique_ptr<ILinearOp>> shards_;
    std::array<std::vector<uint16_t>, kMaxShards> shard_outputs_;
    std::vector<uint16_t> prepared_output_;
    std::array<const uint16_t*, kMaxShards> last_shard_outputs_{};
    std::array<const float*, kMaxShards> last_shard_outputs_f32_{};
    std::vector<uint16_t> prepared_k_shard_input_;
    std::vector<uint16_t> k_shard_output_;
    std::vector<float> k_shard_accum_;
    uint16_t* prepared_input_ = nullptr;
    int prepared_M_ = 0;

    bool bind_shared_prepared_inputs();
    bool configure_layout(int K, int N);
    bool init_k_sharded(int K, int N, const uint16_t* weight_kn);
    bool init_k_sharded_from_cache(int K, int N);
    bool prepare_k_shard_inputs();
    bool run_k_shard_accumulate(float* accum_f32);
    bool run_prepared_output_shards();
    void clear_prepared_state();
    void copy_shard_outputs_to(uint16_t* output_f16, int M) const;
};
