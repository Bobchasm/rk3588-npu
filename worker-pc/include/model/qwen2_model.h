#pragma once

#include "backend/device_type.h"
#include "backend/gpu_kv_cache.h"
#include "core/model_config.h"
#include "model/kv_cache.h"
#include "model/transformer_layer.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// ============================================================
// Qwen2Model（PC 版）
//
// 设计目标：
// 1. 复用现有 worker 的模型组织方式，降低未来分布式对齐成本。
// 2. 当前先保证 CPU 节点可运行。
// 3. 通过 device 抽象预留 GPU 接入点，但不把模型主流程与具体后端耦死。
// ============================================================

class Qwen2Model {
public:
    struct PartitionConfig {
        int layer_begin = 0;
        int layer_end = -1;  // <0 means [layer_begin, num_hidden_layers)
        bool include_embedding = true;
        bool include_final_norm_and_head = true;
    };

    Qwen2Model();
    ~Qwen2Model();

    Qwen2Model(const Qwen2Model&) = delete;
    Qwen2Model& operator=(const Qwen2Model&) = delete;

    bool load(const std::string& model_dir,
              ComputeDevice device = ComputeDevice::kCpu);
    bool load(const std::string& model_dir,
              ComputeDevice device,
              const PartitionConfig& partition);

    void destroy();
    void reset_kv_cache();
    int forward_next_token(const std::vector<int>& tokens);
    bool forward_tokens_to_hidden(const std::vector<int>& tokens,
                                  std::vector<uint16_t>& output_f16);
    bool forward_hidden_states(const uint16_t* input_f16,
                               int seq,
                               int pos_base,
                               std::vector<uint16_t>& output_f16);
    bool forward_hidden_to_token(const uint16_t* input_f16,
                                 int seq,
                                 int pos_base,
                                 int& output_token_id);

    bool can_generate_tokens() const {
        return partition_.include_embedding && partition_.include_final_norm_and_head;
    }
    bool can_tokens_to_hidden() const {
        return partition_.include_embedding && !layers_.empty();
    }
    bool can_hidden_to_token() const {
        return !layers_.empty() && partition_.include_final_norm_and_head;
    }
    bool can_forward_hidden() const { return !layers_.empty(); }

    const Qwen2Config& config() const { return config_; }

private:
    struct ForwardScratch {
        std::vector<float> hidden;
        std::vector<float> norm_buf;
        std::vector<float> q;
        std::vector<float> k;
        std::vector<float> v;
        std::vector<float> attn_out;
        std::vector<float> gate;
        std::vector<float> up;
        std::vector<float> ffn_out;
        std::vector<float> last;
        std::vector<uint16_t> in_f16;
        std::vector<uint16_t> q_f16;
        std::vector<uint16_t> k_f16;
        std::vector<uint16_t> v_f16;
        std::vector<uint16_t> out_f16;
        std::vector<uint16_t> gate_f16;
        std::vector<uint16_t> up_f16;
        std::vector<uint16_t> ffn_in_f16;
        std::vector<uint16_t> ffn_out_f16;
        std::vector<uint16_t> lm_in;
        std::vector<uint16_t> lm_out;
    };

    void ensure_scratch(int seq);
    void apply_rope(float* q_row, float* k_row, int pos) const;
    void execute_loaded_layers(float* hidden, int seq, int pos_base);
    static void add_bias(float* x, const float* bias, int rows, int cols);
    static void add_residual(float* dst, const float* src, int n);
    static void silu_inplace(float* x, int n);

    Qwen2Config config_;
    ComputeDevice device_ = ComputeDevice::kCpu;
    PartitionConfig partition_;
    std::vector<uint16_t> embed_tokens_;
    std::vector<std::unique_ptr<TransformerLayer>> layers_;
    std::vector<float> norm_weight_;
    std::unique_ptr<ILinearOp> lm_head_;
    KVCache kv_cache_;
    GpuKvCache gpu_kv_cache_;
    ForwardScratch scratch_;
};
