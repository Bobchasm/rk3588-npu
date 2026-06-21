#pragma once
#include "core/model_config.h"
#include "model/transformer_layer.h"
#include "model/kv_cache.h"
#include "ops/op_linear.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// ============================================================
// Qwen2Model: 模型组织层
//
// 职责：
//   - 从 safetensors 加载全部权重、分配 KV Cache
//   - 提供 forward_next_token()：输入 token id 序列 -> 输出 greedy token
//   - KV Cache 位置在每次 forward 内部累加，支持 prefill + 逐步 decode
//
// 所有 Linear 通过 ILinearOp 抽象创建（可替换后端，见 op_linear.h）
// ============================================================

class Qwen2Model {
public:
    struct KvState {
        KVCache::State kv_cache;
    };

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

    // 加载权重（model_dir/model.safetensors）
    bool load(const std::string& model_dir,
              LinearBackend backend = LinearBackend::NPU);
    bool load(const std::string& model_dir,
              LinearBackend backend,
              const PartitionConfig& partition);

    // 主动释放所有 NPU handle（析构前也会自动调用）
    void destroy();

    // 重置 KV Cache（新对话）
    void reset_kv_cache();
    KvState snapshot_kv_state() const;
    bool restore_kv_state(const KvState& state);

    // 前向：输入 token ids，返回最后一个位置的 greedy token
    // 内部自动使用当前 kv_cache.cur_pos 作为起始位置
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
        std::vector<float> q;
        std::vector<float> k;
        std::vector<float> v;
        std::vector<float> attn_out;
        std::vector<float> last;
        std::vector<uint16_t> npu_in;
        std::vector<uint16_t> npu_out;
        std::vector<uint16_t> qkv_f16;
        std::vector<uint16_t> q_f16;
        std::vector<uint16_t> k_f16;
        std::vector<uint16_t> v_f16;
        std::vector<uint16_t> gate_up_f16;
        std::vector<uint16_t> gate_f16;
        std::vector<uint16_t> up_f16;
        std::vector<uint16_t> ffn_in_f16;
        std::vector<uint16_t> ffn_out_f16;
        std::vector<uint16_t> lm_in;
    };

    int forward_internal(const std::vector<int>& tokens);
    void execute_loaded_layers(float* hidden, int seq, int pos_base);
    void ensure_rope_cache(int required_positions);

    Qwen2Config config_;

    std::vector<uint16_t>                           embed_tokens_;  // [vocab, hidden]
    std::vector<std::unique_ptr<TransformerLayer>>  layers_;
    std::vector<float>                              norm_weight_;   // [hidden]
    std::unique_ptr<ILinearOp>                      lm_head_;       // [hidden, vocab]

    KVCache kv_cache_;
    ForwardScratch scratch_;
    PartitionConfig partition_;
    std::vector<float> rope_cos_;
    std::vector<float> rope_sin_;
    int rope_cached_positions_ = 0;
    int rope_cached_head_dim_ = 0;
    float rope_cached_theta_ = 0.0f;
};
