#pragma once
#include "core/model_config.h"
#include "model/model_source.h"
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
//   - 提供 forward()：输入 token id 序列 -> 输出最后一个位置的 logits
//   - KV Cache 位置在每次 forward 内部累加，支持 prefill + 逐步 decode
//
// 所有 Linear 通过 ILinearOp 抽象创建（可替换后端，见 op_linear.h）
// ============================================================

class Qwen2Model {
public:
    Qwen2Model();
    ~Qwen2Model();

    Qwen2Model(const Qwen2Model&) = delete;
    Qwen2Model& operator=(const Qwen2Model&) = delete;

    // 加载权重（model_dir/model.safetensors）
    bool load(const std::string& model_dir,
              LinearBackend backend = LinearBackend::NPU);

    // 主动释放所有 NPU handle（析构前也会自动调用）
    void destroy();

    // 重置 KV Cache（新对话）
    void reset_kv_cache();

    // 前向：输入 token ids，返回最后一个位置的 logits（FP32，长度 vocab_size）
    // 内部自动使用当前 kv_cache.cur_pos 作为起始位置
    std::vector<float> forward(const std::vector<int>& tokens);

    const Qwen2Config& config() const { return config_; }

private:
    ResolvedModelSource model_source_;
    Qwen2Config config_;

    std::vector<uint16_t>                           embed_tokens_;  // [vocab, hidden]
    std::vector<std::unique_ptr<TransformerLayer>>  layers_;
    std::vector<float>                              norm_weight_;   // [hidden]
    std::unique_ptr<ILinearOp>                      lm_head_;       // [hidden, vocab]

    KVCache kv_cache_;
};
