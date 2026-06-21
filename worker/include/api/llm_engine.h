#pragma once
#include "api/generation_config.h"
#include "model/qwen2_model.h"
#include "ops/op_linear.h"

#include <functional>
#include <memory>
#include <string>
#include <vector>

// ============================================================
// LLMEngine: 上层推理引擎（token id in/out）
//
// 定位：包装模型 + KV Cache + 采样 + 生成循环
//   - 不涉及文字 tokenization（文字层面留给更上层 / Python 包装器）
//   - 本层就是目前 main.cpp 的「Prefill + Decode 循环」逻辑的 API 封装
//
// 后续扩展：
//   - vLLM-like 调度器直接调 LLMEngine
//   - 支持更多采样策略、流式输出、多请求 batch
// ============================================================

// 回调（可选）：每生成一个新 token 就触发一次，用于流式打印/上报
//   step : 当前是第几个新 token（0-based）
//   id   : 生成的 token id
//   elapsed_ms : 本步耗时
using TokenCallback = std::function<void(int step, int id, float elapsed_ms)>;

class LLMEngine {
public:
    using PartitionConfig = Qwen2Model::PartitionConfig;
    using KvState = Qwen2Model::KvState;

    LLMEngine();
    ~LLMEngine();

    LLMEngine(const LLMEngine&) = delete;
    LLMEngine& operator=(const LLMEngine&) = delete;

    // 加载模型。必须在 generate() 之前调用一次。
    bool load(const std::string& model_dir,
              LinearBackend backend = LinearBackend::NPU);
    bool load(const std::string& model_dir,
              LinearBackend backend,
              const PartitionConfig& partition);

    // 主动释放 NPU 资源（信号处理 / 提前退出场景）
    void destroy();

    // 重置会话（清空 KV Cache）。单轮对话前必须调用。
    void reset();
    KvState snapshot_kv_state() const;
    bool restore_kv_state(const KvState& state);

    // 生成：输入 token id -> 新生成的 token id 序列
    GenerationResult generate(
        const std::vector<int>& input_ids,
        const GenerationConfig& cfg,
        TokenCallback on_token = nullptr);

    bool forward_tokens_to_hidden(const std::vector<int>& input_ids,
                                  std::vector<uint16_t>& output_f16,
                                  std::string* error = nullptr);
    bool forward_hidden_states(const std::vector<uint16_t>& input_f16,
                               int seq,
                               int pos_base,
                               std::vector<uint16_t>& output_f16,
                               std::string* error = nullptr);
    bool forward_hidden_to_token(const std::vector<uint16_t>& input_f16,
                                 int seq,
                                 int pos_base,
                                 int& output_token_id,
                                 std::string* error = nullptr);
    bool supports_tokens_to_hidden() const { return model_ && model_->can_tokens_to_hidden(); }
    bool supports_token_generation() const { return model_ && model_->can_generate_tokens(); }
    bool supports_stage_forward() const { return model_ && model_->can_forward_hidden(); }
    bool supports_hidden_to_token() const { return model_ && model_->can_hidden_to_token(); }
    int hidden_size() const { return model_ ? model_->config().hidden_size : 0; }

private:
    std::unique_ptr<Qwen2Model> model_;
};
