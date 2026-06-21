#pragma once

#include "api/generation_config.h"
#include "backend/device_type.h"
#include "model/qwen2_model.h"

#include <functional>
#include <string>
#include <vector>

using TokenCallback = std::function<void(int step, int id, float elapsed_ms)>;

// ============================================================
// PC 版 LLMEngine
//
// 定位：
// - 与板端 worker 的 API 语义保持接近
// - 对上层暴露统一的 token 级生成接口
// - 通过 device 参数解耦 CPU / GPU 设备选择
// ============================================================

class LLMEngine {
public:
    using PartitionConfig = Qwen2Model::PartitionConfig;
    using KvState = Qwen2Model::KvState;

    LLMEngine();
    ~LLMEngine();

    LLMEngine(const LLMEngine&) = delete;
    LLMEngine& operator=(const LLMEngine&) = delete;

    bool load(const std::string& model_dir,
              ComputeDevice device = ComputeDevice::kCpu);
    bool load(const std::string& model_dir,
              ComputeDevice device,
              const PartitionConfig& partition);

    void destroy();
    void reset();
    KvState snapshot_kv_state() const;
    bool restore_kv_state(const KvState& state);

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

    DeviceConfig device_config() const { return device_cfg_; }

private:
    std::unique_ptr<Qwen2Model> model_;
    DeviceConfig device_cfg_;
};
