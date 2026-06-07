#pragma once

#include "api/generation_config.h"
#include "backend/device_type.h"

#include <functional>
#include <memory>
#include <string>
#include <vector>

class Qwen2Model;

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
    LLMEngine();
    ~LLMEngine();

    LLMEngine(const LLMEngine&) = delete;
    LLMEngine& operator=(const LLMEngine&) = delete;

    bool load(const std::string& model_dir,
              ComputeDevice device = ComputeDevice::kCpu);

    void destroy();
    void reset();

    GenerationResult generate(
        const std::vector<int>& input_ids,
        const GenerationConfig& cfg,
        TokenCallback on_token = nullptr);

    DeviceConfig device_config() const { return device_cfg_; }

private:
    std::unique_ptr<Qwen2Model> model_;
    DeviceConfig device_cfg_;
};

