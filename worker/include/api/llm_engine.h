#pragma once
#include "api/generation_config.h"
#include "ops/op_linear.h"

#include <functional>
#include <memory>
#include <string>
#include <vector>

// 前向声明，避免 header 污染
class Qwen2Model;

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
    LLMEngine();
    ~LLMEngine();

    LLMEngine(const LLMEngine&) = delete;
    LLMEngine& operator=(const LLMEngine&) = delete;

    // 加载模型。必须在 generate() 之前调用一次。
    bool load(const std::string& model_dir,
              LinearBackend backend = LinearBackend::NPU);

    // 主动释放 NPU 资源（信号处理 / 提前退出场景）
    void destroy();

    // 重置会话（清空 KV Cache）。单轮对话前必须调用。
    void reset();

    // 生成：输入 token id -> 新生成的 token id 序列
    GenerationResult generate(
        const std::vector<int>& input_ids,
        const GenerationConfig& cfg,
        TokenCallback on_token = nullptr);

private:
    std::unique_ptr<Qwen2Model> model_;
};
