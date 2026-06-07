#pragma once

#include "backend/cpu_linear.h"

// ============================================================
// GpuLinear: GPU 后端占位实现
//
// 当前版本主要解决“架构解耦”问题：
// - 模型层可以请求 GPU 设备
// - 设备选择逻辑与 CPU 路径分离
// - 若未来接入 CUDA / Vulkan / ROCm，只需要替换本类实现
//
// 现阶段若未提供真正 GPU 内核，则退回 CPU 计算并打印提示。
// ============================================================

class GpuLinear : public ILinearOp {
public:
    GpuLinear() = default;
    ~GpuLinear() override { destroy(); }

    GpuLinear(const GpuLinear&) = delete;
    GpuLinear& operator=(const GpuLinear&) = delete;

    bool init(int K, int N, const uint16_t* weight_kn) override;
    bool forward(const uint16_t* input_f16, int M, uint16_t* output_f16) override;
    bool supports_batch(int M) const override { return cpu_fallback_.supports_batch(M); }
    void destroy() override;

private:
    CpuLinear cpu_fallback_;
    bool warned_once_ = false;
};

