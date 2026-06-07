#pragma once

#include "backend/device_type.h"

#include <cstdint>
#include <memory>

// ============================================================
// worker-pc Linear 抽象
//
// 与板端 worker 保持相近的抽象层次：
// - 模型层只依赖 ILinearOp
// - CPU / GPU 后端各自实现
// - 后续若接 CUDA，只需替换 GPU 后端，不需要改模型主流程
// ============================================================

class ILinearOp {
public:
    virtual ~ILinearOp() = default;

    virtual bool init(int K, int N, const uint16_t* weight_kn) = 0;
    virtual bool forward(const uint16_t* input_f16, int M, uint16_t* output_f16) = 0;
    virtual bool supports_batch(int M) const { return M > 0; }
    virtual void destroy() = 0;
};

std::unique_ptr<ILinearOp> make_linear(ComputeDevice device);

