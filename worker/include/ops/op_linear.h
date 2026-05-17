#pragma once
#include <cstdint>
#include <memory>

// ============================================================
// op_linear: Linear 算子抽象接口
//
// 模型里所有线性变换统一通过 ILinearOp 调用，具体后端在 backend/ 下实现：
//   - NPU 单核（当前）
//   - NPU 按 N 维分片并发（lm_head 等大矩阵）
//   - NPU 多核调度（规划中）
//   - CPU fallback
//
// 约定：
//   - 权重格式为 [K, N] 的 FP16，即原 PyTorch 权重的转置结果
//   - A : [M, K] FP16
//   - C : [M, N] FP16
//   - M 可变，后端需自行处理（可能内部重建 buffer）
// ============================================================

class ILinearOp {
public:
    virtual ~ILinearOp() = default;

    // 用权重初始化后端（K, N 与权重数据）
    virtual bool init(int K, int N, const uint16_t* weight_kn) = 0;

    // 前向：input[M*K] FP16 -> output[M*N] FP16
    virtual bool forward(const uint16_t* input_f16, int M, uint16_t* output_f16) = 0;

    // 主动释放后端资源（析构前可提前触发，用于信号处理等场景）
    virtual void destroy() = 0;
};

// 线性后端类型
enum class LinearBackend {
    NPU,
    NPU_SHARDED,
    CPU,
    // 预留：NPU_MULTI_CORE 等
};

// 工厂：根据后端类型创建对应的 ILinearOp 实例
std::unique_ptr<ILinearOp> make_linear(LinearBackend backend = LinearBackend::NPU);
