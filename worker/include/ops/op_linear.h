#pragma once
#include <cstdint>
#include <memory>

// ============================================================
// op_linear: Linear 算子抽象接口
//
// 模型里所有线性变换统一通过 ILinearOp 调用，具体后端在 backend/ 下实现：
//   - NPU 自动规划（根据矩阵规模选择单核或分片）
//   - NPU 单核
//   - NPU 按 N 维分片并发（lm_head 等大矩阵）
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

    // 可选快路径：FP16 input[M*K] -> FP16 output[M*N]，并直接累加到 accum[M*N] FP32。
    // 默认不支持，调用方应 fallback 到 forward() + add。
    virtual bool forward_accumulate(const uint16_t* input_f16, int M, float* accum_f32) {
        (void)input_f16;
        (void)M;
        (void)accum_f32;
        return false;
    }

    // 可选快路径：FP32 input[M*K] 直接写入后端 A buffer，再把 output 累加到 accum[M*N]。
    virtual bool forward_f32_accumulate(const float* input_f32, int M, float* accum_f32) {
        (void)input_f32;
        (void)M;
        (void)accum_f32;
        return false;
    }

    virtual bool supports_batch(int M) const {
        return M == 1;
    }

    // 可选快路径：M=1 时直接返回 output argmax，避免 lm_head 大输出拼接。
    virtual bool forward_argmax(const uint16_t* input_f16, int M,
                                int* argmax_id, uint16_t* argmax_value = nullptr) {
        (void)input_f16;
        (void)M;
        (void)argmax_id;
        (void)argmax_value;
        return false;
    }

    // 主动释放后端资源（析构前可提前触发，用于信号处理等场景）
    virtual void destroy() = 0;
};

// 线性后端类型
enum class LinearBackend {
    NPU,          // 自动规划：大矩阵使用三核分片，小矩阵使用单核
    NPU_SINGLE,   // 强制单核 NPU
    NPU_SHARDED,  // 强制三核 NPU 分片
    CPU,
};

// 工厂：根据后端类型创建对应的 ILinearOp 实例。
// layer_idx/role 用于按层量化筛选。
std::unique_ptr<ILinearOp> make_linear(LinearBackend backend,
                                       int layer_idx,
                                       const char* role);
