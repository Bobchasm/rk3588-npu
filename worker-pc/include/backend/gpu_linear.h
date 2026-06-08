#pragma once

#include "backend/cpu_linear.h"

struct cublasContext;
using cublasHandle_t = cublasContext*;

// ============================================================
// GpuLinear: worker-pc 的 CUDA 线性层实现
//
// 设计目标：
// - 上层模型仍然只依赖 ILinearOp
// - 若本机存在可用 CUDA 设备，则线性层用 cuBLAS 跑 FP16 GEMM
// - 若 CUDA 环境不可用，则平滑退回 CPULinear，保持功能可用
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
    bool init_cuda();
    bool reserve_workspace(int M);
    bool forward_cuda(const uint16_t* input_f16, int M, uint16_t* output_f16);
    void destroy_cuda();

    int K_ = 0;
    int N_ = 0;
    int workspace_rows_ = 0;
    bool using_cuda_ = false;
    CpuLinear cpu_fallback_;
    bool warned_once_ = false;
    cublasHandle_t cublas_ = nullptr;
    void* d_weight_ = nullptr;
    void* d_input_ = nullptr;
    void* d_output_ = nullptr;
};
