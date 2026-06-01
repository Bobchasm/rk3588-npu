#pragma once
#include "ops/op_linear.h"
#include "rknn_matmul_api.h"
#include <cstdint>
#include <vector>

// ============================================================
// NpuLinear: 基于 rknn_matmul_api 的 Linear 实现
//
// 计算：C = A * B
//   A: [M, K] FP16 输入
//   B: [K, N] FP16 权重（初始化一次，native layout）
//   C: [M, N] FP16 输出
//
// 约束：
//   - 上下文以 M=1 创建；forward 时按需增长 A/C buffer，M 变小时复用
//   - 每个 NpuLinear 实例占用 1 个 NPU handle，析构时必须释放
// ============================================================

class NpuLinear : public ILinearOp {
public:
    NpuLinear() = default;
    ~NpuLinear() override { destroy(); }

    NpuLinear(const NpuLinear&) = delete;
    NpuLinear& operator=(const NpuLinear&) = delete;

    bool init(int K, int N, const uint16_t* weight_kn) override;
    bool forward(const uint16_t* input_f16, int M, uint16_t* output_f16) override;
    bool forward_accumulate(const uint16_t* input_f16, int M, float* accum_f32) override;
    bool forward_f32_accumulate(const float* input_f32, int M, float* accum_f32) override;
    uint16_t* prepare_input_f16(int M) override;
    bool forward_prepared(uint16_t* output_f16) override;
    const uint16_t* forward_prepared_output_f16() override;
    bool forward_prepared_accumulate(float* accum_f32) override;
    bool supports_batch(int M) const override;
    bool forward_argmax(const uint16_t* input_f16, int M,
                        int* argmax_id, uint16_t* argmax_value = nullptr) override;
    bool forward_prepared_argmax(int* argmax_id, uint16_t* argmax_value = nullptr) override;
    void destroy() override;

    // 可选：绑定单个 NPU core。必须在 init() 前调用。
    void set_core_mask(rknn_core_mask mask);

    const rknn_tensor_mem* prepared_input_mem() const;
    bool bind_external_input_f16(int M, const rknn_tensor_mem* external_mem);

private:
    bool ensure_ac(int M);
    bool rebuild_ac(int M);
    bool bind_ac(int M, bool quiet = false);
    bool ac_sizes(int M, uint32_t* A_size, uint32_t* C_size) const;
    int dynamic_index_for_m(int M) const;
    void release_ac();

    int K_ = 0, N_ = 0;
    bool has_core_mask_ = false;
    rknn_core_mask core_mask_ = RKNN_NPU_CORE_AUTO;

    rknn_matmul_ctx     ctx_ = 0;
    rknn_matmul_io_attr io_attr_{};
    std::vector<rknn_matmul_shape> dynamic_shapes_;
    std::vector<rknn_matmul_io_attr> dynamic_io_attrs_;
    bool dynamic_m_ = false;
    int dynamic_max_m_ = 1;

    rknn_tensor_mem* A_mem_ = nullptr;
    rknn_tensor_mem* B_mem_ = nullptr;
    rknn_tensor_mem* C_mem_ = nullptr;
    bool A_mem_external_ = false;
    int external_A_fd_ = -1;
    void* external_A_virt_addr_ = nullptr;
    int external_A_offset_ = 0;

    int cur_M_ = 0;
    int alloc_M_ = 0;
};
