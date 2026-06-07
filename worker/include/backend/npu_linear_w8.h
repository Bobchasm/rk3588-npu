#pragma once
#include "ops/op_linear.h"
#include "rknn_matmul_api.h"

#include <cstdint>
#include <vector>

class NpuLinearW8 : public ILinearOp {
public:
    NpuLinearW8() = default;
    ~NpuLinearW8() override { destroy(); }

    NpuLinearW8(const NpuLinearW8&) = delete;
    NpuLinearW8& operator=(const NpuLinearW8&) = delete;

    bool init(int K, int N, const uint16_t* weight_kn) override;
    bool forward(const uint16_t* input_f16, int M, uint16_t* output_f16) override;
    bool forward_argmax(const uint16_t* input_f16, int M,
                        int* argmax_id, uint16_t* argmax_value = nullptr) override;
    void destroy() override;

    void set_core_mask(rknn_core_mask mask);
    static float quantize_input_row(int K, const uint16_t* input_f16, int8_t* input_i8);

private:
    static float quantize_float_row(int K, const float* input, int8_t* input_i8);

    bool ensure_ac(int M);
    bool rebuild_ac(int M);
    bool bind_ac(int M, bool quiet = false);
    void release_ac();
    void scale_output_f16(const int32_t* raw, float input_scale, int M, uint16_t* out) const;
    float quantize_current_input(const uint16_t* input_f16, int8_t* input_i8);

    int K_ = 0;         // original input width
    int K_matmul_ = 0;  // actual RKNN matmul width
    int N_ = 0;
    bool use_hadamard_ = false;
    int hadamard_block_ = 0;
    bool has_core_mask_ = false;
    rknn_core_mask core_mask_ = RKNN_NPU_CORE_AUTO;

    rknn_matmul_ctx ctx_ = 0;
    rknn_matmul_io_attr io_attr_{};
    rknn_tensor_mem* A_mem_ = nullptr;
    rknn_tensor_mem* B_mem_ = nullptr;
    rknn_tensor_mem* C_mem_ = nullptr;

    int cur_M_ = 0;
    int alloc_M_ = 0;
    std::vector<float> scales_;
    std::vector<float> hadamard_buf_;
};
