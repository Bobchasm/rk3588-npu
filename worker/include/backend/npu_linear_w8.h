#pragma once
#include "ops/op_linear.h"
#include "rknn_matmul_api.h"

#include <cstdint>
#include <string>
#include <vector>

class NpuLinearW8 : public ILinearOp {
public:
    NpuLinearW8() = default;
    ~NpuLinearW8() override { destroy(); }

    NpuLinearW8(const NpuLinearW8&) = delete;
    NpuLinearW8& operator=(const NpuLinearW8&) = delete;

    bool init(int K, int N, const uint16_t* weight_kn) override;
    void set_cache_key(const std::string& key) override { cache_key_ = key; }
    bool init_from_cache(int K, int N) override;
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

    void set_core_mask(rknn_core_mask mask);
    static float quantize_input_row(int K, const uint16_t* input_f16, int8_t* input_i8);
    bool can_share_quantized_input() const;
    bool quantize_prepared_input();
    const rknn_tensor_mem* prepared_quantized_input_mem() const;
    const float* prepared_input_scales_data() const;
    int prepared_input_scale_count() const;
    bool bind_external_quantized_input(int M, const rknn_tensor_mem* external_mem,
                                       const float* input_scales, int input_scale_count);

private:
    static float quantize_float_row(int K, const float* input, int8_t* input_i8);

    bool configure_shape(int K, int N);
    bool create_context_and_b();
    rknn_matmul_tensor_attr current_b_attr() const;
    bool bind_b_mem();
    uint32_t cache_flags(bool dynamic_m) const;
    bool ac_sizes(int M, uint32_t* A_size, uint32_t* C_size) const;
    bool ensure_ac(int M);
    bool rebuild_ac(int M);
    bool bind_ac(int M, bool quiet = false);
    int dynamic_index_for_m(int M) const;
    void release_ac();
    void scale_output_f16(const int32_t* raw, const float* input_scales,
                          int M, uint16_t* out) const;
    void accumulate_output_f32(const int32_t* raw, const float* input_scales,
                               int M, float* accum_f32) const;
    float quantize_current_input(const uint16_t* input_f16, int8_t* input_i8);
    bool run_prepared_raw(std::vector<float>* input_scales);

    int K_ = 0;         // original input width
    int K_matmul_ = 0;  // actual RKNN matmul width
    int N_ = 0;
    std::string cache_key_;
    bool use_hadamard_ = false;
    int hadamard_block_ = 0;
    bool has_core_mask_ = false;
    rknn_core_mask core_mask_ = RKNN_NPU_CORE_AUTO;

    rknn_matmul_ctx ctx_ = 0;
    rknn_matmul_io_attr io_attr_{};
    std::vector<rknn_matmul_shape> dynamic_shapes_;
    std::vector<rknn_matmul_io_attr> dynamic_io_attrs_;
    std::vector<int> dynamic_ms_;
    bool dynamic_m_ = false;
    int dynamic_max_m_ = 1;
    rknn_tensor_mem* A_mem_ = nullptr;
    rknn_tensor_mem* B_mem_ = nullptr;
    rknn_tensor_mem* C_mem_ = nullptr;
    bool A_mem_external_ = false;
    bool quantized_input_ready_ = false;
    int external_A_fd_ = -1;
    void* external_A_virt_addr_ = nullptr;
    int external_A_offset_ = 0;

    int cur_M_ = 0;
    int alloc_M_ = 0;
    int prepared_M_ = 0;
    std::vector<float> scales_;
    std::vector<float> hadamard_buf_;
    std::vector<float> prepared_input_scales_;
    std::vector<uint16_t> prepared_input_f16_;
    std::vector<uint16_t> prepared_output_f16_;
};
