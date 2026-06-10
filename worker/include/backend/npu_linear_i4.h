#pragma once
#include "ops/op_linear.h"
#include "rknn_matmul_api.h"

#include <cstdint>
#include <string>
#include <vector>

class NpuLinearI4 : public ILinearOp {
public:
    NpuLinearI4() = default;
    ~NpuLinearI4() override { destroy(); }

    NpuLinearI4(const NpuLinearI4&) = delete;
    NpuLinearI4& operator=(const NpuLinearI4&) = delete;

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

private:
    struct KSplitChunk {
        int offset = 0;
        int K = 0;
        int K_matmul = 0;
        rknn_matmul_ctx ctx = 0;
        rknn_matmul_io_attr io_attr{};
        std::vector<rknn_matmul_shape> dynamic_shapes;
        std::vector<rknn_matmul_io_attr> dynamic_io_attrs;
        bool dynamic_m = false;
        int dynamic_max_m = 1;
        rknn_tensor_mem* A_mem = nullptr;
        rknn_tensor_mem* B_mem = nullptr;
        rknn_tensor_mem* C_mem = nullptr;
        int cur_M = 0;
        int alloc_M = 0;
        std::vector<float> scales;
    };

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
    float quantize_current_input(const uint16_t* input_f16, int8_t* input_i8);
    bool run_prepared_raw(std::vector<float>* input_scales);
    void scale_output_f16(const int32_t* raw, const float* input_scales,
                          int M, uint16_t* out) const;
    bool init_ksplit(const uint16_t* weight_kn);
    bool create_ksplit_context_and_b(KSplitChunk* chunk);
    bool bind_ksplit_b_mem(KSplitChunk* chunk);
    bool ksplit_ac_sizes(const KSplitChunk& chunk, int M,
                         uint32_t* A_size, uint32_t* C_size) const;
    int ksplit_dynamic_index_for_m(const KSplitChunk& chunk, int M) const;
    bool bind_ksplit_ac(KSplitChunk* chunk, int M, bool quiet = false);
    bool rebuild_ksplit_ac(KSplitChunk* chunk, int M);
    bool ensure_ksplit_ac(int M);
    void release_ksplit_ac(KSplitChunk* chunk);
    void destroy_ksplit();
    float quantize_ksplit_input_chunk(const uint16_t* input_f16,
                                      const KSplitChunk& chunk,
                                      uint8_t* input_i4) const;
    bool run_ksplit_accumulate(std::vector<float>* output_f32);

    int K_ = 0;
    int K_matmul_ = 0;
    int N_ = 0;
    std::string cache_key_;
    bool has_core_mask_ = false;
    rknn_core_mask core_mask_ = RKNN_NPU_CORE_AUTO;

    rknn_matmul_ctx ctx_ = 0;
    rknn_matmul_io_attr io_attr_{};
    std::vector<rknn_matmul_shape> dynamic_shapes_;
    std::vector<rknn_matmul_io_attr> dynamic_io_attrs_;
    bool dynamic_m_ = false;
    int dynamic_max_m_ = 1;
    rknn_tensor_mem* A_mem_ = nullptr;
    rknn_tensor_mem* B_mem_ = nullptr;
    rknn_tensor_mem* C_mem_ = nullptr;

    int cur_M_ = 0;
    int alloc_M_ = 0;
    int prepared_M_ = 0;
    bool ksplit_ = false;
    int ksplit_chunk_k_ = 0;
    std::vector<KSplitChunk> ksplit_chunks_;
    std::vector<float> ksplit_output_f32_;
    std::vector<float> scales_;
    std::vector<float> prepared_input_scales_;
    std::vector<uint16_t> prepared_input_f16_;
    std::vector<uint16_t> prepared_output_f16_;
};
