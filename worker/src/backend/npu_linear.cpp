#include "backend/npu_linear.h"
#include <cstdio>
#include <cstring>

bool NpuLinear::init(int K, int N, const uint16_t* weight_kn) {
    K_ = K; N_ = N;

    // 创建 matmul 上下文（M=1 作为基础，实际 M 在 rebuild_ac 中适配）
    rknn_matmul_info info{};
    info.M             = 1;
    info.K             = K_;
    info.N             = N_;
    info.type          = RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT16;
    info.B_layout      = 1;  // native layout（性能更好）
    info.AC_layout     = 0;  // normal layout
    info.B_quant_type  = 0;
    info.AC_quant_type = 0;

    int ret = rknn_matmul_create(&ctx_, &info, &io_attr_);
    if (ret < 0) {
        std::fprintf(stderr, "[NpuLinear] rknn_matmul_create failed: %d (K=%d N=%d)\n", ret, K_, N_);
        return false;
    }

    // 分配 B 内存
    B_mem_ = rknn_create_mem(ctx_, io_attr_.B.size);
    if (!B_mem_) {
        std::fprintf(stderr, "[NpuLinear] rknn_create_mem(B) failed\n");
        rknn_matmul_destroy(ctx_);
        ctx_ = 0;
        return false;
    }

    // normal layout 权重 -> native layout
    ret = rknn_B_normal_layout_to_native_layout(
        (void*)weight_kn, B_mem_->virt_addr, K_, N_, &info);
    if (ret < 0) {
        std::fprintf(stderr, "[NpuLinear] rknn_B_normal_layout_to_native_layout failed: %d\n", ret);
        return false;
    }

    ret = rknn_matmul_set_io_mem(ctx_, B_mem_, &io_attr_.B);
    if (ret < 0) {
        std::fprintf(stderr, "[NpuLinear] rknn_matmul_set_io_mem(B) failed: %d\n", ret);
        return false;
    }

    return rebuild_ac(1);
}

bool NpuLinear::rebuild_ac(int M) {
    if (A_mem_) { rknn_destroy_mem(ctx_, A_mem_); A_mem_ = nullptr; }
    if (C_mem_) { rknn_destroy_mem(ctx_, C_mem_); C_mem_ = nullptr; }

    // FP16 = 2 bytes/elem
    uint32_t A_size = (uint32_t)(M * K_ * 2);
    uint32_t C_size = (uint32_t)(M * N_ * 2);

    A_mem_ = rknn_create_mem(ctx_, A_size);
    C_mem_ = rknn_create_mem(ctx_, C_size);
    if (!A_mem_ || !C_mem_) {
        std::fprintf(stderr, "[NpuLinear] rknn_create_mem(A/C) failed M=%d\n", M);
        return false;
    }

    rknn_matmul_tensor_attr A_attr = io_attr_.A;
    A_attr.dims[0] = M;
    A_attr.size    = A_size;

    rknn_matmul_tensor_attr C_attr = io_attr_.C;
    C_attr.dims[0] = M;
    C_attr.size    = C_size;

    int ret = rknn_matmul_set_io_mem(ctx_, A_mem_, &A_attr);
    if (ret < 0) { std::fprintf(stderr, "[NpuLinear] set A failed: %d\n", ret); return false; }

    ret = rknn_matmul_set_io_mem(ctx_, C_mem_, &C_attr);
    if (ret < 0) { std::fprintf(stderr, "[NpuLinear] set C failed: %d\n", ret); return false; }

    cur_M_ = M;
    return true;
}

bool NpuLinear::forward(const uint16_t* input_f16, int M, uint16_t* output_f16) {
    if (M != cur_M_) {
        if (!rebuild_ac(M)) return false;
    }

    std::memcpy(A_mem_->virt_addr, input_f16, (size_t)M * K_ * 2);

    int ret = rknn_matmul_run(ctx_);
    if (ret < 0) {
        std::fprintf(stderr, "[NpuLinear] rknn_matmul_run failed: %d\n", ret);
        return false;
    }

    std::memcpy(output_f16, C_mem_->virt_addr, (size_t)M * N_ * 2);
    return true;
}

void NpuLinear::destroy() {
    if (A_mem_) { rknn_destroy_mem(ctx_, A_mem_); A_mem_ = nullptr; }
    if (B_mem_) { rknn_destroy_mem(ctx_, B_mem_); B_mem_ = nullptr; }
    if (C_mem_) { rknn_destroy_mem(ctx_, C_mem_); C_mem_ = nullptr; }
    if (ctx_)   { rknn_matmul_destroy(ctx_);      ctx_    = 0;     }
    cur_M_ = 0;
}

// ---------- 工厂实现：op_linear.h 的 make_linear ----------
std::unique_ptr<ILinearOp> make_linear(LinearBackend backend) {
    switch (backend) {
        case LinearBackend::NPU:
        default:
            return std::unique_ptr<ILinearOp>(new NpuLinear());
    }
}
