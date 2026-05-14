#include "npu_matmul.h"
#include <cstdio>
#include <cstring>


// NpuLinear 实现
bool NpuLinear::init(int k, int n, const uint16_t* weight_kn) {
    K = k; N = n;

    // 创建 matmul 上下文（M=1 作为基础，实际 M 由 rebuild_ac 动态调整）
    rknn_matmul_info info{};
    info.M          = 1;
    info.K          = K;
    info.N          = N;
    info.type       = RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT16;
    info.B_layout   = 1;  // native layout（性能更好）
    info.AC_layout  = 0;  // normal layout
    info.B_quant_type  = 0;
    info.AC_quant_type = 0;

    int ret = rknn_matmul_create(&ctx, &info, &io_attr);
    if (ret < 0) {
        fprintf(stderr, "[NpuLinear] rknn_matmul_create failed: %d (K=%d N=%d)\n", ret, K, N);
        return false;
    }

    // 分配 B 内存
    B_mem = rknn_create_mem(ctx, io_attr.B.size);
    if (!B_mem) {
        fprintf(stderr, "[NpuLinear] rknn_create_mem(B) failed\n");
        rknn_matmul_destroy(ctx);  // 防止 ctx 泄漏
        ctx = 0;
        return false;
    }

    // 将 normal layout 的权重转换为 native layout 并写入 B_mem
    ret = rknn_B_normal_layout_to_native_layout(
        (void*)weight_kn, B_mem->virt_addr, K, N, &info);
    if (ret < 0) {
        fprintf(stderr, "[NpuLinear] rknn_B_normal_layout_to_native_layout failed: %d\n", ret);
        return false;
    }

    // 绑定 B
    ret = rknn_matmul_set_io_mem(ctx, B_mem, &io_attr.B);
    if (ret < 0) {
        fprintf(stderr, "[NpuLinear] rknn_matmul_set_io_mem(B) failed: %d\n", ret);
        return false;
    }

    // 初始化 A/C（M=1）
    return rebuild_ac(1);
}

bool NpuLinear::rebuild_ac(int M) {
    // 释放旧的 A/C（如果有）
    if (A_mem) { rknn_destroy_mem(ctx, A_mem); A_mem = nullptr; }
    if (C_mem) { rknn_destroy_mem(ctx, C_mem); C_mem = nullptr; }

    // A: [M, K]，C: [M, N]
    // 注意：io_attr 是用 M=1 创建的，大小需要手动按 M 缩放
    uint32_t A_size = (uint32_t)(M * K * 2);  // FP16 = 2 bytes/elem
    uint32_t C_size = (uint32_t)(M * N * 2);

    A_mem = rknn_create_mem(ctx, A_size);
    C_mem = rknn_create_mem(ctx, C_size);
    if (!A_mem || !C_mem) {
        fprintf(stderr, "[NpuLinear] rknn_create_mem(A/C) failed M=%d\n", M);
        return false;
    }

    // 更新 io_attr 中 A/C 的尺寸信息以匹配当前 M
    rknn_matmul_tensor_attr A_attr = io_attr.A;
    A_attr.dims[0] = M;
    A_attr.size    = A_size;

    rknn_matmul_tensor_attr C_attr = io_attr.C;
    C_attr.dims[0] = M;
    C_attr.size    = C_size;

    int ret = rknn_matmul_set_io_mem(ctx, A_mem, &A_attr);
    if (ret < 0) { fprintf(stderr, "[NpuLinear] set A failed: %d\n", ret); return false; }

    ret = rknn_matmul_set_io_mem(ctx, C_mem, &C_attr);
    if (ret < 0) { fprintf(stderr, "[NpuLinear] set C failed: %d\n", ret); return false; }

    cur_M = M;
    return true;
}

bool NpuLinear::forward(const uint16_t* input_f16, int M, uint16_t* output_f16) {
    // 若 M 改变则重建 A/C
    if (M != cur_M) {
        if (!rebuild_ac(M)) return false;
    }

    // 拷贝输入到 A
    memcpy(A_mem->virt_addr, input_f16, (size_t)M * K * 2);

    // 运行
    int ret = rknn_matmul_run(ctx);
    if (ret < 0) {
        fprintf(stderr, "[NpuLinear] rknn_matmul_run failed: %d\n", ret);
        return false;
    }

    // 读取输出
    memcpy(output_f16, C_mem->virt_addr, (size_t)M * N * 2);
    return true;
}

void NpuLinear::destroy() {
    if (A_mem) { rknn_destroy_mem(ctx, A_mem); A_mem = nullptr; }
    if (B_mem) { rknn_destroy_mem(ctx, B_mem); B_mem = nullptr; }
    if (C_mem) { rknn_destroy_mem(ctx, C_mem); C_mem = nullptr; }
    if (ctx)   { rknn_matmul_destroy(ctx);       ctx   = 0;     }
    cur_M = 0;
}
