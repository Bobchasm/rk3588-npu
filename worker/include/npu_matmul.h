#pragma once
#include <cstdint>
#include <vector>
#include <string>

#include "rknn_matmul_api.h"

// ============================================================
// NpuLinear：对单个权重矩阵的 RKNN matmul 封装
//
// 计算：C = A * B
//   A: [M, K]  FP16 输入激活（由调用方提供）
//   B: [K, N]  FP16 权重（= 原模型权重的转置，初始化时加载）
//   C: [M, N]  FP16 输出
//
// 使用方式：
//   1. 构造时传入 K/N 和权重数据（已转置为 [K, N]）
//   2. 调用 forward(input_f16, M) -> output_f16
//   3. 析构时自动释放 RKNN 资源
// ============================================================

struct NpuLinear {
    // 权重维度
    int K = 0, N = 0;

    // RKNN 上下文及 IO 属性
    rknn_matmul_ctx     ctx    = 0;
    rknn_matmul_io_attr io_attr{};

    // RKNN 管理的内存
    rknn_tensor_mem* A_mem = nullptr;  // 每次推理重用
    rknn_tensor_mem* B_mem = nullptr;  // 权重，只初始化一次
    rknn_tensor_mem* C_mem = nullptr;  // 每次推理重用

    // 当前分配的 A/C 对应的 M（M 变化时需重新分配）
    int cur_M = 0;

    // 初始化：加载权重（已转置为 [K, N] 的 FP16 数组）
    // 内部创建 RKNN 上下文（M=1），并将权重转成 native B layout
    bool init(int K, int N, const uint16_t* weight_kn);

    // 推理：input_f16[M*K] -> output_f16[M*N]
    // M 改变时内部会重建 A/C 内存
    bool forward(const uint16_t* input_f16, int M, uint16_t* output_f16);

    // 释放所有 RKNN 资源
    void destroy();

    ~NpuLinear() { destroy(); }

    // 禁止拷贝
    NpuLinear() = default;
    NpuLinear(const NpuLinear&) = delete;
    NpuLinear& operator=(const NpuLinear&) = delete;

private:
    bool rebuild_ac(int M);
};
