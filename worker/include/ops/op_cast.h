#pragma once
#include <cstdint>

// ============================================================
// op_cast: 批量 FP32 -> FP16 转换
// NPU 边界处大量使用；单独成文件便于后续做向量化加速
// ============================================================

void op_bf16_to_f32(const uint16_t* src, float* dst, int n);
void op_f32_to_f16(const float* src, uint16_t* dst, int n);
void op_add_f16_to_f32_inplace(float* dst, const uint16_t* src, int n);
