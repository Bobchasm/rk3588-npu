#include "ops/op_cast.h"
#include "core/half.h"

void op_f16_to_f32(const uint16_t* src, float* dst, int n) {
    for (int i = 0; i < n; ++i) dst[i] = f16_to_f32(src[i]);
}

void op_f32_to_f16(const float* src, uint16_t* dst, int n) {
    for (int i = 0; i < n; ++i) dst[i] = f32_to_f16(src[i]);
}
