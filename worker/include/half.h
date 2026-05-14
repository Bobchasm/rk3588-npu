#pragma once
#include <cstdint>
#include <cstring>

// ---------- BF16 <-> FP32 ----------
// BF16 = 高16位的 float32（1 sign, 8 exp, 7 mant）
inline float bf16_to_f32(uint16_t b) {
    uint32_t u = (uint32_t)b << 16;
    float f; memcpy(&f, &u, sizeof(f)); return f;
}

inline uint16_t f32_to_bf16(float f) {
    uint32_t u; memcpy(&u, &f, sizeof(u));
    return (uint16_t)(u >> 16);
}

// ---------- FP16 <-> FP32 ----------
// FP16: 1 sign, 5 exp, 10 mant  (IEEE 754 half)
inline uint16_t f32_to_f16(float f) {
    uint32_t x; memcpy(&x, &f, sizeof(x));
    uint16_t sign = (uint16_t)((x >> 16) & 0x8000u);
    int32_t  exp  = (int32_t)((x >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = x & 0x7FFFFFu;
    if (exp <= 0)  return sign;            // underflow -> ±0
    if (exp >= 31) return sign | 0x7C00u;  // overflow  -> ±inf
    return sign | (uint16_t)((exp << 10) | (mant >> 13));
}

inline float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t x;
    if      (exp == 0)  x = sign | (mant << 13);               // subnormal
    else if (exp == 31) x = sign | 0x7F800000u | (mant << 13); // inf / nan
    else                x = sign | ((exp + 112u) << 23) | (mant << 13);
    float result; memcpy(&result, &x, sizeof(result)); return result;
}

// ---------- BF16 -> FP16（经 FP32 中转）----------
inline uint16_t bf16_to_f16(uint16_t b) {
    return f32_to_f16(bf16_to_f32(b));
}
