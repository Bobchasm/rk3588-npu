#include "ops/op_cast.h"
#include "core/half.h"

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

void op_bf16_to_f32(const uint16_t* src, float* dst, int n) {
    // BF16 -> FP32 的批量转换。aarch64 上直接用 NEON 扩展位模式。
    int i = 0;
#if defined(__aarch64__)
    for (; i + 4 <= n; i += 4) {
        uint16x4_t b = vld1_u16(src + i);
        uint32x4_t w = vmovl_u16(b);
        uint32x4_t x = vshlq_n_u32(w, 16);
        vst1q_f32(dst + i, vreinterpretq_f32_u32(x));
    }
#endif
    for (; i < n; ++i) dst[i] = bf16_to_f32(src[i]);
}

void op_f32_to_f16(const float* src, uint16_t* dst, int n) {
    // CPU FP32 激活进入 NPU 前的通用转换。aarch64 上用 NEON 批量转。
    int i = 0;
#if defined(__aarch64__)
    for (; i + 4 <= n; i += 4) {
        float32x4_t f = vld1q_f32(src + i);
        float16x4_t h = vcvt_f16_f32(f);
        vst1_f16(reinterpret_cast<float16_t*>(dst + i), h);
    }
#endif
    for (; i < n; ++i) dst[i] = f32_to_f16(src[i]);
}

void op_add_f16_to_f32_inplace(float* dst, const uint16_t* src, int n) {
    // NPU Linear 输出 FP16，residual/accum buffer 是 FP32；
    // 这个函数用于“转换并累加”而不是先落一个 FP32 临时数组。
    int i = 0;
#if defined(__aarch64__)
    for (; i + 4 <= n; i += 4) {
        float16x4_t h = vld1_f16(reinterpret_cast<const float16_t*>(src + i));
        float32x4_t f = vcvt_f32_f16(h);
        float32x4_t d = vld1q_f32(dst + i);
        vst1q_f32(dst + i, vaddq_f32(d, f));
    }
#endif
    for (; i < n; ++i) dst[i] += f16_to_f32(src[i]);
}
