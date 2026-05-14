#pragma once
#include <cstdint>

// ============================================================
// 全局数据类型别名
// 集中放在这里，便于将来切换底层表示（比如 FP16 换成 BF16 时只改一处）
// ============================================================

namespace rk {

// FP16 在程序中以 uint16_t 承载（无原生 __fp16 假设，跨平台安全）
using f16_t = uint16_t;

// 权重目前统一按 FP16 存储
using weight_t = f16_t;

// 激活在 CPU 上用 FP32，NPU 边界处再转 FP16
using act_t = float;

} // namespace rk
