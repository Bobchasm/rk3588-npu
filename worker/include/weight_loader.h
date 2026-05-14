#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>

// 单个权重张量的元数据
struct TensorMeta {
    std::string dtype;           // "BF16" / "F16" / "F32" 等
    std::vector<int64_t> shape;  // e.g. [1536, 1536]
    int64_t data_begin;          // 字节偏移（相对于数据区起始）
    int64_t data_end;
};

// 整个 safetensors 文件的权重表（名称 -> 元数据）
using TensorMap = std::unordered_map<std::string, TensorMeta>;

// 解析 safetensors 文件头，返回 TensorMap
// 不加载实际权重数据
TensorMap load_safetensors_meta(const std::string& path);

// 加载指定张量，以 FP16（uint16_t）返回
// 支持 BF16 / F16 / F32 输入，均转换为 FP16
// 支持可选转置（transpose=true：存储形状 [rows, cols] -> 加载为 [cols, rows]）
std::vector<uint16_t> load_tensor_f16(
    const std::string& path,
    const TensorMeta&  meta,
    bool               transpose = false
);

// 加载 bias 为 FP32
std::vector<float> load_tensor_f32(
    const std::string& path,
    const TensorMeta&  meta
);
