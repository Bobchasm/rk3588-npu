#pragma once
#include "model/model_reader.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

// ============================================================
// safetensors 解析器（无第三方依赖）
//
// 文件布局：
//   [0, 8)          : uint64 header_size
//   [8, 8+hsize)    : JSON 头
//   [8+hsize, ...)  : 所有张量的二进制数据（连续）
// ============================================================

struct TensorMeta {
    std::string dtype;           // "BF16" / "F16" / "F32"
    std::vector<int64_t> shape;
    int64_t data_begin;          // 相对数据区起始字节
    int64_t data_end;
};

using TensorMap = std::unordered_map<std::string, TensorMeta>;

// 只解析文件头，不读取张量数据
TensorMap load_safetensors_meta(IModelReader& reader);
TensorMap load_safetensors_meta(const std::string& locator);

// 读取指定张量，统一转为 FP16 (uint16_t)
// transpose=true 时：[rows, cols] 按 [cols, rows] 存储（用于把 PyTorch 权重转 K×N）
std::vector<uint16_t> load_tensor_f16(
    IModelReader&       reader,
    const TensorMeta&  meta,
    bool               transpose = false);
std::vector<uint16_t> load_tensor_f16(
    const std::string& locator,
    const TensorMeta&  meta,
    bool               transpose = false);

// 读取指定张量为 FP32（用于 bias / norm weight）
std::vector<float> load_tensor_f32(
    IModelReader&      reader,
    const TensorMeta&  meta);
std::vector<float> load_tensor_f32(
    const std::string& locator,
    const TensorMeta&  meta);
