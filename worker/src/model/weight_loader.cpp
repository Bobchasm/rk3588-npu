#include "model/weight_loader.h"
#include "core/half.h"
#include "ops/op_cast.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>

// ============================================================
// 极简 safetensors JSON 头解析
// 只识别顶层 object：每个 key 对应一个 {dtype, shape, data_offsets}，
// __metadata__ 字段直接跳过。
// ============================================================

static const char* skip_ws(const char* p, const char* end) {
    while (p < end && (*p==' '||*p=='\t'||*p=='\n'||*p=='\r')) ++p;
    return p;
}

static const char* read_string(const char* p, const char* end, std::string& out) {
    assert(*p == '"');
    ++p;
    out.clear();
    while (p < end && *p != '"') {
        if (*p == '\\') { ++p; }  // 简单跳过转义
        out.push_back(*p++);
    }
    if (p < end) ++p;
    return p;
}

static const char* read_int(const char* p, const char* end, int64_t& out) {
    bool neg = false;
    if (p < end && *p == '-') { neg = true; ++p; }
    out = 0;
    while (p < end && *p >= '0' && *p <= '9') {
        out = out * 10 + (*p++ - '0');
    }
    if (neg) out = -out;
    return p;
}

static const char* skip_value(const char* p, const char* end);
static const char* skip_value(const char* p, const char* end) {
    p = skip_ws(p, end);
    if (p >= end) return p;
    if (*p == '"') { std::string t; return read_string(p, end, t); }
    if (*p == '{') {
        ++p; p = skip_ws(p, end);
        while (p < end && *p != '}') {
            std::string k; p = read_string(p, end, k);
            p = skip_ws(p, end); ++p;   // ':'
            p = skip_value(p, end);
            p = skip_ws(p, end);
            if (p < end && *p == ',') ++p;
            p = skip_ws(p, end);
        }
        if (p < end) ++p;
        return p;
    }
    if (*p == '[') {
        ++p; p = skip_ws(p, end);
        while (p < end && *p != ']') {
            p = skip_value(p, end);
            p = skip_ws(p, end);
            if (p < end && *p == ',') ++p;
            p = skip_ws(p, end);
        }
        if (p < end) ++p;
        return p;
    }
    // number / true / false / null
    while (p < end && *p != ',' && *p != '}' && *p != ']') ++p;
    return p;
}

TensorMap load_safetensors_meta(const std::string& path) {
    // safetensors 的 header 是 JSON，真实 tensor 数据在 header 后连续存放。
    // 这里只读 header，构建 key -> TensorMeta 的索引，避免加载阶段一开始
    // 就把所有权重读入内存。
    FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) throw std::runtime_error("Cannot open: " + path);

    uint64_t hdr_size = 0;
    if (std::fread(&hdr_size, 1, 8, fp) != 8) {
        std::fclose(fp); throw std::runtime_error("Bad safetensors header");
    }

    std::string json(hdr_size, '\0');
    if (std::fread(&json[0], 1, hdr_size, fp) != hdr_size) {
        std::fclose(fp); throw std::runtime_error("Header read failed");
    }
    std::fclose(fp);

    TensorMap result;
    const char* p   = json.c_str();
    const char* end = p + json.size();

    p = skip_ws(p, end); assert(*p == '{'); ++p;
    p = skip_ws(p, end);

    while (p < end && *p != '}') {
        std::string key;
        p = read_string(p, end, key);
        p = skip_ws(p, end); assert(*p == ':'); ++p;
        p = skip_ws(p, end);

        if (key == "__metadata__") {
            p = skip_value(p, end);
        } else {
            TensorMeta meta;
            assert(*p == '{'); ++p;
            p = skip_ws(p, end);
            while (p < end && *p != '}') {
                std::string field;
                p = read_string(p, end, field);
                p = skip_ws(p, end); assert(*p == ':'); ++p;
                p = skip_ws(p, end);

                if (field == "dtype") {
                    p = read_string(p, end, meta.dtype);
                } else if (field == "shape") {
                    assert(*p == '['); ++p; p = skip_ws(p, end);
                    while (p < end && *p != ']') {
                        int64_t dim; p = read_int(p, end, dim);
                        meta.shape.push_back(dim);
                        p = skip_ws(p, end);
                        if (*p == ',') ++p;
                        p = skip_ws(p, end);
                    }
                    ++p;
                } else if (field == "data_offsets") {
                    assert(*p == '['); ++p; p = skip_ws(p, end);
                    p = read_int(p, end, meta.data_begin);
                    p = skip_ws(p, end); assert(*p == ','); ++p;
                    p = skip_ws(p, end);
                    p = read_int(p, end, meta.data_end);
                    p = skip_ws(p, end); assert(*p == ']'); ++p;
                } else {
                    p = skip_value(p, end);
                }

                p = skip_ws(p, end);
                if (p < end && *p == ',') ++p;
                p = skip_ws(p, end);
            }
            ++p;
            result[key] = std::move(meta);
        }

        p = skip_ws(p, end);
        if (p < end && *p == ',') ++p;
        p = skip_ws(p, end);
    }
    return result;
}

// data 区起始 = 8 + hdr_size
static int64_t get_data_base(const std::string& path) {
    FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) throw std::runtime_error("Cannot open: " + path);
    uint64_t hdr_size = 0;
    if (std::fread(&hdr_size, 1, 8, fp) != 8) {
        std::fclose(fp);
        throw std::runtime_error("Bad safetensors header");
    }
    std::fclose(fp);
    return (int64_t)(8 + hdr_size);
}

std::vector<uint16_t> load_tensor_f16(
    const std::string& path, const TensorMeta& meta, bool transpose)
{
    // 每次只按 TensorMeta 的 data_offsets 读取一个 tensor。Linear 权重会在
    // 调用处设置 transpose=true，把 PyTorch [out, in] 转成本项目统一的 [K, N]。
    int64_t base     = get_data_base(path);
    int64_t byte_len = meta.data_end - meta.data_begin;

    FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) throw std::runtime_error("Cannot open: " + path);
    std::fseek(fp, (long)(base + meta.data_begin), SEEK_SET);

    std::vector<uint8_t> raw(byte_len);
    if ((int64_t)std::fread(raw.data(), 1, byte_len, fp) != byte_len) {
        std::fclose(fp); throw std::runtime_error("Read failed: " + meta.dtype);
    }
    std::fclose(fp);

    int64_t n = 1;
    for (int64_t d : meta.shape) n *= d;

    std::vector<uint16_t> out(n);
    if (meta.dtype == "BF16") {
        const uint16_t* src = reinterpret_cast<const uint16_t*>(raw.data());
        for (int64_t i = 0; i < n; ++i) out[i] = bf16_to_f16(src[i]);
    } else if (meta.dtype == "F16") {
        std::memcpy(out.data(), raw.data(), n * 2);
    } else if (meta.dtype == "F32") {
        const float* src = reinterpret_cast<const float*>(raw.data());
        for (int64_t i = 0; i < n; ++i) out[i] = f32_to_f16(src[i]);
    } else {
        throw std::runtime_error("Unsupported dtype: " + meta.dtype);
    }

    if (transpose && meta.shape.size() == 2) {
        // PyTorch Linear weight 通常是 [out_features, in_features]。
        // 本项目所有 Linear 后端统一使用 A[M,K] * B[K,N]，因此加载时转置。
        int64_t rows = meta.shape[0], cols = meta.shape[1];
        std::vector<uint16_t> tmp(n);
        for (int64_t r = 0; r < rows; ++r)
            for (int64_t c = 0; c < cols; ++c)
                tmp[c * rows + r] = out[r * cols + c];
        out = std::move(tmp);
    }
    return out;
}

std::vector<float> load_tensor_f32(const std::string& path, const TensorMeta& meta) {
    int64_t base     = get_data_base(path);
    int64_t byte_len = meta.data_end - meta.data_begin;

    FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) throw std::runtime_error("Cannot open: " + path);
    std::fseek(fp, (long)(base + meta.data_begin), SEEK_SET);

    std::vector<uint8_t> raw(byte_len);
    if ((int64_t)std::fread(raw.data(), 1, byte_len, fp) != byte_len) {
        std::fclose(fp);
        throw std::runtime_error("Read failed: " + meta.dtype);
    }
    std::fclose(fp);

    int64_t n = 1;
    for (int64_t d : meta.shape) n *= d;
    std::vector<float> out(n);

    if (meta.dtype == "BF16") {
        if (n > std::numeric_limits<int>::max()) {
            throw std::runtime_error("Tensor too large for BF16->FP32 cast");
        }
        const uint16_t* src = reinterpret_cast<const uint16_t*>(raw.data());
        op_bf16_to_f32(src, out.data(), (int)n);
    } else if (meta.dtype == "F16") {
        const uint16_t* src = reinterpret_cast<const uint16_t*>(raw.data());
        for (int64_t i = 0; i < n; ++i) out[i] = f16_to_f32(src[i]);
    } else if (meta.dtype == "F32") {
        std::memcpy(out.data(), raw.data(), n * 4);
    } else {
        throw std::runtime_error("Unsupported dtype: " + meta.dtype);
    }
    return out;
}
