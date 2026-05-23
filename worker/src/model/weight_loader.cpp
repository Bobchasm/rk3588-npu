#include "model/weight_loader.h"
#include "model/model_reader.h"
#include "core/half.h"

#include <cassert>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void read_or_throw(IModelReader& reader,
                   int64_t offset,
                   size_t size,
                   void* out,
                   const std::string& what) {
    if (!reader.read_exact(offset, size, out)) {
        throw std::runtime_error(what + ": " + reader.last_error());
    }
}

int64_t get_data_base(IModelReader& reader) {
    uint64_t hdr_size = 0;
    read_or_throw(reader, 0, 8, &hdr_size, "failed to read safetensors header size");
    return static_cast<int64_t>(8 + hdr_size);
}

// ============================================================
// 极简 safetensors JSON 头解析
// 只识别顶层 object：每个 key 对应一个 {dtype, shape, data_offsets}，
// __metadata__ 字段直接跳过。
// ============================================================

const char* skip_ws(const char* p, const char* end) {
    while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) {
        ++p;
    }
    return p;
}

const char* read_string(const char* p, const char* end, std::string& out) {
    assert(*p == '"');
    ++p;
    out.clear();
    while (p < end && *p != '"') {
        if (*p == '\\') {
            ++p;
        }
        out.push_back(*p++);
    }
    if (p < end) {
        ++p;
    }
    return p;
}

const char* read_int(const char* p, const char* end, int64_t& out) {
    bool neg = false;
    if (p < end && *p == '-') {
        neg = true;
        ++p;
    }
    out = 0;
    while (p < end && *p >= '0' && *p <= '9') {
        out = out * 10 + (*p++ - '0');
    }
    if (neg) {
        out = -out;
    }
    return p;
}

const char* skip_value(const char* p, const char* end);
const char* skip_value(const char* p, const char* end) {
    p = skip_ws(p, end);
    if (p >= end) {
        return p;
    }
    if (*p == '"') {
        std::string tmp;
        return read_string(p, end, tmp);
    }
    if (*p == '{') {
        ++p;
        p = skip_ws(p, end);
        while (p < end && *p != '}') {
            std::string key;
            p = read_string(p, end, key);
            p = skip_ws(p, end);
            ++p;
            p = skip_value(p, end);
            p = skip_ws(p, end);
            if (p < end && *p == ',') {
                ++p;
            }
            p = skip_ws(p, end);
        }
        if (p < end) {
            ++p;
        }
        return p;
    }
    if (*p == '[') {
        ++p;
        p = skip_ws(p, end);
        while (p < end && *p != ']') {
            p = skip_value(p, end);
            p = skip_ws(p, end);
            if (p < end && *p == ',') {
                ++p;
            }
            p = skip_ws(p, end);
        }
        if (p < end) {
            ++p;
        }
        return p;
    }
    while (p < end && *p != ',' && *p != '}' && *p != ']') {
        ++p;
    }
    return p;
}

}  // namespace

TensorMap load_safetensors_meta(IModelReader& reader) {
    uint64_t hdr_size = 0;
    read_or_throw(reader, 0, 8, &hdr_size, "bad safetensors header");

    std::string json(hdr_size, '\0');
    read_or_throw(reader, 8, static_cast<size_t>(hdr_size), &json[0], "header read failed");

    TensorMap result;
    const char* p = json.c_str();
    const char* end = p + json.size();

    p = skip_ws(p, end);
    assert(*p == '{');
    ++p;
    p = skip_ws(p, end);

    while (p < end && *p != '}') {
        std::string key;
        p = read_string(p, end, key);
        p = skip_ws(p, end);
        assert(*p == ':');
        ++p;
        p = skip_ws(p, end);

        if (key == "__metadata__") {
            p = skip_value(p, end);
        } else {
            TensorMeta meta;
            assert(*p == '{');
            ++p;
            p = skip_ws(p, end);
            while (p < end && *p != '}') {
                std::string field;
                p = read_string(p, end, field);
                p = skip_ws(p, end);
                assert(*p == ':');
                ++p;
                p = skip_ws(p, end);

                if (field == "dtype") {
                    p = read_string(p, end, meta.dtype);
                } else if (field == "shape") {
                    assert(*p == '[');
                    ++p;
                    p = skip_ws(p, end);
                    while (p < end && *p != ']') {
                        int64_t dim = 0;
                        p = read_int(p, end, dim);
                        meta.shape.push_back(dim);
                        p = skip_ws(p, end);
                        if (*p == ',') {
                            ++p;
                        }
                        p = skip_ws(p, end);
                    }
                    ++p;
                } else if (field == "data_offsets") {
                    assert(*p == '[');
                    ++p;
                    p = skip_ws(p, end);
                    p = read_int(p, end, meta.data_begin);
                    p = skip_ws(p, end);
                    assert(*p == ',');
                    ++p;
                    p = skip_ws(p, end);
                    p = read_int(p, end, meta.data_end);
                    p = skip_ws(p, end);
                    assert(*p == ']');
                    ++p;
                } else {
                    p = skip_value(p, end);
                }

                p = skip_ws(p, end);
                if (p < end && *p == ',') {
                    ++p;
                }
                p = skip_ws(p, end);
            }
            ++p;
            result[key] = std::move(meta);
        }

        p = skip_ws(p, end);
        if (p < end && *p == ',') {
            ++p;
        }
        p = skip_ws(p, end);
    }
    return result;
}

TensorMap load_safetensors_meta(const std::string& locator) {
    return load_safetensors_meta(get_cached_model_reader(locator));
}

std::vector<uint16_t> load_tensor_f16(IModelReader& reader,
                                      const TensorMeta& meta,
                                      bool transpose) {
    const int64_t base = get_data_base(reader);
    const int64_t byte_len = meta.data_end - meta.data_begin;

    std::vector<uint8_t> raw(static_cast<size_t>(byte_len));
    read_or_throw(reader,
                  base + meta.data_begin,
                  static_cast<size_t>(byte_len),
                  raw.data(),
                  "tensor payload read failed");

    int64_t n = 1;
    for (int64_t d : meta.shape) {
        n *= d;
    }

    std::vector<uint16_t> out(static_cast<size_t>(n));
    if (meta.dtype == "BF16") {
        const uint16_t* src = reinterpret_cast<const uint16_t*>(raw.data());
        for (int64_t i = 0; i < n; ++i) {
            out[static_cast<size_t>(i)] = bf16_to_f16(src[i]);
        }
    } else if (meta.dtype == "F16") {
        std::memcpy(out.data(), raw.data(), static_cast<size_t>(n) * 2);
    } else if (meta.dtype == "F32") {
        const float* src = reinterpret_cast<const float*>(raw.data());
        for (int64_t i = 0; i < n; ++i) {
            out[static_cast<size_t>(i)] = f32_to_f16(src[i]);
        }
    } else {
        throw std::runtime_error("Unsupported dtype: " + meta.dtype);
    }

    if (transpose && meta.shape.size() == 2) {
        const int64_t rows = meta.shape[0];
        const int64_t cols = meta.shape[1];
        std::vector<uint16_t> tmp(static_cast<size_t>(n));
        for (int64_t r = 0; r < rows; ++r) {
            for (int64_t c = 0; c < cols; ++c) {
                tmp[static_cast<size_t>(c * rows + r)] =
                    out[static_cast<size_t>(r * cols + c)];
            }
        }
        out = std::move(tmp);
    }
    return out;
}

std::vector<uint16_t> load_tensor_f16(const std::string& locator,
                                      const TensorMeta& meta,
                                      bool transpose) {
    return load_tensor_f16(get_cached_model_reader(locator), meta, transpose);
}

std::vector<float> load_tensor_f32(IModelReader& reader, const TensorMeta& meta) {
    const int64_t base = get_data_base(reader);
    const int64_t byte_len = meta.data_end - meta.data_begin;

    std::vector<uint8_t> raw(static_cast<size_t>(byte_len));
    read_or_throw(reader,
                  base + meta.data_begin,
                  static_cast<size_t>(byte_len),
                  raw.data(),
                  "tensor payload read failed");

    int64_t n = 1;
    for (int64_t d : meta.shape) {
        n *= d;
    }
    std::vector<float> out(static_cast<size_t>(n));

    if (meta.dtype == "BF16") {
        const uint16_t* src = reinterpret_cast<const uint16_t*>(raw.data());
        for (int64_t i = 0; i < n; ++i) {
            out[static_cast<size_t>(i)] = bf16_to_f32(src[i]);
        }
    } else if (meta.dtype == "F16") {
        const uint16_t* src = reinterpret_cast<const uint16_t*>(raw.data());
        for (int64_t i = 0; i < n; ++i) {
            out[static_cast<size_t>(i)] = f16_to_f32(src[i]);
        }
    } else if (meta.dtype == "F32") {
        std::memcpy(out.data(), raw.data(), static_cast<size_t>(n) * 4);
    } else {
        throw std::runtime_error("Unsupported dtype: " + meta.dtype);
    }
    return out;
}

std::vector<float> load_tensor_f32(const std::string& locator, const TensorMeta& meta) {
    return load_tensor_f32(get_cached_model_reader(locator), meta);
}
