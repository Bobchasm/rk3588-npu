#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace npu_weight_cache {

// 缓存的不是 PyTorch 原始权重，而是已经转换好的 RKNN native-layout B。
// FP16 后端只需要保存 native B；A8W8 后端还要额外保存每列反量化 scale。
enum CacheKind {
    CACHE_KIND_FP16_NATIVE = 1,
    CACHE_KIND_A8W8_NATIVE = 2,
    CACHE_KIND_A4W4_NATIVE = 3,
};

struct CacheSpec {
    std::string key;        // 逻辑权重名，例如某层 qkv_fused 或某个 shard。
    uint32_t kind = 0;      // CacheKind，区分 FP16 native B 和 A8W8 native B。
    int K = 0;              // 原始矩阵输入维度。
    int N = 0;              // 原始矩阵输出维度。
    int K_matmul = 0;       // 实际交给 RKNN 的 K；A8W8 hadamard 可能 padding。
    uint32_t flags = 0;     // 后端配置标记，例如 A8W8 hadamard block。
    size_t packed_bytes = 0;// RKNN B tensor native-layout 字节数。
    size_t aux_bytes = 0;   // 附加数据字节数；A8W8 用来存 scales_。
};

// 每次加载模型时调用一次。model.safetensors 的 size/mtime 会参与目录名，
// 模型文件变化后自动进入新的缓存目录，避免误用旧权重。
void configure_for_model(const std::string& model_dir,
                         const std::string& model_path);
bool enabled();
bool load_profile_enabled();

// 轻量探测缓存文件是否存在。用于避免 cold load 时为了确认 miss 而先创建
// RKNN context/B buffer；只有存在时才进入真正 read()。
bool exists(const CacheSpec& spec);

bool read(const CacheSpec& spec,
          void* packed_dst,
          size_t packed_bytes,
          void* aux_dst = nullptr,
          size_t aux_bytes = 0);

void write(const CacheSpec& spec,
           const void* packed_src,
           size_t packed_bytes,
           const void* aux_src = nullptr,
           size_t aux_bytes = 0);

void print_summary_if_enabled();

}  // namespace npu_weight_cache
