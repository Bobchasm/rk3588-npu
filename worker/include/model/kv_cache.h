#pragma once
#include <cstdint>
#include <memory>
#include <vector>

// ============================================================
// KVCache: 每层存放所有历史 K/V（FP16）
//
// 当前实现为最朴素的连续张量：
//   k_cache_[layer]: [capacity * kv_dim]
//   v_cache_[layer]: [capacity * kv_dim]
//
// 后续可以替换为：
//   - PagedKVCache（类似 vLLM 的 block 管理）
//   - Compressed/Quantized KVCache
//   - Windowed / Sliding KVCache
// 只要保持同样的读写接口即可。
// ============================================================

class KVCache {
public:
    void init(int num_layers, int capacity, int kv_dim);
    void reset();

    int  cur_pos()  const { return cur_pos_; }
    void set_cur_pos(int pos) { cur_pos_ = pos; }
    int  capacity() const { return capacity_; }
    int  kv_dim()   const { return kv_dim_; }

    // 直接暴露每层底层指针，attention 算子和写入路径都通过它访问。
    // 调用方负责按 [position, kv_dim] 计算偏移。
    uint16_t*       k_ptr(int layer)       { return k_cache_[layer].get(); }
    uint16_t*       v_ptr(int layer)       { return v_cache_[layer].get(); }
    const uint16_t* k_ptr(int layer) const { return k_cache_[layer].get(); }
    const uint16_t* v_ptr(int layer) const { return v_cache_[layer].get(); }

private:
    std::vector<std::unique_ptr<uint16_t[]>> k_cache_;
    std::vector<std::unique_ptr<uint16_t[]>> v_cache_;
    int capacity_ = 0;
    int kv_dim_   = 0;
    int cur_pos_  = 0;
};
