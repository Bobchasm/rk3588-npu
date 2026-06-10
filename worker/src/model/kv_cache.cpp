#include "model/kv_cache.h"
#include <cstddef>

void KVCache::init(int num_layers, int capacity, int kv_dim) {
    // 每层各一块 K/V 连续数组。索引方式：
    //   k_cache_[layer][position * kv_dim + d]
    // 这里一次性分配最大上下文，避免 decode 时反复扩容导致指针失效。
    // 不做清零：cur_pos 控制可读前缀，未写入的位置不会被 attention 读取。
    capacity_ = capacity;
    kv_dim_   = kv_dim;
    cur_pos_  = 0;
    k_cache_.clear();
    v_cache_.clear();
    k_cache_.reserve(num_layers);
    v_cache_.reserve(num_layers);
    const size_t elems = (size_t)capacity * kv_dim;
    for (int i = 0; i < num_layers; ++i) {
        k_cache_.emplace_back(new uint16_t[elems]);
        v_cache_.emplace_back(new uint16_t[elems]);
    }
}

void KVCache::reset() {
    // cur_pos 控制 attention 可读前缀。旧数据虽然还在内存里，但 pos 归零后
    // 不会被读到；因此 reset 不清零整块 32K KV cache，避免每轮请求都耗时。
    cur_pos_ = 0;
}
