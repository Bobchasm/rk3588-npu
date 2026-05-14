#include "model/kv_cache.h"
#include <algorithm>

void KVCache::init(int num_layers, int capacity, int kv_dim) {
    capacity_ = capacity;
    kv_dim_   = kv_dim;
    cur_pos_  = 0;
    k_cache_.assign(num_layers, std::vector<uint16_t>((size_t)capacity * kv_dim, 0));
    v_cache_.assign(num_layers, std::vector<uint16_t>((size_t)capacity * kv_dim, 0));
}

void KVCache::reset() {
    for (auto& v : k_cache_) std::fill(v.begin(), v.end(), 0);
    for (auto& v : v_cache_) std::fill(v.begin(), v.end(), 0);
    cur_pos_ = 0;
}
