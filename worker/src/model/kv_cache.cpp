#include "model/kv_cache.h"
#include <cstddef>

void KVCache::init(int num_layers, int capacity, int kv_dim) {
    capacity_ = capacity;
    kv_dim_   = kv_dim;
    cur_pos_  = 0;
    k_cache_.assign(num_layers, std::vector<uint16_t>((size_t)capacity * kv_dim, 0));
    v_cache_.assign(num_layers, std::vector<uint16_t>((size_t)capacity * kv_dim, 0));
}

void KVCache::reset() {
    // cur_pos controls the readable prefix. Old entries past cur_pos are ignored,
    // so reset must not memset the full 32K KV cache on every new request.
    cur_pos_ = 0;
}

KVCache::State KVCache::snapshot() const {
    State state;
    state.k_cache = k_cache_;
    state.v_cache = v_cache_;
    state.capacity = capacity_;
    state.kv_dim = kv_dim_;
    state.cur_pos = cur_pos_;
    return state;
}

bool KVCache::restore(const State& state) {
    if (state.capacity != capacity_ || state.kv_dim != kv_dim_) {
        return false;
    }
    if (state.k_cache.size() != k_cache_.size() || state.v_cache.size() != v_cache_.size()) {
        return false;
    }
    k_cache_ = state.k_cache;
    v_cache_ = state.v_cache;
    cur_pos_ = state.cur_pos;
    return true;
}
