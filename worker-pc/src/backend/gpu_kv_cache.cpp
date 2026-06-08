#include "backend/gpu_kv_cache.h"

#include <cstdio>

#if defined(WORKER_PC_ENABLE_CUDA)
#include <cuda_runtime_api.h>
#endif

namespace {

#if defined(WORKER_PC_ENABLE_CUDA)
bool cuda_ok(cudaError_t err, const char* where) {
    if (err == cudaSuccess) {
        return true;
    }
    std::fprintf(stderr, "[worker-pc/GpuKvCache] %s failed: %s\n", where, cudaGetErrorString(err));
    return false;
}
#endif

}  // namespace

bool GpuKvCache::init(int num_layers, int capacity, int kv_dim) {
    destroy();
    if (num_layers <= 0 || capacity <= 0 || kv_dim <= 0) {
        return false;
    }

#if defined(WORKER_PC_ENABLE_CUDA)
    num_layers_ = num_layers;
    max_capacity_ = capacity;
    kv_dim_ = kv_dim;
    ready_ = true;
    k_cache_dev_.assign(num_layers, nullptr);
    v_cache_dev_.assign(num_layers, nullptr);
    const int initial_capacity = capacity >= 1024 ? 1024 : capacity;
    return reserve_capacity(initial_capacity);
#else
    (void)num_layers;
    (void)capacity;
    (void)kv_dim;
    return false;
#endif
}

void GpuKvCache::reset() {
    // No device memset here. Logical valid length is tracked by the host-side cur_pos.
}

void GpuKvCache::destroy() {
#if defined(WORKER_PC_ENABLE_CUDA)
    for (void*& ptr : k_cache_dev_) {
        if (ptr) {
            cudaFree(ptr);
            ptr = nullptr;
        }
    }
    for (void*& ptr : v_cache_dev_) {
        if (ptr) {
            cudaFree(ptr);
            ptr = nullptr;
        }
    }
#endif
    k_cache_dev_.clear();
    v_cache_dev_.clear();
    num_layers_ = 0;
    max_capacity_ = 0;
    capacity_ = 0;
    kv_dim_ = 0;
    ready_ = false;
}

bool GpuKvCache::copy_layer_slice_from_host(int layer,
                                            int pos,
                                            const uint16_t* k_host,
                                            const uint16_t* v_host,
                                            int seq) {
#if defined(WORKER_PC_ENABLE_CUDA)
    if (!ready_ || layer < 0 || layer >= static_cast<int>(k_cache_dev_.size()) ||
        pos < 0 || seq <= 0 || !k_host || !v_host || pos + seq > max_capacity_) {
        return false;
    }
    if (pos + seq > capacity_ && !reserve_capacity(pos + seq)) {
        return false;
    }
    const size_t elems = static_cast<size_t>(seq) * kv_dim_;
    const size_t bytes = elems * sizeof(uint16_t);
    uint16_t* k_dst = reinterpret_cast<uint16_t*>(k_cache_dev_[static_cast<size_t>(layer)]) +
                      static_cast<size_t>(pos) * kv_dim_;
    uint16_t* v_dst = reinterpret_cast<uint16_t*>(v_cache_dev_[static_cast<size_t>(layer)]) +
                      static_cast<size_t>(pos) * kv_dim_;
    return cuda_ok(cudaMemcpy(k_dst, k_host, bytes, cudaMemcpyHostToDevice), "cudaMemcpy(k_slice)") &&
           cuda_ok(cudaMemcpy(v_dst, v_host, bytes, cudaMemcpyHostToDevice), "cudaMemcpy(v_slice)");
#else
    (void)layer;
    (void)pos;
    (void)k_host;
    (void)v_host;
    (void)seq;
    return false;
#endif
}

bool GpuKvCache::reserve_capacity(int required_capacity) {
#if defined(WORKER_PC_ENABLE_CUDA)
    if (!ready_) {
        return false;
    }
    if (required_capacity <= capacity_) {
        return true;
    }
    int new_capacity = capacity_ > 0 ? capacity_ : 1;
    while (new_capacity < required_capacity) {
        new_capacity = std::min(max_capacity_, new_capacity * 2);
        if (new_capacity == capacity_) {
            break;
        }
    }
    if (new_capacity < required_capacity) {
        return false;
    }

    const size_t new_bytes = static_cast<size_t>(new_capacity) * kv_dim_ * sizeof(uint16_t);
    const size_t old_bytes = static_cast<size_t>(capacity_) * kv_dim_ * sizeof(uint16_t);
    for (int i = 0; i < num_layers_; ++i) {
        void* new_k = nullptr;
        void* new_v = nullptr;
        if (!cuda_ok(cudaMalloc(&new_k, new_bytes), "cudaMalloc(grow_k)") ||
            !cuda_ok(cudaMalloc(&new_v, new_bytes), "cudaMalloc(grow_v)")) {
            if (new_k) cudaFree(new_k);
            if (new_v) cudaFree(new_v);
            return false;
        }
        if (old_bytes > 0) {
            if (!cuda_ok(cudaMemcpy(new_k, k_cache_dev_[static_cast<size_t>(i)], old_bytes, cudaMemcpyDeviceToDevice),
                         "cudaMemcpy(grow_k)") ||
                !cuda_ok(cudaMemcpy(new_v, v_cache_dev_[static_cast<size_t>(i)], old_bytes, cudaMemcpyDeviceToDevice),
                         "cudaMemcpy(grow_v)")) {
                cudaFree(new_k);
                cudaFree(new_v);
                return false;
            }
            cudaFree(k_cache_dev_[static_cast<size_t>(i)]);
            cudaFree(v_cache_dev_[static_cast<size_t>(i)]);
        }
        k_cache_dev_[static_cast<size_t>(i)] = new_k;
        v_cache_dev_[static_cast<size_t>(i)] = new_v;
    }
    capacity_ = new_capacity;
    return true;
#else
    (void)required_capacity;
    return false;
#endif
}

const uint16_t* GpuKvCache::k_ptr(int layer) const {
    if (layer < 0 || layer >= static_cast<int>(k_cache_dev_.size())) {
        return nullptr;
    }
    return reinterpret_cast<const uint16_t*>(k_cache_dev_[static_cast<size_t>(layer)]);
}

const uint16_t* GpuKvCache::v_ptr(int layer) const {
    if (layer < 0 || layer >= static_cast<int>(v_cache_dev_.size())) {
        return nullptr;
    }
    return reinterpret_cast<const uint16_t*>(v_cache_dev_[static_cast<size_t>(layer)]);
}
