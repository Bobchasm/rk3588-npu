#pragma once

#include <cstdint>
#include <vector>

class GpuKvCache {
public:
    GpuKvCache() = default;
    ~GpuKvCache() { destroy(); }

    GpuKvCache(const GpuKvCache&) = delete;
    GpuKvCache& operator=(const GpuKvCache&) = delete;

    bool init(int num_layers, int capacity, int kv_dim);
    void reset();
    void destroy();

    bool copy_layer_slice_from_host(int layer,
                                    int pos,
                                    const uint16_t* k_host,
                                    const uint16_t* v_host,
                                    int seq);

    const uint16_t* k_ptr(int layer) const;
    const uint16_t* v_ptr(int layer) const;
    bool ready() const { return ready_; }

private:
    bool reserve_capacity(int required_capacity);

    std::vector<void*> k_cache_dev_;
    std::vector<void*> v_cache_dev_;
    int num_layers_ = 0;
    int max_capacity_ = 0;
    int capacity_ = 0;
    int kv_dim_ = 0;
    bool ready_ = false;
};
