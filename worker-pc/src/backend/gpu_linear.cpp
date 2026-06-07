#include "backend/gpu_linear.h"

#include <cstdio>

bool GpuLinear::init(int K, int N, const uint16_t* weight_kn) {
    if (!warned_once_) {
        warned_once_ = true;
        std::fprintf(stderr,
                     "[worker-pc/GpuLinear] GPU backend is not implemented yet; fallback to CPU backend.\n");
    }
    return cpu_fallback_.init(K, N, weight_kn);
}

bool GpuLinear::forward(const uint16_t* input_f16, int M, uint16_t* output_f16) {
    return cpu_fallback_.forward(input_f16, M, output_f16);
}

void GpuLinear::destroy() {
    cpu_fallback_.destroy();
}

