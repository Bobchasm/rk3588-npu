#include "backend/device_type.h"

#include <cstdio>

#if defined(WORKER_PC_ENABLE_CUDA)
#include <cuda_runtime_api.h>
#endif

int main() {
    std::printf("worker-pc gpu probe\n");
    std::printf("cuda_build=%s\n",
#if defined(WORKER_PC_ENABLE_CUDA)
                "on"
#else
                "off"
#endif
    );

#if defined(WORKER_PC_ENABLE_CUDA)
    int count = 0;
    const cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        std::printf("cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    std::printf("device_count=%d\n", count);
    for (int i = 0; i < count; ++i) {
        cudaDeviceProp prop{};
        const cudaError_t prop_err = cudaGetDeviceProperties(&prop, i);
        if (prop_err != cudaSuccess) {
            std::printf("device[%d] property query failed: %s\n", i, cudaGetErrorString(prop_err));
            continue;
        }
        std::printf("device[%d]: name=%s sm=%d.%d total_mem_mb=%zu\n",
                    i, prop.name, prop.major, prop.minor, prop.totalGlobalMem / (1024 * 1024));
    }
#else
    std::printf("CUDA backend was not compiled in.\n");
#endif

    const DeviceConfig auto_cfg = resolve_device_config(ComputeDevice::kAuto);
    std::printf("auto_resolved=%s fallback=%s\n",
                compute_device_name(auto_cfg.resolved),
                auto_cfg.used_fallback ? "true" : "false");
    return 0;
}
