#include "backend/device_type.h"

#include <algorithm>
#include <cctype>

#if defined(WORKER_PC_ENABLE_CUDA)
#include <cuda_runtime_api.h>
#endif

namespace {

std::string lowercase(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return text;
}

}  // namespace

ComputeDevice parse_compute_device(const std::string& text) {
    const std::string value = lowercase(text);
    if (value.empty() || value == "cpu") {
        return ComputeDevice::kCpu;
    }
    if (value == "gpu" || value == "cuda") {
        return ComputeDevice::kGpu;
    }
    if (value == "auto") {
        return ComputeDevice::kAuto;
    }
    return ComputeDevice::kCpu;
}

const char* compute_device_name(ComputeDevice device) {
    switch (device) {
    case ComputeDevice::kAuto: return "auto";
    case ComputeDevice::kCpu: return "cpu";
    case ComputeDevice::kGpu: return "gpu";
    default: return "cpu";
    }
}

bool is_gpu_available() {
#if defined(WORKER_PC_ENABLE_CUDA)
    int device_count = 0;
    const cudaError_t err = cudaGetDeviceCount(&device_count);
    return err == cudaSuccess && device_count > 0;
#else
    return false;
#endif
}

DeviceConfig resolve_device_config(ComputeDevice requested) {
    DeviceConfig cfg;
    cfg.requested = requested;

    if (requested == ComputeDevice::kAuto) {
        cfg.resolved = is_gpu_available() ? ComputeDevice::kGpu : ComputeDevice::kCpu;
        cfg.used_fallback = (cfg.resolved != ComputeDevice::kGpu);
        return cfg;
    }

    if (requested == ComputeDevice::kGpu && !is_gpu_available()) {
        cfg.resolved = ComputeDevice::kCpu;
        cfg.used_fallback = true;
        return cfg;
    }

    cfg.resolved = requested;
    cfg.used_fallback = false;
    return cfg;
}
