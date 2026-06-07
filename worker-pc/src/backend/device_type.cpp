#include "backend/device_type.h"

#include <algorithm>
#include <cctype>

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

