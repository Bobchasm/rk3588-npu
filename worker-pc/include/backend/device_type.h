#pragma once

#include <string>

// ============================================================
// ComputeDevice: PC worker 的计算设备抽象
//
// 设计目的：
// 1. 让上层能像使用普通推理框架一样，通过一个 device 入口选择 CPU / GPU。
// 2. 当前先保证 CPU 路径可用；GPU 路径以独立后端类承接，后续可替换为 CUDA / Vulkan 等实现。
// 3. 业务层只依赖这个抽象，不直接依赖某个具体后端类。
// ============================================================

enum class ComputeDevice {
    kAuto = 0,
    kCpu,
    kGpu,
};

struct DeviceConfig {
    ComputeDevice requested = ComputeDevice::kCpu;
    ComputeDevice resolved = ComputeDevice::kCpu;
    bool used_fallback = false;
};

ComputeDevice parse_compute_device(const std::string& text);
const char* compute_device_name(ComputeDevice device);

