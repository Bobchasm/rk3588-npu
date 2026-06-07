#include "ops/op_linear.h"

#include "backend/cpu_linear.h"
#include "backend/gpu_linear.h"

#include <cstdio>

std::unique_ptr<ILinearOp> make_linear(ComputeDevice device) {
    switch (device) {
    case ComputeDevice::kGpu:
        return std::unique_ptr<ILinearOp>(new GpuLinear());
    case ComputeDevice::kAuto:
        std::fprintf(stderr, "[worker-pc] auto device currently resolves to CPU\n");
        [[fallthrough]];
    case ComputeDevice::kCpu:
    default:
        return std::unique_ptr<ILinearOp>(new CpuLinear());
    }
}

