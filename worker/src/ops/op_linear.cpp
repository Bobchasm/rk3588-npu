#include "ops/op_linear.h"

#include "backend/cpu_linear.h"
#include "backend/npu_linear.h"
#include "backend/sharded_npu_linear.h"

std::unique_ptr<ILinearOp> make_linear(LinearBackend backend) {
    switch (backend) {
        case LinearBackend::CPU:
            return std::unique_ptr<ILinearOp>(new CpuLinear());
        case LinearBackend::NPU_SHARDED:
            return std::unique_ptr<ILinearOp>(new ShardedNpuLinear());
        case LinearBackend::NPU:
        default:
            return std::unique_ptr<ILinearOp>(new NpuLinear());
    }
}
