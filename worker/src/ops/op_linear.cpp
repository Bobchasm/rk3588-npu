#include "ops/op_linear.h"

#include "backend/cpu_linear.h"
#include "backend/npu_linear.h"
#include "backend/sharded_npu_linear.h"

#include <cstdlib>
#include <cstdio>
#include <cstring>

namespace {

constexpr int64_t kDefaultShardMinOps = 3000000LL;

bool auto_sharding_enabled() {
    const char* v = std::getenv("RKLLM_NPU_AUTO_SHARD");
    return !(v && (std::strcmp(v, "0") == 0 ||
                   std::strcmp(v, "false") == 0 ||
                   std::strcmp(v, "FALSE") == 0 ||
                   std::strcmp(v, "off") == 0 ||
                   std::strcmp(v, "OFF") == 0));
}

int64_t shard_min_ops() {
    const char* v = std::getenv("RKLLM_NPU_SHARD_MIN_OPS");
    if (!v || v[0] == '\0') {
        return kDefaultShardMinOps;
    }

    char* end = nullptr;
    long long parsed = std::strtoll(v, &end, 10);
    if (end == v || parsed <= 0) {
        std::fprintf(stderr,
                     "[LinearPlanner] invalid RKLLM_NPU_SHARD_MIN_OPS=%s, use %lld\n",
                     v, (long long)kDefaultShardMinOps);
        return kDefaultShardMinOps;
    }
    return (int64_t)parsed;
}

bool should_shard_npu_linear(int K, int N) {
    if (!auto_sharding_enabled()) {
        return false;
    }
    if (K <= 0 || N <= 0 || (N % 16) != 0) {
        return false;
    }
    return (int64_t)K * (int64_t)N >= shard_min_ops();
}

class AutoNpuLinear : public ILinearOp {
public:
    AutoNpuLinear() = default;
    ~AutoNpuLinear() override { destroy(); }

    AutoNpuLinear(const AutoNpuLinear&) = delete;
    AutoNpuLinear& operator=(const AutoNpuLinear&) = delete;

    bool init(int K, int N, const uint16_t* weight_kn) override {
        destroy();

        selected_ = should_shard_npu_linear(K, N)
            ? LinearBackend::NPU_SHARDED
            : LinearBackend::NPU_SINGLE;

        impl_ = (selected_ == LinearBackend::NPU_SHARDED)
            ? std::unique_ptr<ILinearOp>(new ShardedNpuLinear())
            : std::unique_ptr<ILinearOp>(new NpuLinear());

        if (!impl_ || !impl_->init(K, N, weight_kn)) {
            if (impl_) impl_->destroy();
            impl_.reset();
            return false;
        }
        return true;
    }

    bool forward(const uint16_t* input_f16, int M, uint16_t* output_f16) override {
        return impl_ && impl_->forward(input_f16, M, output_f16);
    }

    void destroy() override {
        if (impl_) {
            impl_->destroy();
            impl_.reset();
        }
        selected_ = LinearBackend::NPU_SINGLE;
    }

private:
    LinearBackend selected_ = LinearBackend::NPU_SINGLE;
    std::unique_ptr<ILinearOp> impl_;
};

}  // namespace

std::unique_ptr<ILinearOp> make_linear(LinearBackend backend) {
    switch (backend) {
        case LinearBackend::CPU:
            return std::unique_ptr<ILinearOp>(new CpuLinear());
        case LinearBackend::NPU_SHARDED:
            return std::unique_ptr<ILinearOp>(new ShardedNpuLinear());
        case LinearBackend::NPU_SINGLE:
            return std::unique_ptr<ILinearOp>(new NpuLinear());
        case LinearBackend::NPU:
        default:
            return std::unique_ptr<ILinearOp>(new AutoNpuLinear());
    }
}
