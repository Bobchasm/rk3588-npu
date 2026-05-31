#include "ops/op_linear.h"

#include "backend/cpu_linear.h"
#include "backend/npu_linear.h"
#include "backend/npu_linear_w8.h"
#include "backend/sharded_npu_linear.h"
#include "core/half.h"

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cctype>
#include <climits>
#include <memory>
#include <string>
#include <sys/time.h>
#include <vector>

namespace {

constexpr int64_t kDefaultShardMinOps = 3000000LL;
constexpr int kDefaultShardMinN = 16;
constexpr int kDefaultShardMinLargeDim = 4096;
constexpr int64_t kAutotuneMinOps = 3000000LL;
constexpr int64_t kAutotuneMaxOps = 50000000LL;

struct CachedPlan {
    int K = 0;
    int N = 0;
    LinearBackend backend = LinearBackend::NPU_SINGLE;
};

std::vector<CachedPlan>& plan_cache() {
    static std::vector<CachedPlan> cache;
    return cache;
}

int64_t linear_now_us() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return (int64_t)tv.tv_sec * 1000000LL + tv.tv_usec;
}

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

int shard_min_n() {
    const char* v = std::getenv("RKLLM_NPU_SHARD_MIN_N");
    if (!v || v[0] == '\0') {
        return kDefaultShardMinN;
    }

    char* end = nullptr;
    long parsed = std::strtol(v, &end, 10);
    if (end == v || parsed <= 0) {
        std::fprintf(stderr,
                     "[LinearPlanner] invalid RKLLM_NPU_SHARD_MIN_N=%s, use %d\n",
                     v, kDefaultShardMinN);
        return kDefaultShardMinN;
    }
    return (int)parsed;
}

int shard_min_large_dim() {
    const char* v = std::getenv("RKLLM_NPU_SHARD_MIN_LARGE_DIM");
    if (!v || v[0] == '\0') {
        return kDefaultShardMinLargeDim;
    }

    char* end = nullptr;
    long parsed = std::strtol(v, &end, 10);
    if (end == v || parsed <= 0) {
        std::fprintf(stderr,
                     "[LinearPlanner] invalid RKLLM_NPU_SHARD_MIN_LARGE_DIM=%s, use %d\n",
                     v, kDefaultShardMinLargeDim);
        return kDefaultShardMinLargeDim;
    }
    return (int)parsed;
}

bool autotune_enabled() {
    const char* v = std::getenv("RKLLM_NPU_AUTOTUNE");
    return v && (std::strcmp(v, "1") == 0 ||
                 std::strcmp(v, "true") == 0 ||
                 std::strcmp(v, "TRUE") == 0 ||
                 std::strcmp(v, "on") == 0 ||
                 std::strcmp(v, "ON") == 0);
}

bool npu_weight_int8_enabled() {
    const char* v = std::getenv("RKLLM_NPU_WEIGHT_DTYPE");
    if (v && (std::strcmp(v, "w8a16") == 0 ||
              std::strcmp(v, "W8A16") == 0)) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            std::fprintf(stderr,
                         "[LinearPlanner] RKLLM_NPU_WEIGHT_DTYPE=w8a16 is unsupported by this RK3588 rknn_matmul runtime, use FP16\n");
        }
        return false;
    }
    return v && (std::strcmp(v, "int8") == 0 ||
                 std::strcmp(v, "INT8") == 0 ||
                 std::strcmp(v, "a8w8") == 0 ||
                 std::strcmp(v, "A8W8") == 0 ||
                 std::strcmp(v, "w8") == 0 ||
                 std::strcmp(v, "W8") == 0);
}

bool token_eq(const char* begin, const char* end, const char* expected) {
    const size_t len = (size_t)(end - begin);
    return std::strlen(expected) == len && std::strncmp(begin, expected, len) == 0;
}

bool npu_int8_layer_allowed(int layer_idx) {
    const char* spec = std::getenv("RKLLM_NPU_INT8_LAYERS");
    if (!spec || spec[0] == '\0') {
        return true;
    }

    // lm_head and legacy callers do not have a transformer layer index.
    if (layer_idx < 0) {
        return true;
    }

    const char* p = spec;
    while (*p) {
        while (*p == ',' || std::isspace((unsigned char)*p)) {
            ++p;
        }
        const char* begin = p;
        while (*p && *p != ',') {
            ++p;
        }
        const char* end = p;
        while (end > begin && std::isspace((unsigned char)*(end - 1))) {
            --end;
        }
        while (begin < end && std::isspace((unsigned char)*begin)) {
            ++begin;
        }

        if (begin == end) {
            continue;
        }
        if (token_eq(begin, end, "all")) {
            return true;
        }
        if (token_eq(begin, end, "none") || token_eq(begin, end, "off")) {
            return false;
        }

        char* parse_end = nullptr;
        long first = std::strtol(begin, &parse_end, 10);
        if (parse_end == begin) {
            std::fprintf(stderr,
                         "[LinearPlanner] invalid RKLLM_NPU_INT8_LAYERS token: %.*s\n",
                         (int)(end - begin), begin);
            continue;
        }
        long last = first;
        if (parse_end < end && *parse_end == '-') {
            const char* second_begin = parse_end + 1;
            char* second_end = nullptr;
            last = std::strtol(second_begin, &second_end, 10);
            if (second_end == second_begin) {
                std::fprintf(stderr,
                             "[LinearPlanner] invalid RKLLM_NPU_INT8_LAYERS range: %.*s\n",
                             (int)(end - begin), begin);
                continue;
            }
        }
        if (first > last) {
            const long tmp = first;
            first = last;
            last = tmp;
        }
        if ((long)layer_idx >= first && (long)layer_idx <= last) {
            return true;
        }
    }

    return false;
}

bool npu_int8_for_context(int K, int N, int layer_idx, const char* role) {
    if (!npu_weight_int8_enabled()) {
        return false;
    }
    if (!npu_int8_layer_allowed(layer_idx)) {
        return false;
    }

    const bool has_role = role && role[0] != '\0';
    const bool is_gate_up = has_role ? std::strcmp(role, "gate_up") == 0
                                     : (K == 1536 && N == 17920);
    const char* scope = std::getenv("RKLLM_NPU_INT8_SCOPE");
    if (!scope || scope[0] == '\0') {
        return is_gate_up;
    }
    if (std::strcmp(scope, "gate_up") == 0) {
        return is_gate_up;
    }
    if (std::strcmp(scope, "off") == 0 || std::strcmp(scope, "none") == 0) {
        return false;
    }
    std::fprintf(stderr,
                 "[LinearPlanner] unknown RKLLM_NPU_INT8_SCOPE=%s, use gate_up\n",
                 scope);
    return is_gate_up;
}

std::unique_ptr<ILinearOp> make_single_npu_impl(int K = 0, int N = 0,
                                                int layer_idx = -1,
                                                const char* role = nullptr) {
    if (npu_int8_for_context(K, N, layer_idx, role)) {
        return std::unique_ptr<ILinearOp>(new NpuLinearW8());
    }
    return std::unique_ptr<ILinearOp>(new NpuLinear());
}

bool can_shard_npu_linear(int K, int N) {
    return auto_sharding_enabled() && K > 0 && N > 0 && (N % 16) == 0;
}

bool force_shard_for_shape(int K, int N) {
    const char* scope = std::getenv("RKLLM_NPU_SHARD_SCOPE");
    if (!scope || scope[0] == '\0' ||
        std::strcmp(scope, "auto") == 0 ||
        std::strcmp(scope, "AUTO") == 0) {
        return false;
    }
    const bool is_qkv = (K == 1536 && N == 2048);
    const bool is_o_proj = (K == 1536 && N == 1536);
    const bool is_gate_up = (K == 1536 && N == 17920);
    const bool is_down = (K == 8960 && N == 1536);
    const bool is_lm_head = (K == 1536 && N > 100000);

    if (std::strcmp(scope, "qkv") == 0) {
        return is_qkv;
    }
    if (std::strcmp(scope, "o_proj") == 0) {
        return is_o_proj;
    }
    if (std::strcmp(scope, "attn") == 0) {
        return is_qkv || is_o_proj;
    }
    if (std::strcmp(scope, "mlp") == 0) {
        return is_gate_up || is_down;
    }
    if (std::strcmp(scope, "lm_head") == 0) {
        return is_lm_head;
    }
    if (std::strcmp(scope, "all") == 0) {
        return true;
    }
    if (std::strcmp(scope, "off") == 0 || std::strcmp(scope, "none") == 0) {
        return false;
    }
    std::fprintf(stderr,
                 "[LinearPlanner] unknown RKLLM_NPU_SHARD_SCOPE=%s, use auto\n",
                 scope);
    return false;
}

bool should_shard_npu_linear(int K, int N) {
    if (!can_shard_npu_linear(K, N)) {
        return false;
    }
    if (force_shard_for_shape(K, N)) {
        return true;
    }
    if (N < shard_min_n()) {
        return false;
    }
    const int large_dim = shard_min_large_dim();
    if (K < large_dim && N < large_dim) {
        return false;
    }
    return (int64_t)K * (int64_t)N >= shard_min_ops();
}

bool find_cached_plan(int K, int N, LinearBackend* backend) {
    for (const auto& p : plan_cache()) {
        if (p.K == K && p.N == N) {
            *backend = p.backend;
            return true;
        }
    }
    return false;
}

void cache_plan(int K, int N, LinearBackend backend) {
    plan_cache().push_back(CachedPlan{K, N, backend});
}

int64_t bench_linear_us(ILinearOp* linear, int K, int N) {
    if (!linear) return LLONG_MAX;

    std::vector<uint16_t> input((size_t)K, f32_to_f16(0.125f));
    std::vector<uint16_t> output((size_t)N);

    if (!linear->forward(input.data(), 1, output.data())) {
        return LLONG_MAX;
    }

    constexpr int kRepeats = 3;
    int64_t total = 0;
    for (int i = 0; i < kRepeats; ++i) {
        const int64_t t0 = linear_now_us();
        if (!linear->forward(input.data(), 1, output.data())) {
            return LLONG_MAX;
        }
        total += linear_now_us() - t0;
    }
    return total / kRepeats;
}

LinearBackend tune_or_plan_backend(int K, int N, const uint16_t* weight_kn) {
    LinearBackend cached;
    if (find_cached_plan(K, N, &cached)) {
        return cached;
    }

    const int64_t ops = (int64_t)K * (int64_t)N;
    LinearBackend planned = should_shard_npu_linear(K, N)
        ? LinearBackend::NPU_SHARDED
        : LinearBackend::NPU_SINGLE;

    if (!autotune_enabled() || !can_shard_npu_linear(K, N) ||
        ops < kAutotuneMinOps || ops > kAutotuneMaxOps) {
        cache_plan(K, N, planned);
        return planned;
    }

    std::unique_ptr<ILinearOp> single(new NpuLinear());
    std::unique_ptr<ILinearOp> sharded(new ShardedNpuLinear());

    const bool single_ok = single->init(K, N, weight_kn);
    const int64_t single_us = single_ok ? bench_linear_us(single.get(), K, N) : LLONG_MAX;

    const bool sharded_ok = sharded->init(K, N, weight_kn);
    const int64_t sharded_us = sharded_ok ? bench_linear_us(sharded.get(), K, N) : LLONG_MAX;

    if (single_ok) single->destroy();
    if (sharded_ok) sharded->destroy();

    LinearBackend selected = planned;
    if (single_us != LLONG_MAX || sharded_us != LLONG_MAX) {
        selected = (sharded_us < single_us)
            ? LinearBackend::NPU_SHARDED
            : LinearBackend::NPU_SINGLE;
    }

    std::fprintf(stderr,
                 "[LinearPlanner] autotune K=%d N=%d single=%.3f ms sharded=%.3f ms -> %s\n",
                 K, N,
                 single_us == LLONG_MAX ? -1.0 : (double)single_us / 1000.0,
                 sharded_us == LLONG_MAX ? -1.0 : (double)sharded_us / 1000.0,
                 selected == LinearBackend::NPU_SHARDED ? "NPU_SHARDED" : "NPU_SINGLE");

    cache_plan(K, N, selected);
    return selected;
}

class AutoNpuLinear : public ILinearOp {
public:
    AutoNpuLinear(LinearBackend requested_backend,
                  int layer_idx,
                  const char* role)
        : requested_backend_(requested_backend),
          layer_idx_(layer_idx),
          role_(role ? role : "") {}
    ~AutoNpuLinear() override { destroy(); }

    AutoNpuLinear(const AutoNpuLinear&) = delete;
    AutoNpuLinear& operator=(const AutoNpuLinear&) = delete;

    bool init(int K, int N, const uint16_t* weight_kn) override {
        destroy();

        selected_ = (requested_backend_ == LinearBackend::NPU)
            ? tune_or_plan_backend(K, N, weight_kn)
            : requested_backend_;

        const char* role = role_.empty() ? nullptr : role_.c_str();
        if (selected_ == LinearBackend::NPU_SHARDED) {
            std::unique_ptr<ShardedNpuLinear> sharded(new ShardedNpuLinear());
            sharded->set_allow_a8w8(npu_int8_for_context(K, N, layer_idx_, role));
            impl_ = std::move(sharded);
        } else {
            impl_ = make_single_npu_impl(K, N, layer_idx_, role);
        }
        if (!impl_ || !impl_->init(K, N, weight_kn)) {
            if (impl_) impl_->destroy();
            if (selected_ == LinearBackend::NPU_SINGLE &&
                npu_int8_for_context(K, N, layer_idx_, role)) {
                std::fprintf(stderr,
                             "[LinearPlanner] W8 init failed K=%d N=%d, fallback FP16\n",
                             K, N);
                impl_.reset(new NpuLinear());
                if (impl_ && impl_->init(K, N, weight_kn)) {
                    return true;
                }
                if (impl_) impl_->destroy();
            }
            impl_.reset();
            return false;
        }
        return true;
    }

    bool forward(const uint16_t* input_f16, int M, uint16_t* output_f16) override {
        return impl_ && impl_->forward(input_f16, M, output_f16);
    }

    bool forward_accumulate(const uint16_t* input_f16, int M, float* accum_f32) override {
        return impl_ && impl_->forward_accumulate(input_f16, M, accum_f32);
    }

    bool forward_f32_accumulate(const float* input_f32, int M, float* accum_f32) override {
        return impl_ && impl_->forward_f32_accumulate(input_f32, M, accum_f32);
    }

    bool supports_batch(int M) const override {
        return impl_ && impl_->supports_batch(M);
    }

    bool forward_argmax(const uint16_t* input_f16, int M,
                        int* argmax_id, uint16_t* argmax_value = nullptr) override {
        return impl_ && impl_->forward_argmax(input_f16, M, argmax_id, argmax_value);
    }

    void destroy() override {
        if (impl_) {
            impl_->destroy();
            impl_.reset();
        }
        selected_ = LinearBackend::NPU_SINGLE;
    }

private:
    LinearBackend requested_backend_ = LinearBackend::NPU;
    LinearBackend selected_ = LinearBackend::NPU_SINGLE;
    int layer_idx_ = -1;
    std::string role_;
    std::unique_ptr<ILinearOp> impl_;
};

}  // namespace

std::unique_ptr<ILinearOp> make_linear(LinearBackend backend,
                                       int layer_idx,
                                       const char* role) {
    switch (backend) {
        case LinearBackend::CPU:
            return std::unique_ptr<ILinearOp>(new CpuLinear());
        case LinearBackend::NPU_SHARDED:
        case LinearBackend::NPU_SINGLE:
        case LinearBackend::NPU:
        default:
            return std::unique_ptr<ILinearOp>(new AutoNpuLinear(backend, layer_idx, role));
    }
}
