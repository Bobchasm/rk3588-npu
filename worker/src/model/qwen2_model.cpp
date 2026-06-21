#include "model/qwen2_model.h"
#include "model/weight_loader.h"

#include "ops/op_rmsnorm.h"
#include "ops/op_cast.h"
#include "ops/op_embedding.h"
#include "ops/op_attention.h"

#include "core/half.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <sys/time.h>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

// ============================================================
// 辅助：创建并初始化一个 Linear（抽象后端）
//   weight_name 是 safetensors 中的名字
//   transpose=true 表示从 PyTorch [out, in] 转到 [in, out]（= [K, N]）
// ============================================================
static bool init_linear(std::unique_ptr<ILinearOp>& linear,
                        LinearBackend backend,
                        const std::string& sf_path,
                        const TensorMap&   meta,
                        const std::string& weight_name,
                        int K, int N,
                        int layer_idx = -1,
                        const char* role = nullptr)
{
    auto w = load_tensor_f16(sf_path, meta.at(weight_name), /*transpose=*/true);
    linear = make_linear(backend, layer_idx, role);
    if (!linear || !linear->init(K, N, w.data())) {
        if (linear) linear->destroy();
        linear.reset();
        return false;
    }
    return true;
}

static bool init_fused_qkv_linear(std::unique_ptr<ILinearOp>& linear,
                                  LinearBackend backend,
                                  const std::string& sf_path,
                                  const TensorMap&   meta,
                                  const std::string& q_weight_name,
                                  const std::string& k_weight_name,
                                  const std::string& v_weight_name,
                                  int hidden, int kv_dim,
                                  int layer_idx)
{
    auto q_w = load_tensor_f16(sf_path, meta.at(q_weight_name), /*transpose=*/true);
    auto k_w = load_tensor_f16(sf_path, meta.at(k_weight_name), /*transpose=*/true);
    auto v_w = load_tensor_f16(sf_path, meta.at(v_weight_name), /*transpose=*/true);

    const int fused_N = hidden + kv_dim * 2;
    std::vector<uint16_t> fused_w((size_t)hidden * fused_N);
    for (int k = 0; k < hidden; ++k) {
        uint16_t* dst = fused_w.data() + (size_t)k * fused_N;
        std::memcpy(dst,
                    q_w.data() + (size_t)k * hidden,
                    (size_t)hidden * sizeof(uint16_t));
        std::memcpy(dst + hidden,
                    k_w.data() + (size_t)k * kv_dim,
                    (size_t)kv_dim * sizeof(uint16_t));
        std::memcpy(dst + hidden + kv_dim,
                    v_w.data() + (size_t)k * kv_dim,
                    (size_t)kv_dim * sizeof(uint16_t));
    }

    linear = make_linear(backend, layer_idx, "qkv");
    if (!linear || !linear->init(hidden, fused_N, fused_w.data())) {
        if (linear) linear->destroy();
        linear.reset();
        return false;
    }
    return true;
}

static bool init_fused_gate_up_linear(std::unique_ptr<ILinearOp>& linear,
                                      LinearBackend backend,
                                      const std::string& sf_path,
                                      const TensorMap&   meta,
                                      const std::string& gate_weight_name,
                                      const std::string& up_weight_name,
                                      int K, int intermediate,
                                      int layer_idx)
{
    auto gate_w = load_tensor_f16(sf_path, meta.at(gate_weight_name), /*transpose=*/true);
    auto up_w   = load_tensor_f16(sf_path, meta.at(up_weight_name),   /*transpose=*/true);

    const int fused_N = intermediate * 2;
    std::vector<uint16_t> fused_w((size_t)K * fused_N);
    for (int k = 0; k < K; ++k) {
        uint16_t* dst = fused_w.data() + (size_t)k * fused_N;
        std::memcpy(dst,
                    gate_w.data() + (size_t)k * intermediate,
                    (size_t)intermediate * sizeof(uint16_t));
        std::memcpy(dst + intermediate,
                    up_w.data() + (size_t)k * intermediate,
                    (size_t)intermediate * sizeof(uint16_t));
    }

    linear = make_linear(backend, layer_idx, "gate_up");
    if (!linear || !linear->init(K, fused_N, fused_w.data())) {
        if (linear) linear->destroy();
        linear.reset();
        return false;
    }
    return true;
}

static void run_linear_or_throw(const std::unique_ptr<ILinearOp>& linear,
                                const char* name,
                                const uint16_t* input_f16,
                                int M,
                                uint16_t* output_f16)
{
    if (!linear || !linear->forward(input_f16, M, output_f16)) {
        throw std::runtime_error(std::string("Linear forward failed: ") + name);
    }
}

static int linear_batch_rows() {
    constexpr int kDefaultBatchRows = 1;
    static int cached = []() {
        const char* v = std::getenv("RKLLM_LINEAR_BATCH");
        if (!v || v[0] == '\0') {
            return kDefaultBatchRows;
        }

        char* end = nullptr;
        long parsed = std::strtol(v, &end, 10);
        if (end == v || parsed <= 0) {
            std::fprintf(stderr,
                         "[Qwen2Model] invalid RKLLM_LINEAR_BATCH=%s, use %d\n",
                         v, kDefaultBatchRows);
            return kDefaultBatchRows;
        }
        return (int)parsed;
    }();
    return cached;
}

static void run_linear_batched_or_throw(const std::unique_ptr<ILinearOp>& linear,
                                        const char* name,
                                        const uint16_t* input_f16,
                                        int rows,
                                        int input_stride,
                                        int output_stride,
                                        uint16_t* output_f16)
{
    static bool batch_disabled_after_failure = false;
    const int batch_rows = batch_disabled_after_failure ? 1 : std::max(1, linear_batch_rows());
    for (int start = 0; start < rows; start += batch_rows) {
        const int chunk = std::min(batch_rows, rows - start);
        const uint16_t* in = input_f16 + (size_t)start * input_stride;
        uint16_t* out = output_f16 + (size_t)start * output_stride;
        if (chunk == 1) {
            run_linear_or_throw(linear, name, in, 1, out);
            continue;
        }

        const bool can_batch = linear && linear->supports_batch(batch_rows);
        if (!can_batch) {
            for (int r = 0; r < chunk; ++r) {
                run_linear_or_throw(linear, name,
                                    in + (size_t)r * input_stride,
                                    1,
                                    out + (size_t)r * output_stride);
            }
            continue;
        }

        if (chunk == batch_rows && linear->forward(in, chunk, out)) {
            continue;
        }

        static thread_local std::vector<uint16_t> padded_input;
        static thread_local std::vector<uint16_t> padded_output;
        padded_input.assign((size_t)batch_rows * input_stride, 0);
        padded_output.resize((size_t)batch_rows * output_stride);
        for (int r = 0; r < chunk; ++r) {
            std::memcpy(padded_input.data() + (size_t)r * input_stride,
                        in + (size_t)r * input_stride,
                        (size_t)input_stride * sizeof(uint16_t));
        }
        if (linear->forward(padded_input.data(), batch_rows,
                            padded_output.data())) {
            for (int r = 0; r < chunk; ++r) {
                std::memcpy(out + (size_t)r * output_stride,
                            padded_output.data() + (size_t)r * output_stride,
                            (size_t)output_stride * sizeof(uint16_t));
            }
            continue;
        }

        std::fprintf(stderr,
                     "[Qwen2Model] batched linear failed for %s M=%d; fallback to M=1\n",
                     name, batch_rows);
        batch_disabled_after_failure = true;
        for (int r = 0; r < chunk; ++r) {
            run_linear_or_throw(linear, name,
                                in + (size_t)r * input_stride,
                                1,
                                out + (size_t)r * output_stride);
        }
    }
}

static int64_t qwen2_now_us() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return (int64_t)tv.tv_sec * 1000000LL + tv.tv_usec;
}

static bool qwen2_profile_enabled() {
    const char* v = std::getenv("RKLLM_PROFILE");
    return v && v[0] != '\0' && !(v[0] == '0' && v[1] == '\0');
}

static bool is_npu_backend(LinearBackend backend) {
    return backend == LinearBackend::NPU ||
           backend == LinearBackend::NPU_SINGLE ||
           backend == LinearBackend::NPU_SHARDED;
}

static LinearBackend select_lm_head_backend() {
    const char* v = std::getenv("RKLLM_LM_HEAD_BACKEND");
    if (!v || v[0] == '\0') {
        return LinearBackend::NPU;
    }
    if (std::strcmp(v, "CPU") == 0 || std::strcmp(v, "cpu") == 0) {
        return LinearBackend::CPU;
    }
    if (std::strcmp(v, "NPU") == 0 || std::strcmp(v, "npu") == 0) {
        return LinearBackend::NPU;
    }
    if (std::strcmp(v, "NPU_SINGLE") == 0 || std::strcmp(v, "npu_single") == 0 ||
        std::strcmp(v, "SINGLE_NPU") == 0 || std::strcmp(v, "single_npu") == 0) {
        return LinearBackend::NPU_SINGLE;
    }
    if (std::strcmp(v, "NPU_SHARDED") == 0 || std::strcmp(v, "npu_sharded") == 0 ||
        std::strcmp(v, "SHARDED_NPU") == 0 || std::strcmp(v, "sharded_npu") == 0) {
        return LinearBackend::NPU_SHARDED;
    }

    std::fprintf(stderr,
                 "[load] unknown RKLLM_LM_HEAD_BACKEND=%s, use NPU_AUTO\n",
                 v);
    return LinearBackend::NPU;
}

static const char* linear_backend_name(LinearBackend backend) {
    switch (backend) {
        case LinearBackend::CPU: return "CPU";
        case LinearBackend::NPU: return "NPU_AUTO";
        case LinearBackend::NPU_SINGLE: return "NPU_SINGLE";
        case LinearBackend::NPU_SHARDED: return "NPU_SHARDED";
        default: return "UNKNOWN";
    }
}

struct ForwardProfile {
    int64_t embedding_us = 0;
    int64_t rmsnorm_us = 0;
    int64_t qkv_proj_us = 0;
    int64_t rope_us = 0;
    int64_t kv_write_us = 0;
    int64_t attention_us = 0;
    int64_t o_proj_us = 0;
    int64_t gate_up_proj_us = 0;
    int64_t silu_mul_us = 0;
    int64_t down_proj_us = 0;
    int64_t lm_head_us = 0;
    int64_t total_us = 0;
};

static double us_to_ms(int64_t us) {
    return (double)us / 1000.0;
}

static void print_forward_profile(int seq, int pos, int total_len,
                                  const ForwardProfile& p) {
    std::fprintf(stderr,
        "[profile] forward seq=%d pos=%d total_len=%d total=%.2f ms | "
        "embedding=%.2f rmsnorm=%.2f qkv=%.2f rope=%.2f kv_write=%.2f "
        "attention=%.2f o_proj=%.2f gate_up=%.2f silu_mul=%.2f "
        "down=%.2f lm_head=%.2f other=%.2f\n",
        seq, pos, total_len, us_to_ms(p.total_us),
        us_to_ms(p.embedding_us), us_to_ms(p.rmsnorm_us),
        us_to_ms(p.qkv_proj_us), us_to_ms(p.rope_us),
        us_to_ms(p.kv_write_us), us_to_ms(p.attention_us),
        us_to_ms(p.o_proj_us), us_to_ms(p.gate_up_proj_us),
        us_to_ms(p.silu_mul_us), us_to_ms(p.down_proj_us),
        us_to_ms(p.lm_head_us),
        us_to_ms(p.total_us - p.embedding_us - p.rmsnorm_us -
                 p.qkv_proj_us - p.rope_us - p.kv_write_us -
                 p.attention_us - p.o_proj_us - p.gate_up_proj_us -
                 p.silu_mul_us - p.down_proj_us - p.lm_head_us));
}

static inline float silu_scalar(float x) {
    return x / (1.0f + std::exp(-x));
}

static const std::vector<float>& silu_f16_lut() {
    static const std::vector<float> lut = []() {
        std::vector<float> table(65536);
        for (int i = 0; i < 65536; ++i) {
            table[(size_t)i] = silu_scalar(f16_to_f32((uint16_t)i));
        }
        return table;
    }();
    return lut;
}

static inline uint16_t fast_f32_to_f16(float x) {
#if defined(__aarch64__)
    __fp16 h = (__fp16)x;
    uint16_t out;
    std::memcpy(&out, &h, sizeof(out));
    return out;
#else
    return f32_to_f16(x);
#endif
}

static void swiglu_lut_to_f16(const uint16_t* gate, const uint16_t* up,
                              uint16_t* out, int n, const float* silu_lut) {
    int i = 0;
#if defined(__aarch64__)
    alignas(16) float gate_vals[4];
    for (; i + 4 <= n; i += 4) {
        gate_vals[0] = silu_lut[gate[i + 0]];
        gate_vals[1] = silu_lut[gate[i + 1]];
        gate_vals[2] = silu_lut[gate[i + 2]];
        gate_vals[3] = silu_lut[gate[i + 3]];

        float32x4_t gv = vld1q_f32(gate_vals);
        float16x4_t uh = vld1_f16(reinterpret_cast<const float16_t*>(up + i));
        float32x4_t uv = vcvt_f32_f16(uh);
        float16x4_t oh = vcvt_f16_f32(vmulq_f32(gv, uv));
        vst1_f16(reinterpret_cast<float16_t*>(out + i), oh);
    }
#endif
    for (; i < n; ++i) {
        out[i] = fast_f32_to_f16(silu_lut[gate[i]] * f16_to_f32(up[i]));
    }
}

static void f16_to_f32_add_bias(const uint16_t* src, const float* bias,
                                float* dst, int n) {
    int i = 0;
#if defined(__aarch64__)
    for (; i + 4 <= n; i += 4) {
        float16x4_t h = vld1_f16(reinterpret_cast<const float16_t*>(src + i));
        float32x4_t v = vcvt_f32_f16(h);
        float32x4_t b = vld1q_f32(bias + i);
        vst1q_f32(dst + i, vaddq_f32(v, b));
    }
#endif
    for (; i < n; ++i) {
        dst[i] = f16_to_f32(src[i]) + bias[i];
    }
}

static void add_f16_to_f32_inplace(float* dst, const uint16_t* src, int n) {
    int i = 0;
#if defined(__aarch64__)
    for (; i + 4 <= n; i += 4) {
        float32x4_t d = vld1q_f32(dst + i);
        float16x4_t h = vld1_f16(reinterpret_cast<const float16_t*>(src + i));
        float32x4_t v = vcvt_f32_f16(h);
        vst1q_f32(dst + i, vaddq_f32(d, v));
    }
#endif
    for (; i < n; ++i) {
        dst[i] += f16_to_f32(src[i]);
    }
}

static void run_linear_accumulate_or_throw(const std::unique_ptr<ILinearOp>& linear,
                                           const char* name,
                                           const uint16_t* input_f16,
                                           int rows,
                                           int input_stride,
                                           int output_stride,
                                           std::vector<uint16_t>& fallback_output,
                                           float* accum_f32)
{
    if (linear && linear->forward_accumulate(input_f16, rows, accum_f32)) {
        return;
    }

    fallback_output.resize((size_t)rows * output_stride);
    run_linear_batched_or_throw(linear, name,
                                input_f16, rows, input_stride, output_stride,
                                fallback_output.data());
    add_f16_to_f32_inplace(accum_f32, fallback_output.data(), rows * output_stride);
}

static void run_linear_f32_accumulate_or_throw(const std::unique_ptr<ILinearOp>& linear,
                                               const char* name,
                                               const float* input_f32,
                                               int rows,
                                               int input_stride,
                                               int output_stride,
                                               std::vector<uint16_t>& fallback_input,
                                               std::vector<uint16_t>& fallback_output,
                                               float* accum_f32)
{
    if (linear && linear->forward_f32_accumulate(input_f32, rows, accum_f32)) {
        return;
    }

    fallback_input.resize((size_t)rows * input_stride);
    fallback_output.resize((size_t)rows * output_stride);
    op_f32_to_f16(input_f32, fallback_input.data(), rows * input_stride);
    run_linear_batched_or_throw(linear, name,
                                fallback_input.data(), rows, input_stride, output_stride,
                                fallback_output.data());
    add_f16_to_f32_inplace(accum_f32, fallback_output.data(), rows * output_stride);
}

static void apply_rope_cached_one(float* v, int head_dim,
                                  const float* cos_row,
                                  const float* sin_row) {
    const int half = head_dim / 2;
    for (int i = 0; i < half; ++i) {
        const float cos_a = cos_row[i];
        const float sin_a = sin_row[i];
        const float v0 = v[i];
        const float v1 = v[i + half];
        v[i]        = v0 * cos_a - v1 * sin_a;
        v[i + half] = v0 * sin_a + v1 * cos_a;
    }
}

static void apply_rope_cached(float* q, float* k,
                              int n_heads, int n_kv_heads, int head_dim,
                              const float* cos_row, const float* sin_row) {
    for (int h = 0; h < n_heads; ++h) {
        apply_rope_cached_one(q + h * head_dim, head_dim, cos_row, sin_row);
    }
    for (int h = 0; h < n_kv_heads; ++h) {
        apply_rope_cached_one(k + h * head_dim, head_dim, cos_row, sin_row);
    }
}

Qwen2Model::Qwen2Model()  = default;
Qwen2Model::~Qwen2Model() { destroy(); }

bool Qwen2Model::load(const std::string& model_dir, LinearBackend backend) {
    return load(model_dir, backend, PartitionConfig{});
}

void Qwen2Model::destroy() {
    // 显式释放所有线性层（顺序：先 lm_head，再逐层），确保 NPU handle 全部归还
    if (lm_head_) { lm_head_->destroy(); lm_head_.reset(); }
    std::vector<std::unique_ptr<TransformerLayer>>().swap(layers_);
    std::vector<uint16_t>().swap(embed_tokens_);
    std::vector<float>().swap(norm_weight_);
    kv_cache_ = KVCache();
    scratch_ = ForwardScratch();
    partition_ = PartitionConfig{};
    std::vector<float>().swap(rope_cos_);
    std::vector<float>().swap(rope_sin_);
    rope_cached_positions_ = 0;
    rope_cached_head_dim_ = 0;
    rope_cached_theta_ = 0.0f;
}

void Qwen2Model::reset_kv_cache() {
    kv_cache_.reset();
}

Qwen2Model::KvState Qwen2Model::snapshot_kv_state() const {
    KvState state;
    state.kv_cache = kv_cache_.snapshot();
    return state;
}

bool Qwen2Model::restore_kv_state(const KvState& state) {
    return kv_cache_.restore(state.kv_cache);
}

void Qwen2Model::ensure_rope_cache(int required_positions) {
    const int head_dim = config_.head_dim;
    const int half = head_dim / 2;
    const float theta = config_.rope_theta;

    if (required_positions <= 0) return;
    if (rope_cached_head_dim_ != head_dim || rope_cached_theta_ != theta) {
        rope_cos_.clear();
        rope_sin_.clear();
        rope_cached_positions_ = 0;
        rope_cached_head_dim_ = head_dim;
        rope_cached_theta_ = theta;
    }
    if (rope_cached_positions_ >= required_positions) return;

    int new_positions = std::max(required_positions, std::max(64, rope_cached_positions_ * 2));
    new_positions = std::min(new_positions, config_.max_position);
    if (new_positions < required_positions) {
        throw std::runtime_error("RoPE cache capacity exceeded");
    }

    const int old_positions = rope_cached_positions_;
    rope_cos_.resize((size_t)new_positions * half);
    rope_sin_.resize((size_t)new_positions * half);
    for (int pos = old_positions; pos < new_positions; ++pos) {
        float* cos_row = rope_cos_.data() + (size_t)pos * half;
        float* sin_row = rope_sin_.data() + (size_t)pos * half;
        for (int i = 0; i < half; ++i) {
            const float freq = 1.0f / std::pow(theta, (float)(2 * i) / (float)head_dim);
            const float angle = (float)pos * freq;
            cos_row[i] = std::cos(angle);
            sin_row[i] = std::sin(angle);
        }
    }
    rope_cached_positions_ = new_positions;
}

// ============================================================
// load: 解析 safetensors、创建每层后端、填充 KV Cache
// ============================================================
bool Qwen2Model::load(const std::string& model_dir,
                      LinearBackend backend,
                      const PartitionConfig& partition) {
    destroy();
    partition_ = partition;

    std::string sf_path = model_dir + "/model.safetensors";
    try {
        std::printf("[load] 解析 safetensors 文件头...\n");
        TensorMap meta = load_safetensors_meta(sf_path);
        std::printf("[load] 找到 %zu 个张量\n", meta.size());

        const auto& c  = config_;
        const int H    = c.hidden_size;
        const int IL   = c.num_hidden_layers;
        const int kvd  = c.kv_dim();
        const int IS   = c.intermediate_size;
        const int V    = c.vocab_size;
        const int layer_begin = std::max(0, partition_.layer_begin);
        const int layer_end = (partition_.layer_end < 0)
            ? IL
            : std::min(IL, partition_.layer_end);

        auto fail = [this]() {
            destroy();
            return false;
        };

        if (layer_begin >= layer_end) {
            throw std::runtime_error("invalid rk3588 partition layer range");
        }

        // ---- Embedding ----
        if (partition_.include_embedding) {
            std::printf("[load] embed_tokens...\n");
            embed_tokens_ = load_tensor_f16(sf_path, meta.at("model.embed_tokens.weight"));
        }

        // ---- 每层 ----
        layers_.resize(static_cast<size_t>(layer_end - layer_begin));
        for (int i = layer_begin; i < layer_end; ++i) {
            std::unique_ptr<TransformerLayer> L(new TransformerLayer());
            std::string pfx = "model.layers." + std::to_string(i) + ".";

            std::printf("[load] layer %d/%d\r", i + 1, IL);
            std::fflush(stdout);

            L->input_layernorm = load_tensor_f32(sf_path, meta.at(pfx + "input_layernorm.weight"));

            if (is_npu_backend(backend)) {
                if (!init_fused_qkv_linear(L->qkv_proj, backend,
                                           sf_path, meta,
                                           pfx + "self_attn.q_proj.weight",
                                           pfx + "self_attn.k_proj.weight",
                                           pfx + "self_attn.v_proj.weight",
                                           H, kvd,
                                           i)) return fail();
            } else {
                if (!init_linear(L->q_proj, backend, sf_path, meta, pfx + "self_attn.q_proj.weight", H, H, i, "q_proj")) return fail();
                if (!init_linear(L->k_proj, backend, sf_path, meta, pfx + "self_attn.k_proj.weight", H, kvd, i, "k_proj")) return fail();
                if (!init_linear(L->v_proj, backend, sf_path, meta, pfx + "self_attn.v_proj.weight", H, kvd, i, "v_proj")) return fail();
            }
            L->q_bias = load_tensor_f32(sf_path, meta.at(pfx + "self_attn.q_proj.bias"));
            L->k_bias = load_tensor_f32(sf_path, meta.at(pfx + "self_attn.k_proj.bias"));
            L->v_bias = load_tensor_f32(sf_path, meta.at(pfx + "self_attn.v_proj.bias"));

            if (!init_linear(L->o_proj, backend, sf_path, meta, pfx + "self_attn.o_proj.weight", H, H, i, "o_proj")) return fail();

            L->post_attention_layernorm = load_tensor_f32(sf_path, meta.at(pfx + "post_attention_layernorm.weight"));

            if (is_npu_backend(backend)) {
                if (!init_fused_gate_up_linear(L->gate_up_proj, backend,
                                               sf_path, meta,
                                               pfx + "mlp.gate_proj.weight",
                                               pfx + "mlp.up_proj.weight",
                                               H, IS,
                                               i)) return fail();
            } else {
                if (!init_linear(L->gate_proj, backend, sf_path, meta, pfx + "mlp.gate_proj.weight", H,  IS, i, "gate")) return fail();
                if (!init_linear(L->up_proj,   backend, sf_path, meta, pfx + "mlp.up_proj.weight",   H,  IS, i, "up")) return fail();
            }
            if (!init_linear(L->down_proj, backend, sf_path, meta, pfx + "mlp.down_proj.weight", IS, H, i, "down")) return fail();

            layers_[static_cast<size_t>(i - layer_begin)] = std::move(L);
        }
        std::printf("\n[load] 所有层加载完毕\n");

        // ---- final norm ----
        if (partition_.include_final_norm_and_head) {
            norm_weight_ = load_tensor_f32(sf_path, meta.at("model.norm.weight"));

            // ---- lm_head（tied weights，复用 embed_tokens 的转置 = [H, V]）----
            LinearBackend lm_head_backend = select_lm_head_backend();
            std::printf("[load] lm_head (%s)...\n", linear_backend_name(lm_head_backend));
            if (!init_linear(lm_head_, lm_head_backend, sf_path, meta, "model.embed_tokens.weight", H, V, -1, "lm_head"))
                return fail();
        }

        // ---- KV Cache ----
        kv_cache_.init(static_cast<int>(layers_.size()), c.max_position, kvd);
        silu_f16_lut();

        std::printf("[load] 加载完成\n");
        return true;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[load] 加载失败: %s\n", e.what());
        destroy();
        return false;
    }
}

int Qwen2Model::forward_next_token(const std::vector<int>& tokens) {
    if (!can_generate_tokens()) {
        throw std::runtime_error("rk3588 partition does not support token generation");
    }
    return forward_internal(tokens);
}

bool Qwen2Model::forward_tokens_to_hidden(const std::vector<int>& tokens,
                                          std::vector<uint16_t>& output_f16) {
    if (tokens.empty() || !can_tokens_to_hidden()) {
        return false;
    }

    const auto& c = config_;
    const int H = c.hidden_size;
    const int seq = static_cast<int>(tokens.size());
    const int pos = kv_cache_.cur_pos();
    const int total_len = pos + seq;
    if (total_len > kv_cache_.capacity()) {
        return false;
    }

    scratch_.hidden.resize(static_cast<size_t>(seq) * H);
    op_embedding_lookup(embed_tokens_.data(), tokens, scratch_.hidden.data(), H);
    execute_loaded_layers(scratch_.hidden.data(), seq, pos);
    output_f16.resize(static_cast<size_t>(seq) * H);
    op_f32_to_f16(scratch_.hidden.data(), output_f16.data(), seq * H);
    kv_cache_.set_cur_pos(total_len);
    return true;
}

bool Qwen2Model::forward_hidden_states(const uint16_t* input_f16,
                                       int seq,
                                       int pos_base,
                                       std::vector<uint16_t>& output_f16) {
    if (!input_f16 || seq <= 0 || !can_forward_hidden()) {
        return false;
    }

    const int H = config_.hidden_size;
    const int total_len = pos_base + seq;
    if (total_len > kv_cache_.capacity()) {
        return false;
    }

    scratch_.hidden.resize(static_cast<size_t>(seq) * H);
    for (int i = 0; i < seq * H; ++i) {
        scratch_.hidden[static_cast<size_t>(i)] = f16_to_f32(input_f16[static_cast<size_t>(i)]);
    }

    execute_loaded_layers(scratch_.hidden.data(), seq, pos_base);
    output_f16.resize(static_cast<size_t>(seq) * H);
    op_f32_to_f16(scratch_.hidden.data(), output_f16.data(), seq * H);
    kv_cache_.set_cur_pos(total_len);
    return true;
}

bool Qwen2Model::forward_hidden_to_token(const uint16_t* input_f16,
                                         int seq,
                                         int pos_base,
                                         int& output_token_id) {
    if (!input_f16 || seq <= 0 || !can_hidden_to_token()) {
        return false;
    }

    const auto& c = config_;
    const int H = c.hidden_size;
    const int total_len = pos_base + seq;
    if (total_len > kv_cache_.capacity()) {
        return false;
    }

    scratch_.hidden.resize(static_cast<size_t>(seq) * H);
    for (int i = 0; i < seq * H; ++i) {
        scratch_.hidden[static_cast<size_t>(i)] = f16_to_f32(input_f16[static_cast<size_t>(i)]);
    }

    execute_loaded_layers(scratch_.hidden.data(), seq, pos_base);
    scratch_.last.resize(H);
    scratch_.lm_in.resize(H);
    op_rmsnorm(scratch_.hidden.data() + static_cast<size_t>(seq - 1) * H,
               norm_weight_.data(),
               scratch_.last.data(),
               1, H, c.rms_norm_eps);
    op_f32_to_f16(scratch_.last.data(), scratch_.lm_in.data(), H);
    if (!lm_head_ || !lm_head_->forward_argmax(scratch_.lm_in.data(), 1, &output_token_id)) {
        return false;
    }

    kv_cache_.set_cur_pos(total_len);
    return true;
}

void Qwen2Model::execute_loaded_layers(float* hidden, int seq, int pos_base) {
    const auto& c = config_;
    const int H          = c.hidden_size;
    const int total_len  = pos_base + seq;

    auto& S = scratch_;
    S.npu_in.resize((size_t)seq * std::max(H, c.intermediate_size));

    for (int li = 0; li < static_cast<int>(layers_.size()); ++li) {
        TransformerLayer& L = *layers_[li];

        op_rmsnorm_to_f16(hidden, L.input_layernorm.data(),
                          S.npu_in.data(), seq, H, c.rms_norm_eps);

        S.q.resize((size_t)seq * H);
        S.k.resize((size_t)seq * c.kv_dim());
        S.v.resize((size_t)seq * c.kv_dim());
        if (L.qkv_proj) {
            const int qkv_dim = H + c.kv_dim() * 2;
            S.qkv_f16.resize((size_t)seq * qkv_dim);
            run_linear_batched_or_throw(L.qkv_proj, "qkv_proj",
                                        S.npu_in.data(), seq, H, qkv_dim,
                                        S.qkv_f16.data());
            for (int sidx = 0; sidx < seq; ++sidx) {
                const uint16_t* row = S.qkv_f16.data() + (size_t)sidx * qkv_dim;
                float* qrow = S.q.data() + (size_t)sidx * H;
                float* krow = S.k.data() + (size_t)sidx * c.kv_dim();
                float* vrow = S.v.data() + (size_t)sidx * c.kv_dim();
                f16_to_f32_add_bias(row, L.q_bias.data(), qrow, H);
                f16_to_f32_add_bias(row + H, L.k_bias.data(), krow, c.kv_dim());
                f16_to_f32_add_bias(row + H + c.kv_dim(), L.v_bias.data(), vrow, c.kv_dim());
            }
        } else {
            S.q_f16.resize((size_t)seq * H);
            S.k_f16.resize((size_t)seq * c.kv_dim());
            S.v_f16.resize((size_t)seq * c.kv_dim());
            run_linear_batched_or_throw(L.q_proj, "q_proj",
                                        S.npu_in.data(), seq, H, H,
                                        S.q_f16.data());
            run_linear_batched_or_throw(L.k_proj, "k_proj",
                                        S.npu_in.data(), seq, H, c.kv_dim(),
                                        S.k_f16.data());
            run_linear_batched_or_throw(L.v_proj, "v_proj",
                                        S.npu_in.data(), seq, H, c.kv_dim(),
                                        S.v_f16.data());
            for (int sidx = 0; sidx < seq; ++sidx) {
                float* qrow = S.q.data() + (size_t)sidx * H;
                float* krow = S.k.data() + (size_t)sidx * c.kv_dim();
                float* vrow = S.v.data() + (size_t)sidx * c.kv_dim();
                const uint16_t* qsrc = S.q_f16.data() + (size_t)sidx * H;
                const uint16_t* ksrc = S.k_f16.data() + (size_t)sidx * c.kv_dim();
                const uint16_t* vsrc = S.v_f16.data() + (size_t)sidx * c.kv_dim();
                f16_to_f32_add_bias(qsrc, L.q_bias.data(), qrow, H);
                f16_to_f32_add_bias(ksrc, L.k_bias.data(), krow, c.kv_dim());
                f16_to_f32_add_bias(vsrc, L.v_bias.data(), vrow, c.kv_dim());
            }
        }

        ensure_rope_cache(total_len);
        const int half = c.head_dim / 2;
        for (int sidx = 0; sidx < seq; ++sidx) {
            const int abs_pos = pos_base + sidx;
            const float* cos_row = rope_cos_.data() + (size_t)abs_pos * half;
            const float* sin_row = rope_sin_.data() + (size_t)abs_pos * half;
            apply_rope_cached(S.q.data() + sidx * H,
                              S.k.data() + sidx * c.kv_dim(),
                              c.num_attention_heads, c.num_kv_heads, c.head_dim,
                              cos_row, sin_row);
        }

        uint16_t* kc = kv_cache_.k_ptr(li);
        uint16_t* vc = kv_cache_.v_ptr(li);
        for (int sidx = 0; sidx < seq; ++sidx) {
            const float* ksrc = S.k.data() + sidx * c.kv_dim();
            const float* vsrc = S.v.data() + sidx * c.kv_dim();
            uint16_t* kdst = kc + (pos_base + sidx) * c.kv_dim();
            uint16_t* vdst = vc + (pos_base + sidx) * c.kv_dim();
            op_f32_to_f16(ksrc, kdst, c.kv_dim());
            op_f32_to_f16(vsrc, vdst, c.kv_dim());
        }

        S.attn_out.resize((size_t)seq * H);
        if (seq == 1) {
            op_attention_decode(S.q.data(), kc, vc, S.attn_out.data(),
                                total_len, c.num_attention_heads, c.num_kv_heads, c.head_dim);
        } else {
            op_attention(S.q.data(), kc, vc, S.attn_out.data(),
                         seq, total_len, c.num_attention_heads, c.num_kv_heads, c.head_dim,
                         pos_base);
        }

        run_linear_f32_accumulate_or_throw(L.o_proj, "o_proj",
                                           S.attn_out.data(), seq, H, H,
                                           S.npu_in, S.npu_out,
                                           hidden);

        op_rmsnorm_to_f16(hidden, L.post_attention_layernorm.data(),
                          S.npu_in.data(), seq, H, c.rms_norm_eps);

        const bool fused_gate_up = static_cast<bool>(L.gate_up_proj);
        S.ffn_in_f16.resize((size_t)seq * c.intermediate_size);
        if (fused_gate_up) {
            S.gate_up_f16.resize((size_t)seq * c.intermediate_size * 2);
            run_linear_batched_or_throw(L.gate_up_proj, "gate_up_proj",
                                        S.npu_in.data(), seq, H, c.intermediate_size * 2,
                                        S.gate_up_f16.data());
        } else {
            S.gate_f16.resize((size_t)seq * c.intermediate_size);
            S.up_f16.resize((size_t)seq * c.intermediate_size);
            run_linear_batched_or_throw(L.gate_proj, "gate_proj",
                                        S.npu_in.data(), seq, H, c.intermediate_size,
                                        S.gate_f16.data());
            run_linear_batched_or_throw(L.up_proj, "up_proj",
                                        S.npu_in.data(), seq, H, c.intermediate_size,
                                        S.up_f16.data());
        }

        const float* silu_lut = silu_f16_lut().data();
        if (fused_gate_up) {
            for (int sidx = 0; sidx < seq; ++sidx) {
                const uint16_t* row = S.gate_up_f16.data() + (size_t)sidx * c.intermediate_size * 2;
                uint16_t* out = S.ffn_in_f16.data() + (size_t)sidx * c.intermediate_size;
                swiglu_lut_to_f16(row, row + c.intermediate_size, out, c.intermediate_size, silu_lut);
            }
        } else {
            swiglu_lut_to_f16(S.gate_f16.data(), S.up_f16.data(),
                              S.ffn_in_f16.data(), seq * c.intermediate_size, silu_lut);
        }

        run_linear_accumulate_or_throw(L.down_proj, "down_proj",
                                       S.ffn_in_f16.data(), seq, c.intermediate_size, H,
                                       S.ffn_out_f16, hidden);
    }
}

// ============================================================
// forward: 28 层 Transformer Block + final norm + lm_head
//
// 输入：tokens（本次要处理的 token id 序列）
// 输出：最后一个位置的 greedy token
//
// KV Cache 约定：
//   - 进入时：kv_cache_.cur_pos() 是历史已写入的位置数
//   - 返回时：kv_cache_.cur_pos() += tokens.size()
// ============================================================
int Qwen2Model::forward_internal(const std::vector<int>& tokens) {
    if (tokens.empty()) {
        throw std::runtime_error("Qwen2Model::forward_next_token received empty token list");
    }

    const auto& c = config_;
    const int H          = c.hidden_size;
    const int n_heads    = c.num_attention_heads;
    const int n_kv_heads = c.num_kv_heads;
    const int head_dim   = c.head_dim;
    const int kv_dim     = c.kv_dim();
    const int IS         = c.intermediate_size;
    const int seq        = (int)tokens.size();
    const int pos        = kv_cache_.cur_pos();
    const int total_len  = pos + seq;

    if (total_len > kv_cache_.capacity()) {
        std::fprintf(stderr,
                     "[KVCache] capacity exceeded: pos=%d seq=%d total=%d capacity=%d\n",
                     pos, seq, total_len, kv_cache_.capacity());
        throw std::runtime_error("KVCache capacity exceeded");
    }

    ForwardProfile prof;
    const bool profile = qwen2_profile_enabled();
    const int64_t total_t0 = profile ? qwen2_now_us() : 0;
    auto profile_block = [&](int64_t& bucket, auto&& fn) {
        if (profile) {
            int64_t t0 = qwen2_now_us();
            fn();
            bucket += qwen2_now_us() - t0;
        } else {
            fn();
        }
    };

    auto& S = scratch_;

    // ---------- embedding ----------
    S.hidden.resize((size_t)seq * H);
    profile_block(prof.embedding_us, [&]() {
        op_embedding_lookup(embed_tokens_.data(), tokens, S.hidden.data(), H);
    });

    S.npu_in.resize((size_t)seq * std::max(H, IS));

    for (int li = 0; li < c.num_hidden_layers; ++li) {
        TransformerLayer& L = *layers_[li];

        // ---- 1. Input LayerNorm (CPU, direct FP16 for NPU input) ----
        profile_block(prof.rmsnorm_us, [&]() {
            op_rmsnorm_to_f16(S.hidden.data(), L.input_layernorm.data(),
                              S.npu_in.data(), seq, H, c.rms_norm_eps);
        });

        // ---- 2. Q / K / V proj (NPU, batched by prompt rows) ----
        S.q.resize((size_t)seq * H);
        S.k.resize((size_t)seq * kv_dim);
        S.v.resize((size_t)seq * kv_dim);
        profile_block(prof.qkv_proj_us, [&]() {
            if (L.qkv_proj) {
                const int qkv_dim = H + kv_dim * 2;
                S.qkv_f16.resize((size_t)seq * qkv_dim);
                run_linear_batched_or_throw(L.qkv_proj, "qkv_proj",
                                            S.npu_in.data(), seq, H, qkv_dim,
                                            S.qkv_f16.data());
                for (int sidx = 0; sidx < seq; ++sidx) {
                    const uint16_t* row = S.qkv_f16.data() + (size_t)sidx * qkv_dim;
                    float* qrow = S.q.data() + (size_t)sidx * H;
                    float* krow = S.k.data() + (size_t)sidx * kv_dim;
                    float* vrow = S.v.data() + (size_t)sidx * kv_dim;
                    f16_to_f32_add_bias(row, L.q_bias.data(), qrow, H);
                    f16_to_f32_add_bias(row + H, L.k_bias.data(), krow, kv_dim);
                    f16_to_f32_add_bias(row + H + kv_dim, L.v_bias.data(), vrow, kv_dim);
                }
            } else {
                S.q_f16.resize((size_t)seq * H);
                S.k_f16.resize((size_t)seq * kv_dim);
                S.v_f16.resize((size_t)seq * kv_dim);
                run_linear_batched_or_throw(L.q_proj, "q_proj",
                                            S.npu_in.data(), seq, H, H,
                                            S.q_f16.data());
                run_linear_batched_or_throw(L.k_proj, "k_proj",
                                            S.npu_in.data(), seq, H, kv_dim,
                                            S.k_f16.data());
                run_linear_batched_or_throw(L.v_proj, "v_proj",
                                            S.npu_in.data(), seq, H, kv_dim,
                                            S.v_f16.data());
                for (int sidx = 0; sidx < seq; ++sidx) {
                    float* qrow = S.q.data() + (size_t)sidx * H;
                    float* krow = S.k.data() + (size_t)sidx * kv_dim;
                    float* vrow = S.v.data() + (size_t)sidx * kv_dim;
                    const uint16_t* qsrc = S.q_f16.data() + (size_t)sidx * H;
                    const uint16_t* ksrc = S.k_f16.data() + (size_t)sidx * kv_dim;
                    const uint16_t* vsrc = S.v_f16.data() + (size_t)sidx * kv_dim;
                    f16_to_f32_add_bias(qsrc, L.q_bias.data(), qrow, H);
                    f16_to_f32_add_bias(ksrc, L.k_bias.data(), krow, kv_dim);
                    f16_to_f32_add_bias(vsrc, L.v_bias.data(), vrow, kv_dim);
                }
            }
        });

        // ---- 3. RoPE (CPU) ----
        ensure_rope_cache(total_len);
        profile_block(prof.rope_us, [&]() {
            const int half = head_dim / 2;
            for (int sidx = 0; sidx < seq; ++sidx) {
                const int abs_pos = pos + sidx;
                const float* cos_row = rope_cos_.data() + (size_t)abs_pos * half;
                const float* sin_row = rope_sin_.data() + (size_t)abs_pos * half;
                apply_rope_cached(S.q.data() + sidx * H,
                                  S.k.data() + sidx * kv_dim,
                                  n_heads, n_kv_heads, head_dim,
                                  cos_row, sin_row);
            }
        });

        // ---- 4. 写入 KV Cache ----
        uint16_t* kc = kv_cache_.k_ptr(li);
        uint16_t* vc = kv_cache_.v_ptr(li);
        profile_block(prof.kv_write_us, [&]() {
            for (int sidx = 0; sidx < seq; ++sidx) {
                const float* ksrc = S.k.data() + sidx * kv_dim;
                const float* vsrc = S.v.data() + sidx * kv_dim;
                uint16_t* kdst = kc + (pos + sidx) * kv_dim;
                uint16_t* vdst = vc + (pos + sidx) * kv_dim;
                op_f32_to_f16(ksrc, kdst, kv_dim);
                op_f32_to_f16(vsrc, vdst, kv_dim);
            }
        });

        // ---- 5. Attention (CPU) ----
        S.attn_out.resize((size_t)seq * H);
        profile_block(prof.attention_us, [&]() {
            if (seq == 1) {
                op_attention_decode(S.q.data(), kc, vc, S.attn_out.data(),
                                    total_len, n_heads, n_kv_heads, head_dim);
            } else {
                op_attention(S.q.data(), kc, vc, S.attn_out.data(),
                             seq, total_len, n_heads, n_kv_heads, head_dim,
                             /*pos_base=*/pos);
            }
        });

        // ---- 6. O proj (NPU, batched by prompt rows) ----
        profile_block(prof.o_proj_us, [&]() {
            run_linear_f32_accumulate_or_throw(L.o_proj, "o_proj",
                                               S.attn_out.data(), seq, H, H,
                                               S.npu_in, S.npu_out,
                                               S.hidden.data());
        });

        // ---- 8. Post-Attention LayerNorm (CPU, direct FP16 for NPU input) ----
        profile_block(prof.rmsnorm_us, [&]() {
            op_rmsnorm_to_f16(S.hidden.data(), L.post_attention_layernorm.data(),
                              S.npu_in.data(), seq, H, c.rms_norm_eps);
        });

        // ---- 9. FFN: gate & up proj (NPU, batched by prompt rows) ----
        bool fused_gate_up = (bool)L.gate_up_proj;
        S.ffn_in_f16.resize((size_t)seq * IS);
        profile_block(prof.gate_up_proj_us, [&]() {
            if (fused_gate_up) {
                S.gate_up_f16.resize((size_t)seq * IS * 2);
                run_linear_batched_or_throw(L.gate_up_proj, "gate_up_proj",
                                            S.npu_in.data(), seq, H, IS * 2,
                                            S.gate_up_f16.data());
            } else {
                S.gate_f16.resize((size_t)seq * IS);
                S.up_f16.resize((size_t)seq * IS);
                run_linear_batched_or_throw(L.gate_proj, "gate_proj",
                                            S.npu_in.data(), seq, H, IS,
                                            S.gate_f16.data());
                run_linear_batched_or_throw(L.up_proj, "up_proj",
                                            S.npu_in.data(), seq, H, IS,
                                            S.up_f16.data());
            }
        });

        // SiLU(gate) * up
        profile_block(prof.silu_mul_us, [&]() {
            const float* silu_lut = silu_f16_lut().data();
            if (fused_gate_up) {
                for (int sidx = 0; sidx < seq; ++sidx) {
                    const uint16_t* row = S.gate_up_f16.data() + (size_t)sidx * IS * 2;
                    uint16_t* out = S.ffn_in_f16.data() + (size_t)sidx * IS;
                    swiglu_lut_to_f16(row, row + IS, out, IS, silu_lut);
                }
            } else {
                swiglu_lut_to_f16(S.gate_f16.data(), S.up_f16.data(),
                                  S.ffn_in_f16.data(), seq * IS, silu_lut);
            }
        });

        // ---- 10. down proj (NPU, batched by prompt rows) ----
        profile_block(prof.down_proj_us, [&]() {
            run_linear_accumulate_or_throw(L.down_proj, "down_proj",
                                           S.ffn_in_f16.data(), seq, IS, H,
                                           S.ffn_out_f16, S.hidden.data());
        });
    }

    // ---- Final LayerNorm（仅取最后一个 token，因为只需要下一个 token）----
    S.last.resize(H);
    profile_block(prof.rmsnorm_us, [&]() {
        op_rmsnorm(S.hidden.data() + (seq - 1) * H, norm_weight_.data(), S.last.data(),
                   1, H, c.rms_norm_eps);
    });

    // ---- lm_head ----
    S.lm_in.resize(H);
    int next_token = 0;
    profile_block(prof.lm_head_us, [&]() {
        op_f32_to_f16(S.last.data(), S.lm_in.data(), H);
        if (!lm_head_ || !lm_head_->forward_argmax(S.lm_in.data(), 1, &next_token)) {
            throw std::runtime_error("Linear argmax forward failed: lm_head");
        }
    });

    // 更新 KV Cache 位置
    kv_cache_.set_cur_pos(total_len);

    if (profile) {
        prof.total_us = qwen2_now_us() - total_t0;
        print_forward_profile(seq, pos, total_len, prof);
    }

    return next_token;
}
