#include "model/qwen2_model.h"
#include "model/weight_loader.h"

#include "backend/npu_weight_cache.h"

#include "ops/op_rmsnorm.h"
#include "ops/op_cast.h"
#include "ops/op_embedding.h"
#include "ops/op_attention.h"

#include "core/half.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>

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
    linear = make_linear(backend, layer_idx, role);
    if (linear) {
        // 单个普通权重的 cache key 直接使用 safetensors tensor 名。
        // 命中时后端已经有 native-layout B，可以完全跳过下面的
        // load_tensor_f16()、转置和 rknn_B_normal_layout_to_native_layout()。
        linear->set_cache_key(weight_name);
        if (npu_weight_cache::enabled() && linear->init_from_cache(K, N)) {
            return true;
        }
    }

    auto w = load_tensor_f16(sf_path, meta.at(weight_name), /*transpose=*/true);
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
    linear = make_linear(backend, layer_idx, "qkv");
    if (linear) {
        // q/k/v 在 NPU 路径下会融合成一个 [hidden, hidden+2*kv_dim] 的 B。
        // 这里必须使用融合后的逻辑 key，而不是三个原始 tensor 名，否则
        // warm load 无法跳过三次读取、转置和融合拷贝。
        const std::string cache_key =
            "model.layers." + std::to_string(layer_idx) + ".self_attn.qkv_fused";
        linear->set_cache_key(cache_key);
        const int fused_N = hidden + kv_dim * 2;
        if (npu_weight_cache::enabled() && linear->init_from_cache(hidden, fused_N)) {
            return true;
        }
    }

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
    linear = make_linear(backend, layer_idx, "gate_up");
    if (linear) {
        // gate/up 是当前 MLP 的加载和推理热点。NPU 后端会把它们融合成
        // [gate_all, up_all]，sharded gate_up_pair 还会按三核重排为
        // [gate_slice_i, up_slice_i]。cache 命中后这些重排都不用再做。
        const std::string cache_key =
            "model.layers." + std::to_string(layer_idx) + ".mlp.gate_up_fused";
        linear->set_cache_key(cache_key);
        if (npu_weight_cache::enabled() && linear->init_from_cache(K, intermediate * 2)) {
            return true;
        }
    }

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

static bool qwen2_batch_trace_enabled() {
    static bool enabled = []() {
        const char* v = std::getenv("RKLLM_BATCH_TRACE");
        return v && v[0] != '\0' && !(v[0] == '0' && v[1] == '\0');
    }();
    return enabled;
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
            if (qwen2_batch_trace_enabled()) {
                std::fprintf(stderr,
                             "[batch] %s rows=%d start=%d chunk=%d runM=1 path=single\n",
                             name, rows, start, chunk);
            }
            run_linear_or_throw(linear, name, in, 1, out);
            continue;
        }

        const bool can_batch = linear &&
            (linear->supports_batch(chunk) || linear->supports_batch(batch_rows));
        if (!can_batch) {
            if (qwen2_batch_trace_enabled()) {
                std::fprintf(stderr,
                             "[batch] %s rows=%d start=%d chunk=%d batch_rows=%d path=fallback_per_row\n",
                             name, rows, start, chunk, batch_rows);
            }
            for (int r = 0; r < chunk; ++r) {
                run_linear_or_throw(linear, name,
                                    in + (size_t)r * input_stride,
                                    1,
                                    out + (size_t)r * output_stride);
            }
            continue;
        }

        if (linear->supports_batch(chunk) && linear->forward(in, chunk, out)) {
            if (qwen2_batch_trace_enabled()) {
                std::fprintf(stderr,
                             "[batch] %s rows=%d start=%d chunk=%d runM=%d path=batch\n",
                             name, rows, start, chunk, chunk);
            }
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
            if (qwen2_batch_trace_enabled()) {
                std::fprintf(stderr,
                             "[batch] %s rows=%d start=%d chunk=%d runM=%d path=padded\n",
                             name, rows, start, chunk, batch_rows);
            }
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

static int64_t safetensors_data_base_or_neg1(const std::string& path) {
    FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) {
        return -1;
    }

    uint64_t hdr_size = 0;
    const bool ok = (std::fread(&hdr_size, 1, 8, fp) == 8);
    std::fclose(fp);
    return ok ? (int64_t)(8 + hdr_size) : -1;
}

static bool qwen2_profile_enabled() {
    const char* v = std::getenv("RKLLM_PROFILE");
    return v && v[0] != '\0' && !(v[0] == '0' && v[1] == '\0');
}

static bool qwen2_env_enabled(const char* name) {
    const char* v = std::getenv(name);
    return v && v[0] != '\0' &&
           std::strcmp(v, "0") != 0 &&
           std::strcmp(v, "false") != 0 &&
           std::strcmp(v, "FALSE") != 0 &&
           std::strcmp(v, "off") != 0 &&
           std::strcmp(v, "OFF") != 0;
}

static bool qwen2_mlp_profile_enabled() {
    return qwen2_env_enabled("RKLLM_MLP_PROFILE");
}

static int qwen2_mlp_profile_layer_filter() {
    static int cached = []() {
        const char* v = std::getenv("RKLLM_MLP_PROFILE_LAYER");
        if (!v || v[0] == '\0') {
            return -1;
        }
        char* end = nullptr;
        long parsed = std::strtol(v, &end, 10);
        if (end == v || parsed < 0) {
            std::fprintf(stderr,
                         "[mlp_profile] invalid RKLLM_MLP_PROFILE_LAYER=%s, log all layers\n",
                         v);
            return -1;
        }
        return (int)parsed;
    }();
    return cached;
}

static bool qwen2_mlp_profile_layer_enabled(int layer_idx) {
    if (!qwen2_mlp_profile_enabled()) {
        return false;
    }
    const int filter = qwen2_mlp_profile_layer_filter();
    return filter < 0 || filter == layer_idx;
}

static bool qwen2_load_profile_enabled() {
    return npu_weight_cache::load_profile_enabled();
}

static bool is_npu_backend(LinearBackend backend) {
    return backend == LinearBackend::NPU ||
           backend == LinearBackend::NPU_SINGLE ||
           backend == LinearBackend::NPU_SHARDED;
}

static LinearBackend select_lm_head_backend() {
    const char* v = std::getenv("RKLLM_LM_HEAD_BACKEND");
    if (!v || v[0] == '\0') {
        return LinearBackend::NPU_SHARDED;
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
                 "[load] unknown RKLLM_LM_HEAD_BACKEND=%s, use NPU_SHARDED\n",
                 v);
    return LinearBackend::NPU_SHARDED;
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
        if (qwen2_batch_trace_enabled()) {
            std::fprintf(stderr,
                         "[batch] %s rows=%d runM=%d path=accumulate\n",
                         name, rows, rows);
        }
        return;
    }

    if (qwen2_batch_trace_enabled()) {
        std::fprintf(stderr,
                     "[batch] %s rows=%d runM=0 path=accumulate_fallback\n",
                     name, rows);
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
        if (qwen2_batch_trace_enabled()) {
            std::fprintf(stderr,
                         "[batch] %s rows=%d runM=%d path=f32_accumulate\n",
                         name, rows, rows);
        }
        return;
    }

    if (qwen2_batch_trace_enabled()) {
        std::fprintf(stderr,
                     "[batch] %s rows=%d runM=0 path=f32_accumulate_fallback\n",
                     name, rows);
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

void Qwen2Model::destroy() {
    // 显式释放所有线性层（顺序：先 lm_head，再逐层），确保 NPU handle 全部归还
    if (lm_head_) { lm_head_->destroy(); lm_head_.reset(); }
    std::vector<std::unique_ptr<TransformerLayer>>().swap(layers_);
    unmap_embedding_weight();
    std::vector<uint16_t>().swap(embed_tokens_);
    std::vector<float>().swap(norm_weight_);
    kv_cache_ = KVCache();
    scratch_ = ForwardScratch();
    std::vector<float>().swap(rope_cos_);
    std::vector<float>().swap(rope_sin_);
    rope_cached_positions_ = 0;
    rope_cached_head_dim_ = 0;
    rope_cached_theta_ = 0.0f;
}

void Qwen2Model::unmap_embedding_weight() {
    if (embed_tokens_mmap_) {
        ::munmap(const_cast<void*>(embed_tokens_mmap_), embed_tokens_mmap_bytes_);
    }
    embed_tokens_mmap_ = nullptr;
    embed_tokens_mmap_bytes_ = 0;
    embed_tokens_view_ = nullptr;
    embed_tokens_dtype_ = 0;
}

bool Qwen2Model::map_embedding_weight(const std::string& sf_path,
                                      const TensorMeta& meta) {
    unmap_embedding_weight();

    if (meta.shape.size() != 2 ||
        meta.shape[0] != config_.vocab_size ||
        meta.shape[1] != config_.hidden_size) {
        return false;
    }

    int dtype = 0;
    size_t elem_size = 0;
    if (meta.dtype == "F16") {
        dtype = (int)EmbeddingStorageDType::FP16;
        elem_size = 2;
    } else if (meta.dtype == "BF16") {
        dtype = (int)EmbeddingStorageDType::BF16;
        elem_size = 2;
    } else if (meta.dtype == "F32") {
        dtype = (int)EmbeddingStorageDType::FP32;
        elem_size = 4;
    } else {
        return false;
    }

    const int64_t data_base = safetensors_data_base_or_neg1(sf_path);
    const int64_t tensor_bytes = meta.data_end - meta.data_begin;
    const int64_t expected_bytes =
        (int64_t)config_.vocab_size * config_.hidden_size * (int64_t)elem_size;
    if (data_base < 0 || tensor_bytes != expected_bytes) {
        return false;
    }

    const int fd = ::open(sf_path.c_str(), O_RDONLY);
    if (fd < 0) {
        return false;
    }

    const int64_t tensor_offset = data_base + meta.data_begin;
    const long page_size_long = ::sysconf(_SC_PAGESIZE);
    const size_t page_size = page_size_long > 0 ? (size_t)page_size_long : 4096;
    const int64_t map_offset = tensor_offset & ~((int64_t)page_size - 1);
    const size_t delta = (size_t)(tensor_offset - map_offset);
    const size_t map_bytes = delta + (size_t)tensor_bytes;

    void* mapped = ::mmap(nullptr, map_bytes, PROT_READ, MAP_PRIVATE, fd, map_offset);
    ::close(fd);
    if (mapped == MAP_FAILED) {
        return false;
    }

    embed_tokens_mmap_ = mapped;
    embed_tokens_mmap_bytes_ = map_bytes;
    embed_tokens_view_ = static_cast<const uint8_t*>(mapped) + delta;
    embed_tokens_dtype_ = dtype;
    embed_tokens_.clear();
    std::fprintf(stderr,
                 "[load] embed_tokens mmap %s %.2f MB\n",
                 meta.dtype.c_str(),
                 (double)tensor_bytes / (1024.0 * 1024.0));
    return true;
}

void Qwen2Model::reset_kv_cache() {
    kv_cache_.reset();
}

void Qwen2Model::ensure_rope_cache(int required_positions) {
    // RoPE 的 cos/sin 只和 position、head_dim、theta 有关，与输入内容无关。
    // 这里按需扩容缓存，forward 时直接按绝对位置取一行，避免每层重复算三角函数。
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
//
// 主路线：
//   1. 解析 safetensors header，得到每个 tensor 的 dtype/shape/offset。
//   2. embedding/norm/bias 仍留在 CPU vector 中。
//   3. 每个 Linear 通过 init_linear/init_fused_* 创建后端：
//      - NPU 后端：权重转为 [K, N] 后再转 RKNN native B。
//      - CPU 后端：保存普通 [K, N]。
//      - warm cache 命中时可跳过大权重读取和转换。
//   4. 分配 KV Cache 和常用 LUT。
// ============================================================
bool Qwen2Model::load(const std::string& model_dir, LinearBackend backend) {
    destroy();

    std::string sf_path = model_dir + "/model.safetensors";
    // 先配置 native 权重缓存目录，后面的 init_linear/init_fused_* 才能在
    // 读取大权重前尝试 warm-load。
    npu_weight_cache::configure_for_model(model_dir, sf_path);
    const bool load_profile = qwen2_load_profile_enabled();
    const int64_t load_t0 = load_profile ? qwen2_now_us() : 0;
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

        auto fail = [this]() {
            destroy();
            return false;
        };

        // ---- Embedding ----
        std::printf("[load] embed_tokens...\n");
        const TensorMeta& embed_meta = meta.at("model.embed_tokens.weight");
        if (!map_embedding_weight(sf_path, embed_meta)) {
            embed_tokens_ = load_tensor_f16(sf_path, embed_meta);
        }

        // ---- lm_head（tied weights，复用 embed_tokens 的转置 = [H, V]）----
        // lm_head 是整个模型里最大的 Linear。先分配它的 NPU B shard，
        // 避免所有 transformer 层加载完后再申请大块 B 时被 RKNN/CMA 碎片化卡住。
        LinearBackend lm_head_backend = select_lm_head_backend();
        std::printf("[load] lm_head (%s)...\n", linear_backend_name(lm_head_backend));
        if (!init_linear(lm_head_, lm_head_backend, sf_path, meta, "model.embed_tokens.weight", H, V, -1, "lm_head"))
            return fail();

        // ---- 每层 ----
        layers_.resize(IL);
        for (int i = 0; i < IL; ++i) {
            std::unique_ptr<TransformerLayer> L(new TransformerLayer());
            std::string pfx = "model.layers." + std::to_string(i) + ".";

            std::printf("[load] layer %d/%d\r", i+1, IL);
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

            layers_[i] = std::move(L);
        }
        std::printf("\n[load] 所有层加载完毕\n");

        // ---- final norm ----
        norm_weight_ = load_tensor_f32(sf_path, meta.at("model.norm.weight"));

        // ---- KV Cache ----
        kv_cache_.init(IL, c.max_position, kvd);
        silu_f16_lut();

        std::printf("[load] 加载完成\n");
        npu_weight_cache::print_summary_if_enabled();
        if (load_profile) {
            std::fprintf(stderr, "[load] total=%.2f ms\n",
                         us_to_ms(qwen2_now_us() - load_t0));
        }
        return true;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[load] 加载失败: %s\n", e.what());
        npu_weight_cache::print_summary_if_enabled();
        if (load_profile) {
            std::fprintf(stderr, "[load] total_failed=%.2f ms\n",
                         us_to_ms(qwen2_now_us() - load_t0));
        }
        destroy();
        return false;
    }
}

int Qwen2Model::forward_next_token(const std::vector<int>& tokens) {
    return forward_internal(tokens);
}

// ============================================================
// forward: 28 层 Transformer Block + final norm + lm_head
//
// 输入：tokens（本次要处理的 token id 序列）
// 输出：最后一个位置的 greedy token
//
// 一次 forward 可以是两种形态：
//   - prefill: seq > 1，输入是整段 prompt，KV Cache 从 pos 写入多行。
//   - decode : seq = 1，输入是上一个生成 token，只追加一行 KV。
//
// 数据精度路线：
//   - CPU 激活主干多用 FP32，方便 RMSNorm/attention/residual 累加。
//   - NPU Linear 输入输出为 FP16，因此进入 NPU 前会写入 FP16 buffer。
//   - NPU prepared fast path 允许 RMSNorm/SwiGLU 直接写进后端 A buffer，
//     减少中间 memcpy。
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
    // token id 查表得到 [seq, hidden] FP32 hidden，作为第一层输入。
    S.hidden.resize((size_t)seq * H);
    profile_block(prof.embedding_us, [&]() {
        if (embed_tokens_view_) {
            op_embedding_lookup_typed(embed_tokens_view_,
                                      (EmbeddingStorageDType)embed_tokens_dtype_,
                                      tokens, S.hidden.data(), H);
        } else {
            op_embedding_lookup(embed_tokens_.data(), tokens, S.hidden.data(), H);
        }
    });

    // npu_in 是通用 fallback FP16 输入缓冲。prepared path 可直接拿到
    // 后端 A buffer；拿不到时才使用这个共享 scratch。
    S.npu_in.resize((size_t)seq * std::max(H, IS));

    for (int li = 0; li < c.num_hidden_layers; ++li) {
        TransformerLayer& L = *layers_[li];
        const bool mlp_profile = qwen2_mlp_profile_layer_enabled(li);

        uint16_t* qkv_input_f16 = nullptr;
        bool qkv_input_prepared = false;

        // ---- 1. Input LayerNorm (CPU, direct FP16 for NPU input) ----
        // 输出直接写 FP16，是因为下一步 qkv_proj 要进 NPU。若后端支持
        // prepare_input_f16()，这里会直接写入后端 A buffer。
        profile_block(prof.rmsnorm_us, [&]() {
            qkv_input_f16 = L.qkv_proj ? L.qkv_proj->prepare_input_f16(seq) : nullptr;
            qkv_input_prepared = (qkv_input_f16 != nullptr);
            if (qwen2_batch_trace_enabled()) {
                std::fprintf(stderr,
                             "[batch] qkv_proj rows=%d runM=%d path=%s\n",
                             seq, qkv_input_prepared ? seq : 0,
                             qkv_input_prepared ? "prepared" : "fallback");
            }
            if (!qkv_input_f16) {
                qkv_input_f16 = S.npu_in.data();
            }
            op_rmsnorm_to_f16(S.hidden.data(), L.input_layernorm.data(),
                              qkv_input_f16, seq, H, c.rms_norm_eps);
        });

        // ---- 2. Q / K / V proj (NPU, batched by prompt rows) ----
        // qkv fused 输出布局为 [Q, K, V]。Q/K/V 加 bias 后转成 FP32，
        // 后续 RoPE 和 attention 都在 CPU 上执行。
        S.q.resize((size_t)seq * H);
        S.k.resize((size_t)seq * kv_dim);
        S.v.resize((size_t)seq * kv_dim);
        profile_block(prof.qkv_proj_us, [&]() {
            if (L.qkv_proj) {
                const int qkv_dim = H + kv_dim * 2;
                const uint16_t* qkv_out = nullptr;
                if (qkv_input_prepared) {
                    qkv_out = L.qkv_proj->forward_prepared_output_f16();
                    if (!qkv_out) {
                        throw std::runtime_error("Linear prepared output failed: qkv_proj");
                    }
                } else {
                    S.qkv_f16.resize((size_t)seq * qkv_dim);
                    run_linear_batched_or_throw(L.qkv_proj, "qkv_proj",
                                                qkv_input_f16, seq, H, qkv_dim,
                                                S.qkv_f16.data());
                    qkv_out = S.qkv_f16.data();
                }
                for (int sidx = 0; sidx < seq; ++sidx) {
                    const uint16_t* row = qkv_out + (size_t)sidx * qkv_dim;
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
                                            qkv_input_f16, seq, H, H,
                                            S.q_f16.data());
                run_linear_batched_or_throw(L.k_proj, "k_proj",
                                            qkv_input_f16, seq, H, kv_dim,
                                            S.k_f16.data());
                run_linear_batched_or_throw(L.v_proj, "v_proj",
                                            qkv_input_f16, seq, H, kv_dim,
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
        // RoPE 必须按绝对位置 pos+sidx 应用；prefill 和 decode 共用同一逻辑。
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
        // 当前 token 的 K/V 写入缓存后，attention 就能看到“历史 + 当前”。
        // KV 存 FP16 降低内存带宽和容量压力。
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
        // prefill 需要 causal mask，decode(seq=1) 没有 future token，
        // 使用专用快路径按 head 并行。
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
        // o_proj 的结果直接累加到 residual hidden，避免先落一个完整 FP16 输出。
        profile_block(prof.o_proj_us, [&]() {
            run_linear_f32_accumulate_or_throw(L.o_proj, "o_proj",
                                               S.attn_out.data(), seq, H, H,
                                               S.npu_in, S.npu_out,
                                               S.hidden.data());
        });

        // ---- 8. Post-Attention LayerNorm (CPU, direct FP16 for NPU input) ----
        bool fused_gate_up = (bool)L.gate_up_proj;
        uint16_t* gate_up_input_f16 = nullptr;
        bool gate_up_input_prepared = false;
        profile_block(prof.rmsnorm_us, [&]() {
            const int64_t t0 = mlp_profile ? qwen2_now_us() : 0;
            gate_up_input_f16 = fused_gate_up
                ? L.gate_up_proj->prepare_input_f16(seq)
                : nullptr;
            const int64_t t_prepare = mlp_profile ? qwen2_now_us() : 0;
            gate_up_input_prepared = (gate_up_input_f16 != nullptr);
            if (qwen2_batch_trace_enabled() && fused_gate_up) {
                std::fprintf(stderr,
                             "[batch] gate_up_proj rows=%d runM=%d path=%s\n",
                             seq, gate_up_input_prepared ? seq : 0,
                             gate_up_input_prepared ? "prepared" : "fallback");
            }
            if (!gate_up_input_f16) {
                gate_up_input_f16 = S.npu_in.data();
            }
            op_rmsnorm_to_f16(S.hidden.data(), L.post_attention_layernorm.data(),
                              gate_up_input_f16, seq, H, c.rms_norm_eps);
            if (mlp_profile) {
                const int64_t t1 = qwen2_now_us();
                std::fprintf(stderr,
                             "[mlp_profile] layer=%d seq=%d gate_up_input prepare=%.3f ms rmsnorm_to_f16=%.3f ms path=%s\n",
                             li, seq, us_to_ms(t_prepare - t0),
                             us_to_ms(t1 - t_prepare),
                             gate_up_input_prepared ? "prepared" : "fallback");
            }
        });

        // ---- 9. FFN: gate & up proj (NPU, batched by prompt rows) ----
        // fused gate_up 输出逻辑为 [gate, up]。三核 gate_up_pair 快路径下，
        // 不拼回完整输出，而是直接返回每个 shard 的 [gate_slice, up_slice]。
        const uint16_t* gate_up_output_f16 = nullptr;
        std::array<const uint16_t*, 4> gate_up_shard_outputs{};
        std::array<int, 4> gate_up_shard_offsets{};
        std::array<int, 4> gate_up_shard_sizes{};
        int gate_up_shard_count = 0;
        bool gate_up_output_sharded_pairs = false;
        profile_block(prof.gate_up_proj_us, [&]() {
            const int64_t t0 = mlp_profile ? qwen2_now_us() : 0;
            if (fused_gate_up) {
                if (gate_up_input_prepared) {
                    if (L.gate_up_proj->prepared_output_shards_are_gate_up_pairs()) {
                        if (!L.gate_up_proj->forward_prepared_output_shards_f16(
                                gate_up_shard_outputs.data(),
                                gate_up_shard_offsets.data(),
                                gate_up_shard_sizes.data(),
                                (int)gate_up_shard_outputs.size(),
                                &gate_up_shard_count)) {
                            throw std::runtime_error("Linear prepared shard output failed: gate_up_proj");
                        }
                        gate_up_output_sharded_pairs = true;
                    } else {
                        gate_up_output_f16 = L.gate_up_proj->forward_prepared_output_f16();
                        if (!gate_up_output_f16) {
                            throw std::runtime_error("Linear prepared output failed: gate_up_proj");
                        }
                    }
                } else {
                    S.gate_up_f16.resize((size_t)seq * IS * 2);
                    run_linear_batched_or_throw(L.gate_up_proj, "gate_up_proj",
                                                gate_up_input_f16, seq, H, IS * 2,
                                                S.gate_up_f16.data());
                    gate_up_output_f16 = S.gate_up_f16.data();
                }
            } else {
                S.gate_f16.resize((size_t)seq * IS);
                S.up_f16.resize((size_t)seq * IS);
                run_linear_batched_or_throw(L.gate_proj, "gate_proj",
                                            gate_up_input_f16, seq, H, IS,
                                            S.gate_f16.data());
                run_linear_batched_or_throw(L.up_proj, "up_proj",
                                            gate_up_input_f16, seq, H, IS,
                                            S.up_f16.data());
            }
            if (mlp_profile) {
                std::fprintf(stderr,
                             "[mlp_profile] layer=%d seq=%d gate_up_linear total=%.3f ms fused=%d sharded_pair=%d shards=%d\n",
                             li, seq, us_to_ms(qwen2_now_us() - t0),
                             fused_gate_up ? 1 : 0,
                             gate_up_output_sharded_pairs ? 1 : 0,
                             gate_up_shard_count);
            }
        });

        uint16_t* ffn_down_input = nullptr;
        bool ffn_down_input_prepared = false;

        // SiLU(gate) * up
        // SwiGLU 的非线性仍在 CPU 做。若 down_proj 支持 prepared input，
        // 这里直接把结果写入 down_proj 的 A buffer。
        profile_block(prof.silu_mul_us, [&]() {
            const int64_t t0 = mlp_profile ? qwen2_now_us() : 0;
            const float* silu_lut = silu_f16_lut().data();
            const int64_t t_lut = mlp_profile ? qwen2_now_us() : 0;
            ffn_down_input = L.down_proj ? L.down_proj->prepare_input_f16(seq) : nullptr;
            const int64_t t_prepare = mlp_profile ? qwen2_now_us() : 0;
            ffn_down_input_prepared = (ffn_down_input != nullptr);
            if (qwen2_batch_trace_enabled()) {
                std::fprintf(stderr,
                             "[batch] down_proj rows=%d runM=%d path=%s\n",
                             seq, ffn_down_input_prepared ? seq : 0,
                             ffn_down_input_prepared ? "prepared" : "fallback");
            }
            if (!ffn_down_input) {
                S.ffn_in_f16.resize((size_t)seq * IS);
                ffn_down_input = S.ffn_in_f16.data();
            }

            if (fused_gate_up) {
                if (gate_up_output_sharded_pairs) {
                    for (int sidx = 0; sidx < seq; ++sidx) {
                        uint16_t* out_row = ffn_down_input + (size_t)sidx * IS;
                        for (int si = 0; si < gate_up_shard_count; ++si) {
                            const int offset = gate_up_shard_offsets[(size_t)si];
                            const int size = gate_up_shard_sizes[(size_t)si];
                            const uint16_t* row = gate_up_shard_outputs[(size_t)si]
                                + (size_t)sidx * size * 2;
                            swiglu_lut_to_f16(row, row + size,
                                              out_row + offset, size, silu_lut);
                        }
                    }
                } else {
                    for (int sidx = 0; sidx < seq; ++sidx) {
                        const uint16_t* row = gate_up_output_f16 + (size_t)sidx * IS * 2;
                        uint16_t* out = ffn_down_input + (size_t)sidx * IS;
                        swiglu_lut_to_f16(row, row + IS, out, IS, silu_lut);
                    }
                }
            } else {
                swiglu_lut_to_f16(S.gate_f16.data(), S.up_f16.data(),
                                  ffn_down_input, seq * IS, silu_lut);
            }
            if (mlp_profile) {
                const int64_t t1 = qwen2_now_us();
                std::fprintf(stderr,
                             "[mlp_profile] layer=%d seq=%d swiglu lut=%.3f ms down_prepare=%.3f ms compute=%.3f ms path=%s down_input=%s elems=%d\n",
                             li, seq, us_to_ms(t_lut - t0),
                             us_to_ms(t_prepare - t_lut),
                             us_to_ms(t1 - t_prepare),
                             gate_up_output_sharded_pairs ? "sharded_pair" : "full",
                             ffn_down_input_prepared ? "prepared" : "fallback",
                             seq * IS);
            }
        });

        // ---- 10. down proj (NPU, batched by prompt rows) ----
        // down_proj 输出是 FFN residual 分支，直接累加回 hidden。
        profile_block(prof.down_proj_us, [&]() {
            const int64_t t0 = mlp_profile ? qwen2_now_us() : 0;
            if (ffn_down_input_prepared) {
                if (!L.down_proj || !L.down_proj->forward_prepared_accumulate(S.hidden.data())) {
                    throw std::runtime_error("Linear prepared accumulate failed: down_proj");
                }
            } else {
                run_linear_accumulate_or_throw(L.down_proj, "down_proj",
                                               S.ffn_in_f16.data(), seq, IS, H,
                                               S.ffn_out_f16, S.hidden.data());
            }
            if (mlp_profile) {
                std::fprintf(stderr,
                             "[mlp_profile] layer=%d seq=%d down_total=%.3f ms path=%s\n",
                             li, seq, us_to_ms(qwen2_now_us() - t0),
                             ffn_down_input_prepared ? "prepared_accumulate" : "fallback_accumulate");
            }
        });
    }

    // ---- Final LayerNorm（仅取最后一个 token，因为只需要下一个 token）----
    // 即使 prefill 输入了多个 token，生成下一个 token 只依赖最后一行 hidden。
    uint16_t* lm_input_f16 = lm_head_ ? lm_head_->prepare_input_f16(1) : nullptr;
    const bool lm_input_prepared = (lm_input_f16 != nullptr);
    profile_block(prof.rmsnorm_us, [&]() {
        if (lm_input_prepared) {
            op_rmsnorm_to_f16(S.hidden.data() + (seq - 1) * H, norm_weight_.data(),
                              lm_input_f16, 1, H, c.rms_norm_eps);
        } else {
            S.last.resize(H);
            op_rmsnorm(S.hidden.data() + (seq - 1) * H, norm_weight_.data(), S.last.data(),
                       1, H, c.rms_norm_eps);
        }
    });

    // ---- lm_head ----
    // 这里只需要 greedy argmax，不需要完整 vocab logits。NPU 后端提供
    // forward_argmax/forward_prepared_argmax，避免 lm_head 大输出拼接回 CPU。
    int next_token = 0;
    profile_block(prof.lm_head_us, [&]() {
        if (lm_input_prepared) {
            if (!lm_head_ || !lm_head_->forward_prepared_argmax(&next_token)) {
                throw std::runtime_error("Linear prepared argmax failed: lm_head");
            }
        } else {
            S.lm_in.resize(H);
            op_f32_to_f16(S.last.data(), S.lm_in.data(), H);
            if (!lm_head_ || !lm_head_->forward_argmax(S.lm_in.data(), 1, &next_token)) {
                throw std::runtime_error("Linear argmax forward failed: lm_head");
            }
        }
    });

    // 更新 KV Cache 位置。下一次 decode 会从 total_len 继续追加。
    kv_cache_.set_cur_pos(total_len);

    if (profile) {
        prof.total_us = qwen2_now_us() - total_t0;
        print_forward_profile(seq, pos, total_len, prof);
    }

    return next_token;
}
