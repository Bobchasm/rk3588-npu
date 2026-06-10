#include "ops/op_attention.h"
#include "ops/op_cast.h"
#include "ops/op_softmax.h"
#include "core/half.h"
#include "rknn_matmul_api.h"

#include <algorithm>
#include <cfloat>
#include <condition_variable>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace {

int align_up(int v, int align) {
    return ((v + align - 1) / align) * align;
}

inline float dot_f32_f16(const float* x, const uint16_t* y, int n) {
    int i = 0;
    float sum = 0.0f;
#if defined(__aarch64__)
    float32x4_t acc = vdupq_n_f32(0.0f);
    for (; i + 4 <= n; i += 4) {
        float32x4_t xv = vld1q_f32(x + i);
        float16x4_t yh = vld1_f16(reinterpret_cast<const float16_t*>(y + i));
        float32x4_t yv = vcvt_f32_f16(yh);
        acc = vfmaq_f32(acc, xv, yv);
    }
    sum = vaddvq_f32(acc);
#endif
    for (; i < n; ++i) {
        sum += x[i] * f16_to_f32(y[i]);
    }
    return sum;
}

inline void add_weighted_f16(float* out, const uint16_t* v, float w, int n) {
    int i = 0;
#if defined(__aarch64__)
    float32x4_t weight = vdupq_n_f32(w);
    for (; i + 4 <= n; i += 4) {
        float32x4_t ov = vld1q_f32(out + i);
        float16x4_t vh = vld1_f16(reinterpret_cast<const float16_t*>(v + i));
        float32x4_t vv = vcvt_f32_f16(vh);
        ov = vfmaq_f32(ov, vv, weight);
        vst1q_f32(out + i, ov);
    }
#endif
    for (; i < n; ++i) {
        out[i] += w * f16_to_f32(v[i]);
    }
}

int env_int_or_default(const char* name, int fallback, int min_value, int max_value) {
    const char* v = std::getenv(name);
    if (!v || v[0] == '\0') {
        return fallback;
    }
    char* end = nullptr;
    long parsed = std::strtol(v, &end, 10);
    if (end == v || parsed < min_value) {
        std::fprintf(stderr, "[op_attention] invalid %s=%s, use %d\n",
                     name, v, fallback);
        return fallback;
    }
    if (parsed > max_value) {
        parsed = max_value;
    }
    return (int)parsed;
}

int default_attention_threads() {
    unsigned int hw = std::thread::hardware_concurrency();
    if (hw == 0) {
        hw = 4;
    }
    return std::min<int>((int)hw, 6);
}

int attention_threads() {
    static int threads = env_int_or_default(
        "RKLLM_ATTENTION_THREADS", default_attention_threads(), 1, 16);
    return threads;
}

int attention_parallel_min_len() {
    static int min_len = env_int_or_default(
        "RKLLM_ATTENTION_PARALLEL_MIN_LEN", 64, 1, 4096);
    return min_len;
}

bool env_flag_enabled(const char* name) {
    const char* v = std::getenv(name);
    return v && v[0] != '\0' &&
           std::strcmp(v, "0") != 0 &&
           std::strcmp(v, "false") != 0 &&
           std::strcmp(v, "FALSE") != 0 &&
           std::strcmp(v, "off") != 0 &&
           std::strcmp(v, "OFF") != 0;
}

bool attention_npu_enabled() {
    static bool enabled = env_flag_enabled("RKLLM_NPU_ATTENTION");
    return enabled;
}

bool attention_npu_trace_enabled() {
    static bool enabled = env_flag_enabled("RKLLM_NPU_ATTENTION_TRACE");
    return enabled;
}

bool attention_npu_verify_enabled() {
    static bool enabled = env_flag_enabled("RKLLM_NPU_ATTENTION_VERIFY");
    return enabled;
}

bool attention_npu_group_heads_enabled() {
    static bool enabled = env_flag_enabled("RKLLM_NPU_ATTENTION_GROUP_HEADS");
    return enabled;
}

bool attention_npu_dynamic_shape_enabled() {
    static bool enabled = env_flag_enabled("RKLLM_NPU_ATTENTION_DYNAMIC");
    return enabled;
}

bool attention_npu_full_mode_enabled() {
    static bool enabled = []() {
        const char* v = std::getenv("RKLLM_NPU_ATTENTION_MODE");
        return v && (std::strcmp(v, "full") == 0 ||
                     std::strcmp(v, "FULL") == 0 ||
                     std::strcmp(v, "qk_pv") == 0 ||
                     std::strcmp(v, "QK_PV") == 0);
    }();
    return enabled;
}

float attention_npu_verify_tol() {
    static float tol = []() {
        const char* v = std::getenv("RKLLM_NPU_ATTENTION_VERIFY_TOL");
        if (!v || v[0] == '\0') {
            return 0.05f;
        }
        char* end = nullptr;
        float parsed = std::strtof(v, &end);
        if (end == v || parsed <= 0.0f) {
            std::fprintf(stderr,
                         "[op_attention] invalid RKLLM_NPU_ATTENTION_VERIFY_TOL=%s, use 0.05\n",
                         v);
            return 0.05f;
        }
        return parsed;
    }();
    return tol;
}

bool& attention_npu_disabled_after_failure() {
    static bool disabled = false;
    return disabled;
}

void disable_attention_npu_after_failure(const char* reason) {
    bool& disabled = attention_npu_disabled_after_failure();
    if (!disabled) {
        disabled = true;
        std::fprintf(stderr,
                     "[op_attention] disable NPU attention after failure: %s\n",
                     reason ? reason : "unknown");
    }
}

void softmax_masked_strided(float* scores,
                            int rows,
                            int valid_cols,
                            int stride,
                            int pos_base,
                            float scale) {
    for (int r = 0; r < rows; ++r) {
        float* row = scores + (size_t)r * stride;
        const int abs_pos = pos_base + r;

        float mx = -FLT_MAX;
        for (int c = 0; c < valid_cols; ++c) {
            float v = (c > abs_pos) ? -1e9f : row[c] * scale;
            row[c] = v;
            mx = std::max(mx, v);
        }

        float sum = 0.0f;
        for (int c = 0; c < valid_cols; ++c) {
            row[c] = std::exp(row[c] - mx);
            sum += row[c];
        }
        const float inv = 1.0f / sum;
        for (int c = 0; c < valid_cols; ++c) {
            row[c] *= inv;
        }
        for (int c = valid_cols; c < stride; ++c) {
            row[c] = 0.0f;
        }
    }
}

void softmax_masked_grouped_strided(float* scores,
                                    int seq,
                                    int group,
                                    int valid_cols,
                                    int stride,
                                    int pos_base,
                                    float scale) {
    for (int g = 0; g < group; ++g) {
        softmax_masked_strided(scores + (size_t)g * seq * stride,
                               seq, valid_cols, stride, pos_base, scale);
    }
}

class NpuRuntimeMatmul {
public:
    NpuRuntimeMatmul() = default;
    ~NpuRuntimeMatmul() { destroy(); }

    NpuRuntimeMatmul(const NpuRuntimeMatmul&) = delete;
    NpuRuntimeMatmul& operator=(const NpuRuntimeMatmul&) = delete;

    bool ensure(int M, int K, int N) {
        if (ctx_ && M_ == M && K_ == K && N_ == N) {
            return true;
        }
        destroy();
        return create(M, K, N);
    }

    uint16_t* A_f16() {
        return A_mem_ ? reinterpret_cast<uint16_t*>(A_mem_->virt_addr) : nullptr;
    }

    // B 是 native layout，CPU 不能直接按 (K, N) 行优先写。
    // 调用方先填 B_normal()（普通 (K, N) f16 矩阵），run() 时再转成 native 写入 B_mem。
    uint16_t* B_normal() {
        return B_normal_.data();
    }

    float* C_f32() {
        return C_mem_ ? reinterpret_cast<float*>(C_mem_->virt_addr) : nullptr;
    }

    bool run(const char* name) {
        if (!ctx_ || !A_mem_ || !B_mem_ || !C_mem_) {
            return false;
        }
        // B 用 native layout（和 npu_linear / probe_matmul_types 一致）：
        // B_layout=0 + 运行时变化的 dmabuf 在 RK3588 实测会读不到新数据，
        // 现象正是输出重复/截断。每次 run 前把 CPU 侧 normal layout B
        // 转成 native 写到 dmabuf，再 SYNC_TO_DEVICE。
        int ret = rknn_B_normal_layout_to_native_layout(
            B_normal_.data(), B_mem_->virt_addr, K_, N_, &info_);
        if (ret < 0) {
            std::fprintf(stderr,
                         "[op_attention] %s rknn_B_normal_layout_to_native_layout failed: %d\n",
                         name ? name : "matmul", ret);
            return false;
        }
        // RKNN dmabuf 是 cacheable 的：CPU 写完 A/B 必须 SYNC_TO_DEVICE，
        // run 完后 SYNC_FROM_DEVICE 让 CPU 看到 NPU 写入的 C。
        ret = rknn_mem_sync(ctx_, A_mem_, RKNN_MEMORY_SYNC_TO_DEVICE);
        if (ret >= 0) ret = rknn_mem_sync(ctx_, B_mem_, RKNN_MEMORY_SYNC_TO_DEVICE);
        if (ret < 0) {
            std::fprintf(stderr,
                         "[op_attention] %s rknn_mem_sync(TO_DEVICE) failed: %d\n",
                         name ? name : "matmul", ret);
            return false;
        }
        ret = rknn_matmul_run(ctx_);
        if (ret < 0) {
            std::fprintf(stderr,
                         "[op_attention] %s rknn_matmul_run failed: %d\n",
                         name ? name : "matmul", ret);
            return false;
        }
        ret = rknn_mem_sync(ctx_, C_mem_, RKNN_MEMORY_SYNC_FROM_DEVICE);
        if (ret < 0) {
            std::fprintf(stderr,
                         "[op_attention] %s rknn_mem_sync(FROM_DEVICE) failed: %d\n",
                         name ? name : "matmul", ret);
            return false;
        }
        return true;
    }

    void verify_stage_once(const char* name, int valid_cols = -1) {
        if (!attention_npu_verify_enabled() || stage_verified_) {
            return;
        }
        stage_verified_ = true;

        const uint16_t* A = A_f16();
        const uint16_t* B = B_normal();
        const float* C = C_f32();
        if (!A || !B || !C || M_ <= 0 || K_ <= 0 || N_ <= 0) {
            return;
        }

        const int compare_N = (valid_cols > 0 && valid_cols < N_) ? valid_cols : N_;
        float max_abs = 0.0f;
        float max_rel = 0.0f;
        int max_m = 0;
        int max_n = 0;
        float max_ref = 0.0f;
        float max_got = 0.0f;

        for (int m = 0; m < M_; ++m) {
            for (int n = 0; n < compare_N; ++n) {
                float ref = 0.0f;
                for (int k = 0; k < K_; ++k) {
                    ref += f16_to_f32(A[(size_t)m * K_ + k]) *
                           f16_to_f32(B[(size_t)k * N_ + n]);
                }
                const float got = C[(size_t)m * N_ + n];
                const float diff = std::fabs(got - ref);
                const float rel = diff / std::max(std::fabs(ref), 1e-6f);
                if (diff > max_abs) {
                    max_abs = diff;
                    max_rel = rel;
                    max_m = m;
                    max_n = n;
                    max_ref = ref;
                    max_got = got;
                }
            }
        }

        std::fprintf(stderr,
                     "[op_attention] verify_%s M=%d K=%d N=%d validN=%d max_abs=%.6g max_rel=%.6g at=(%d,%d) npu=%.6g cpu_f16=%.6g\n",
                     name ? name : "matmul", M_, K_, N_, compare_N,
                     max_abs, max_rel, max_m, max_n, max_got, max_ref);
    }

    void destroy() {
        if (A_mem_) { rknn_destroy_mem(ctx_, A_mem_); A_mem_ = nullptr; }
        if (B_mem_) { rknn_destroy_mem(ctx_, B_mem_); B_mem_ = nullptr; }
        if (C_mem_) { rknn_destroy_mem(ctx_, C_mem_); C_mem_ = nullptr; }
        if (ctx_) { rknn_matmul_destroy(ctx_); ctx_ = 0; }
        io_attr_ = {};
        b_attr_ = {};
        info_ = {};
        B_normal_.clear();
        B_normal_.shrink_to_fit();
        stage_verified_ = false;
        M_ = K_ = N_ = 0;
    }

private:
    bool create(int M, int K, int N) {
        if (M <= 0 || K <= 0 || N <= 0) {
            return false;
        }

        rknn_matmul_info info{};
        info.M = M;
        info.K = K;
        info.N = N;
        info.type = RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32;
        info.B_layout = RKNN_MM_LAYOUT_NATIVE;
        info.AC_layout = RKNN_MM_LAYOUT_NORM;
        info.B_quant_type = 0;
        info.AC_quant_type = 0;

        int ret = 0;
        if (M > 1 && attention_npu_dynamic_shape_enabled()) {
            info.M = 1;
            rknn_matmul_shape shapes[2] = {
                {1, K, N},
                {M, K, N},
            };
            rknn_matmul_io_attr attrs[2] = {};
            ret = rknn_matmul_create_dynamic_shape(&ctx_, &info, 2, shapes, attrs);
            if (ret < 0) {
                std::fprintf(stderr,
                             "[op_attention] rknn_matmul_create_dynamic_shape failed: %d (M=%d K=%d N=%d)\n",
                             ret, M, K, N);
                destroy();
                return false;
            }
            io_attr_ = attrs[1];
            b_attr_ = attrs[0].B;
            ret = rknn_matmul_set_dynamic_shape(ctx_, &shapes[1]);
            if (ret < 0) {
                std::fprintf(stderr,
                             "[op_attention] rknn_matmul_set_dynamic_shape failed: %d (M=%d K=%d N=%d)\n",
                             ret, M, K, N);
                destroy();
                return false;
            }
        } else {
            ret = rknn_matmul_create(&ctx_, &info, &io_attr_);
            if (ret < 0) {
                std::fprintf(stderr,
                             "[op_attention] rknn_matmul_create failed: %d (M=%d K=%d N=%d)\n",
                             ret, M, K, N);
                destroy();
                return false;
            }
            b_attr_ = io_attr_.B;
        }

        A_mem_ = rknn_create_mem(ctx_, io_attr_.A.size);
        B_mem_ = rknn_create_mem(ctx_, b_attr_.size);
        C_mem_ = rknn_create_mem(ctx_, io_attr_.C.size);
        if (!A_mem_ || !B_mem_ || !C_mem_) {
            std::fprintf(stderr,
                         "[op_attention] rknn_create_mem failed (M=%d K=%d N=%d)\n",
                         M, K, N);
            destroy();
            return false;
        }

        ret = rknn_matmul_set_io_mem(ctx_, A_mem_, &io_attr_.A);
        if (ret >= 0) ret = rknn_matmul_set_io_mem(ctx_, B_mem_, &b_attr_);
        if (ret >= 0) ret = rknn_matmul_set_io_mem(ctx_, C_mem_, &io_attr_.C);
        if (ret < 0) {
            std::fprintf(stderr,
                         "[op_attention] rknn_matmul_set_io_mem failed: %d (M=%d K=%d N=%d)\n",
                         ret, M, K, N);
            destroy();
            return false;
        }

        M_ = M;
        K_ = K;
        N_ = N;
        info_ = info;
        B_normal_.assign((size_t)K_ * N_, (uint16_t)0);
        return true;
    }

    int M_ = 0;
    int K_ = 0;
    int N_ = 0;
    rknn_matmul_info info_{};
    rknn_matmul_ctx ctx_ = 0;
    rknn_matmul_io_attr io_attr_{};
    rknn_matmul_tensor_attr b_attr_{};
    rknn_tensor_mem* A_mem_ = nullptr;
    rknn_tensor_mem* B_mem_ = nullptr;
    rknn_tensor_mem* C_mem_ = nullptr;
    std::vector<uint16_t> B_normal_;
    bool stage_verified_ = false;
};

class NpuAttentionPrefillRunner {
public:
    bool run(const float* q,
             const uint16_t* k_cache,
             const uint16_t* v_cache,
             float* out,
             int seq,
             int total_len,
             int n_heads,
             int n_kv_heads,
             int head_dim,
             int pos_base) {
        if (!q || !k_cache || !v_cache || !out ||
            seq <= 1 || total_len <= 0 || n_heads <= 0 ||
            n_kv_heads <= 0 || head_dim <= 0 ||
            (n_heads % n_kv_heads) != 0) {
            return false;
        }

        // RK3588 FP16 matmul requires K aligned to 32 and N aligned to 16.
        // Use one aligned sequence length for QK's N and PV's K.
        const int total_aligned = align_up(total_len, 32);
        if ((head_dim % 32) != 0 || total_aligned > 10240) {
            return false;
        }

        const int hidden = n_heads * head_dim;
        const int kv_dim = n_kv_heads * head_dim;
        const int group = n_heads / n_kv_heads;
        const bool full_mode = attention_npu_full_mode_enabled();
        const bool group_heads = attention_npu_group_heads_enabled();
        const bool batch_heads = full_mode ? group_heads : true;
        const int rows = seq * (batch_heads ? group : 1);
        const float scale = 1.0f / std::sqrt((float)head_dim);

        if ((full_mode && !qk_.ensure(rows, head_dim, total_aligned)) ||
            !pv_.ensure(rows, total_aligned, head_dim)) {
            disable_attention_npu_after_failure("create context");
            return false;
        }

        uint16_t* qk_A = full_mode ? qk_.A_f16() : nullptr;
        uint16_t* qk_B = full_mode ? qk_.B_normal() : nullptr;
        float* qk_C = full_mode ? qk_.C_f32() : nullptr;
        uint16_t* pv_A = pv_.A_f16();
        uint16_t* pv_B = pv_.B_normal();
        float* pv_C = pv_.C_f32();
        if ((full_mode && (!qk_A || !qk_B || !qk_C)) ||
            !pv_A || !pv_B || !pv_C) {
            disable_attention_npu_after_failure("null tensor mem");
            return false;
        }

        if (attention_npu_trace_enabled() && !logged_shape_) {
            logged_shape_ = true;
            std::fprintf(stderr,
                         "[op_attention] NPU prefill attention mode=%s seq=%d total_len=%d aligned=%d heads=%d kv_heads=%d group=%d rows=%d head_dim=%d\n",
                         full_mode ? "full" : "pv",
                         seq, total_len, total_aligned,
                         n_heads, n_kv_heads, group, rows, head_dim);
        }

        static thread_local std::vector<float> scores_buf;
        if (!full_mode) {
            scores_buf.resize((size_t)rows * total_aligned);
        }

        for (int kv_h = 0; kv_h < n_kv_heads; ++kv_h) {
            if (full_mode) {
                fill_k_transposed(qk_B, k_cache, total_len, total_aligned,
                                  kv_dim, kv_h, head_dim);
            }
            fill_v_matrix(pv_B, v_cache, total_len, total_aligned,
                          kv_dim, kv_h, head_dim);

            const int h_begin = kv_h * group;
            if (full_mode && group_heads) {
                fill_q_group_matrix(qk_A, q, seq, hidden, h_begin, group, head_dim);
                if (!qk_.run("qk")) {
                    disable_attention_npu_after_failure("qk run");
                    return false;
                }
                qk_.verify_stage_once("qk", total_len);

                softmax_masked_grouped_strided(qk_C, seq, group,
                                               total_len, total_aligned,
                                               pos_base, scale);
                fill_prob_matrix(pv_A, qk_C, rows, total_len,
                                 total_aligned, total_aligned);

                if (!pv_.run("pv")) {
                    disable_attention_npu_after_failure("pv run");
                    return false;
                }
                pv_.verify_stage_once("pv", head_dim);

                copy_group_output(pv_C, out, seq, hidden, h_begin, group, head_dim);
            } else if (!full_mode) {
                for (int gi = 0; gi < group; ++gi) {
                    const int h = h_begin + gi;
                    compute_cpu_scores_for_head(q, k_cache,
                                                scores_buf.data() + (size_t)gi * seq * total_aligned,
                                                seq, total_len, total_aligned,
                                                hidden, kv_dim, h, kv_h,
                                                head_dim, pos_base, scale);
                }

                fill_prob_matrix(pv_A, scores_buf.data(), rows,
                                 total_len, total_aligned, total_aligned);

                if (!pv_.run("pv")) {
                    disable_attention_npu_after_failure("pv run");
                    return false;
                }
                pv_.verify_stage_once("pv", head_dim);

                copy_group_output(pv_C, out, seq, hidden, h_begin, group, head_dim);
            } else {
                for (int gi = 0; gi < group; ++gi) {
                    const int h = h_begin + gi;
                    fill_q_group_matrix(qk_A, q, seq, hidden, h, 1, head_dim);
                    if (!qk_.run("qk")) {
                        disable_attention_npu_after_failure("qk run");
                        return false;
                    }
                    qk_.verify_stage_once("qk", total_len);

                    softmax_masked_strided(qk_C, seq, total_len, total_aligned,
                                           pos_base, scale);
                    fill_prob_matrix(pv_A, qk_C, seq, total_len,
                                     total_aligned, total_aligned);

                    if (!pv_.run("pv")) {
                        disable_attention_npu_after_failure("pv run");
                        return false;
                    }
                    pv_.verify_stage_once("pv", head_dim);

                    copy_group_output(pv_C, out, seq, hidden, h, 1, head_dim);
                }
            }
        }

        return true;
    }

private:
    static void fill_q_group_matrix(uint16_t* dst,
                                    const float* q,
                                    int seq,
                                    int hidden,
                                    int h_begin,
                                    int group,
                                    int head_dim) {
        for (int g = 0; g < group; ++g) {
            const int h = h_begin + g;
            for (int sq = 0; sq < seq; ++sq) {
                const int row = g * seq + sq;
                const float* src = q + (size_t)sq * hidden + h * head_dim;
                op_f32_to_f16(src, dst + (size_t)row * head_dim, head_dim);
            }
        }
    }

    static void fill_k_transposed(uint16_t* dst,
                                  const uint16_t* k_cache,
                                  int total_len,
                                  int total_aligned,
                                  int kv_dim,
                                  int kv_head,
                                  int head_dim) {
        for (int k = 0; k < head_dim; ++k) {
            uint16_t* row = dst + (size_t)k * total_aligned;
            for (int t = 0; t < total_len; ++t) {
                row[t] = k_cache[(size_t)t * kv_dim + kv_head * head_dim + k];
            }
            std::fill(row + total_len, row + total_aligned, (uint16_t)0);
        }
    }

    static void fill_v_matrix(uint16_t* dst,
                              const uint16_t* v_cache,
                              int total_len,
                              int total_aligned,
                              int kv_dim,
                              int kv_head,
                              int head_dim) {
        for (int t = 0; t < total_len; ++t) {
            const uint16_t* src = v_cache + (size_t)t * kv_dim +
                                  kv_head * head_dim;
            std::memcpy(dst + (size_t)t * head_dim, src,
                        (size_t)head_dim * sizeof(uint16_t));
        }
        std::fill(dst + (size_t)total_len * head_dim,
                  dst + (size_t)total_aligned * head_dim,
                  (uint16_t)0);
    }

    static void fill_prob_matrix(uint16_t* dst,
                                 const float* probs,
                                 int rows,
                                 int total_len,
                                 int src_stride,
                                 int total_aligned) {
        for (int r = 0; r < rows; ++r) {
            uint16_t* dst_row = dst + (size_t)r * total_aligned;
            const float* src_row = probs + (size_t)r * src_stride;
            op_f32_to_f16(src_row, dst_row, total_len);
            std::fill(dst_row + total_len, dst_row + total_aligned,
                      (uint16_t)0);
        }
    }

    static void compute_cpu_scores_for_head(const float* q,
                                            const uint16_t* k_cache,
                                            float* scores,
                                            int seq,
                                            int total_len,
                                            int score_stride,
                                            int hidden,
                                            int kv_dim,
                                            int head,
                                            int kv_head,
                                            int head_dim,
                                            int pos_base,
                                            float scale) {
        for (int sq = 0; sq < seq; ++sq) {
            const int abs_sq = pos_base + sq;
            float* row = scores + (size_t)sq * score_stride;
            const float* qh = q + (size_t)sq * hidden + head * head_dim;
            float mx = -FLT_MAX;
            for (int sk = 0; sk < total_len; ++sk) {
                if (sk > abs_sq) {
                    row[sk] = -1e9f;
                } else {
                    const uint16_t* kh = k_cache + (size_t)sk * kv_dim +
                                         kv_head * head_dim;
                    row[sk] = dot_f32_f16(qh, kh, head_dim) * scale;
                }
                mx = std::max(mx, row[sk]);
            }

            float sum = 0.0f;
            for (int sk = 0; sk < total_len; ++sk) {
                row[sk] = std::exp(row[sk] - mx);
                sum += row[sk];
            }
            const float inv = 1.0f / sum;
            for (int sk = 0; sk < total_len; ++sk) {
                row[sk] *= inv;
            }
            for (int sk = total_len; sk < score_stride; ++sk) {
                row[sk] = 0.0f;
            }
        }
    }

    static void copy_group_output(const float* src,
                                  float* out,
                                  int seq,
                                  int hidden,
                                  int h_begin,
                                  int group,
                                  int head_dim) {
        for (int g = 0; g < group; ++g) {
            const int h = h_begin + g;
            for (int sq = 0; sq < seq; ++sq) {
                const int row = g * seq + sq;
                std::memcpy(out + (size_t)sq * hidden + h * head_dim,
                            src + (size_t)row * head_dim,
                            (size_t)head_dim * sizeof(float));
            }
        }
    }

    NpuRuntimeMatmul qk_;
    NpuRuntimeMatmul pv_;
    bool logged_shape_ = false;
};

NpuAttentionPrefillRunner& npu_attention_prefill_runner() {
    static NpuAttentionPrefillRunner runner;
    return runner;
}

bool op_attention_npu_prefill(const float* q,
                              const uint16_t* k_cache,
                              const uint16_t* v_cache,
                              float* out,
                              int seq,
                              int total_len,
                              int n_heads,
                              int n_kv_heads,
                              int head_dim,
                              int pos_base) {
    if (!attention_npu_enabled() || attention_npu_disabled_after_failure()) {
        return false;
    }
    return npu_attention_prefill_runner().run(q, k_cache, v_cache, out,
                                              seq, total_len,
                                              n_heads, n_kv_heads,
                                              head_dim, pos_base);
}

void attention_decode_heads(const float* q,
                            const uint16_t* k_cache,
                            const uint16_t* v_cache,
                            float* out,
                            int total_len,
                            int n_heads,
                            int n_kv_heads,
                            int head_dim,
                            int h_begin,
                            int h_end,
                            std::vector<float>& scores) {
    // decode 阶段 seq=1，每个 query head 只和历史 total_len 个 K 做点积。
    // GQA 下多个 query head 共享同一个 kv head，所以 kv_h 由 h 映射得到。
    const int hidden = n_heads * head_dim;
    const int kv_dim = n_kv_heads * head_dim;
    const float scale = 1.0f / std::sqrt((float)head_dim);

    scores.resize((size_t)total_len);
    float* score_ptr = scores.data();

    for (int h = h_begin; h < h_end; ++h) {
        const int kv_h = h / (n_heads / n_kv_heads);
        const float* qh = q + h * head_dim;

        for (int sk = 0; sk < total_len; ++sk) {
            const uint16_t* kh = k_cache + (size_t)sk * kv_dim + kv_h * head_dim;
#if defined(__aarch64__)
            if (sk + 4 < total_len) {
                __builtin_prefetch(k_cache + (size_t)(sk + 4) * kv_dim + kv_h * head_dim);
            }
#endif
            const float dot = dot_f32_f16(qh, kh, head_dim);
            score_ptr[sk] = dot * scale;
        }

        op_softmax(score_ptr, 1, total_len);

        float* out_row = out + h * head_dim;
        for (int sk = 0; sk < total_len; ++sk) {
            const uint16_t* vh = v_cache + (size_t)sk * kv_dim + kv_h * head_dim;
#if defined(__aarch64__)
            if (sk + 4 < total_len) {
                __builtin_prefetch(v_cache + (size_t)(sk + 4) * kv_dim + kv_h * head_dim);
            }
#endif
            add_weighted_f16(out_row, vh, score_ptr[sk], head_dim);
        }
    }
}

class AttentionDecodePool {
public:
    explicit AttentionDecodePool(int worker_count) {
        worker_count = std::max(1, worker_count);
        workers_.reserve((size_t)worker_count);
        tasks_.resize((size_t)worker_count);
        ok_.resize((size_t)worker_count, false);
        for (int i = 0; i < worker_count; ++i) {
            workers_.emplace_back([this, i]() { worker_loop(i); });
        }
    }

    ~AttentionDecodePool() {
        {
            std::lock_guard<std::mutex> lock(mu_);
            stop_ = true;
            ++generation_;
        }
        cv_task_.notify_all();
        for (auto& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

    AttentionDecodePool(const AttentionDecodePool&) = delete;
    AttentionDecodePool& operator=(const AttentionDecodePool&) = delete;

    bool run(const float* q,
             const uint16_t* k_cache,
             const uint16_t* v_cache,
             float* out,
             int total_len,
             int n_heads,
             int n_kv_heads,
             int head_dim) {
        // decode attention 的主要循环按 query head 可并行。线程池常驻，
        // 避免每个 token 创建/销毁线程。
        if ((int)workers_.size() <= 1 || n_heads <= 1) {
            return false;
        }

        std::unique_lock<std::mutex> run_lock(run_mu_);
        {
            std::lock_guard<std::mutex> lock(mu_);
            active_count_ = std::min<int>((int)workers_.size(), n_heads);
            done_count_ = 0;
            ok_.assign(ok_.size(), false);

            int h = 0;
            for (int i = 0; i < active_count_; ++i) {
                const int remain_heads = n_heads - h;
                const int remain_workers = active_count_ - i;
                const int chunk = (remain_heads + remain_workers - 1) / remain_workers;
                Task& t = tasks_[(size_t)i];
                t.q = q;
                t.k_cache = k_cache;
                t.v_cache = v_cache;
                t.out = out;
                t.total_len = total_len;
                t.n_heads = n_heads;
                t.n_kv_heads = n_kv_heads;
                t.head_dim = head_dim;
                t.h_begin = h;
                t.h_end = h + chunk;
                h += chunk;
            }
            ++generation_;
        }
        cv_task_.notify_all();

        std::unique_lock<std::mutex> lock(mu_);
        cv_done_.wait(lock, [this]() { return done_count_ == active_count_; });
        for (int i = 0; i < active_count_; ++i) {
            if (!ok_[(size_t)i]) {
                return false;
            }
        }
        return true;
    }

private:
    struct Task {
        const float* q = nullptr;
        const uint16_t* k_cache = nullptr;
        const uint16_t* v_cache = nullptr;
        float* out = nullptr;
        int total_len = 0;
        int n_heads = 0;
        int n_kv_heads = 0;
        int head_dim = 0;
        int h_begin = 0;
        int h_end = 0;
    };

    void worker_loop(int worker_id) {
        uint64_t seen_generation = 0;
        std::vector<float> scores;
        while (true) {
            Task task;
            bool active = false;
            {
                std::unique_lock<std::mutex> lock(mu_);
                cv_task_.wait(lock, [this, seen_generation]() {
                    return stop_ || generation_ != seen_generation;
                });
                if (stop_) return;
                seen_generation = generation_;
                active = worker_id < active_count_;
                if (active) {
                    task = tasks_[(size_t)worker_id];
                }
            }

            bool ok = true;
            if (active) {
                attention_decode_heads(task.q, task.k_cache, task.v_cache, task.out,
                                       task.total_len, task.n_heads, task.n_kv_heads,
                                       task.head_dim, task.h_begin, task.h_end, scores);
            }

            if (active) {
                std::lock_guard<std::mutex> lock(mu_);
                ok_[(size_t)worker_id] = ok;
                ++done_count_;
                cv_done_.notify_one();
            }
        }
    }

    std::vector<std::thread> workers_;
    std::vector<Task> tasks_;
    std::vector<bool> ok_;
    std::mutex run_mu_;
    std::mutex mu_;
    std::condition_variable cv_task_;
    std::condition_variable cv_done_;
    uint64_t generation_ = 0;
    int active_count_ = 0;
    int done_count_ = 0;
    bool stop_ = false;
};

AttentionDecodePool& attention_decode_pool() {
    static AttentionDecodePool pool(attention_threads());
    return pool;
}

void op_attention_cpu_prefill(const float* q,
                              const uint16_t* k_cache,
                              const uint16_t* v_cache,
                              float* out,
                              int seq,
                              int total_len,
                              int n_heads,
                              int n_kv_heads,
                              int head_dim,
                              int pos_base) {
    const int hidden = n_heads * head_dim;
    const int kv_dim = n_kv_heads * head_dim;
    const float scale = 1.0f / std::sqrt((float)head_dim);

    std::memset(out, 0, sizeof(float) * seq * hidden);

    static thread_local std::vector<float> scores_buf;
    scores_buf.resize((size_t)seq * total_len);
    float* scores = scores_buf.data();

    for (int h = 0; h < n_heads; ++h) {
        int kv_h = h / (n_heads / n_kv_heads);

        for (int sq = 0; sq < seq; ++sq) {
            int abs_sq = pos_base + sq;
            for (int sk = 0; sk < total_len; ++sk) {
                if (sk > abs_sq) {
                    scores[sq * total_len + sk] = -1e9f;
                    continue;
                }
                const float* qh = q + (size_t)sq * hidden + h * head_dim;
                const uint16_t* kh = k_cache + (size_t)sk * kv_dim +
                                     kv_h * head_dim;
                const float dot = dot_f32_f16(qh, kh, head_dim);
                scores[sq * total_len + sk] = dot * scale;
            }
        }

        op_softmax(scores, seq, total_len);

        for (int sq = 0; sq < seq; ++sq) {
            float* out_row = out + (size_t)sq * hidden + h * head_dim;
            for (int sk = 0; sk < total_len; ++sk) {
                const uint16_t* vh = v_cache + (size_t)sk * kv_dim +
                                     kv_h * head_dim;
                float w = scores[sq * total_len + sk];
                add_weighted_f16(out_row, vh, w, head_dim);
            }
        }
    }
}

bool verify_attention_npu_or_fallback(const float* q,
                                      const uint16_t* k_cache,
                                      const uint16_t* v_cache,
                                      float* out,
                                      int seq,
                                      int total_len,
                                      int n_heads,
                                      int n_kv_heads,
                                      int head_dim,
                                      int pos_base) {
    if (!attention_npu_verify_enabled()) {
        return true;
    }

    const int hidden = n_heads * head_dim;
    static thread_local std::vector<float> ref;
    ref.resize((size_t)seq * hidden);
    op_attention_cpu_prefill(q, k_cache, v_cache, ref.data(),
                             seq, total_len, n_heads, n_kv_heads,
                             head_dim, pos_base);

    float max_abs = 0.0f;
    float max_rel = 0.0f;
    int max_idx = 0;
    const int count = seq * hidden;
    for (int i = 0; i < count; ++i) {
        const float diff = std::fabs(out[i] - ref[(size_t)i]);
        const float denom = std::max(std::fabs(ref[(size_t)i]), 1e-6f);
        const float rel = diff / denom;
        if (diff > max_abs) {
            max_abs = diff;
            max_rel = rel;
            max_idx = i;
        }
    }

    std::fprintf(stderr,
                 "[op_attention] verify seq=%d total_len=%d max_abs=%.6g max_rel=%.6g idx=%d npu=%.6g cpu=%.6g\n",
                 seq, total_len, max_abs, max_rel, max_idx,
                 out[max_idx], ref[(size_t)max_idx]);

    if (max_abs > attention_npu_verify_tol()) {
        std::memcpy(out, ref.data(), (size_t)count * sizeof(float));
        disable_attention_npu_after_failure("verify diff");
        return false;
    }
    return true;
}

}  // namespace

void op_attention(
    const float*    q,
    const uint16_t* k_cache,
    const uint16_t* v_cache,
    float*          out,
    int seq, int total_len,
    int n_heads, int n_kv_heads, int head_dim,
    int pos_base)
{
    // prefill 路径：seq 可以大于 1，需要 causal mask。
    // 可选 NPU 快路径只搬运两次 GEMM：QK^T 和 softmax(QK)V；
    // softmax/causal mask 仍在 CPU 做。创建或运行失败时回落 CPU。
    if (op_attention_npu_prefill(q, k_cache, v_cache, out,
                                 seq, total_len,
                                 n_heads, n_kv_heads, head_dim,
                                 pos_base) &&
        verify_attention_npu_or_fallback(q, k_cache, v_cache, out,
                                         seq, total_len,
                                         n_heads, n_kv_heads, head_dim,
                                         pos_base)) {
        return;
    }

    // CPU fallback 当前按 head 顺序计算，适合 prompt 不太长的测试场景。
    op_attention_cpu_prefill(q, k_cache, v_cache, out,
                             seq, total_len,
                             n_heads, n_kv_heads, head_dim,
                             pos_base);
}

void op_attention_decode(
    const float*    q,
    const uint16_t* k_cache,
    const uint16_t* v_cache,
    float*          out,
    int total_len,
    int n_heads, int n_kv_heads, int head_dim)
{
    // decode 路径：seq 固定为 1，当前 token 的 KV 已经写入 cache。
    // 因为没有未来 token，省掉 causal mask，并优先使用 head 并行线程池。
    const int hidden = n_heads * head_dim;

    std::memset(out, 0, sizeof(float) * hidden);

    if (total_len >= attention_parallel_min_len() &&
        attention_decode_pool().run(q, k_cache, v_cache, out,
                                    total_len, n_heads, n_kv_heads, head_dim)) {
        return;
    }

    static thread_local std::vector<float> scores_buf;
    attention_decode_heads(q, k_cache, v_cache, out,
                           total_len, n_heads, n_kv_heads, head_dim,
                           0, n_heads, scores_buf);
}
