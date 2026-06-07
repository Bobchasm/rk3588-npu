#include "ops/op_attention.h"
#include "ops/op_softmax.h"
#include "core/half.h"

#include <algorithm>
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
    const int hidden = n_heads * head_dim;
    const int kv_dim = n_kv_heads * head_dim;
    const float scale = 1.0f / std::sqrt((float)head_dim);

    std::memset(out, 0, sizeof(float) * seq * hidden);

    static thread_local std::vector<float> scores_buf;
    scores_buf.resize((size_t)seq * total_len);
    float* scores = scores_buf.data();

    for (int h = 0; h < n_heads; ++h) {
        int kv_h = h / (n_heads / n_kv_heads);  // GQA: 多个 query head 共享一个 kv head

        // scores = (q · k^T) * scale，并应用 causal mask
        for (int sq = 0; sq < seq; ++sq) {
            int abs_sq = pos_base + sq;
            for (int sk = 0; sk < total_len; ++sk) {
                if (sk > abs_sq) { scores[sq * total_len + sk] = -1e9f; continue; }
                const float*    qh = q       + sq * hidden + h    * head_dim;
                const uint16_t* kh = k_cache + sk * kv_dim + kv_h * head_dim;
                const float dot = dot_f32_f16(qh, kh, head_dim);
                scores[sq * total_len + sk] = dot * scale;
            }
        }

        op_softmax(scores, seq, total_len);

        // out += scores · v
        for (int sq = 0; sq < seq; ++sq) {
            float* out_row = out + sq * hidden + h * head_dim;
            for (int sk = 0; sk < total_len; ++sk) {
                const uint16_t* vh = v_cache + sk * kv_dim + kv_h * head_dim;
                float w = scores[sq * total_len + sk];
                add_weighted_f16(out_row, vh, w, head_dim);
            }
        }
    }
}

void op_attention_decode(
    const float*    q,
    const uint16_t* k_cache,
    const uint16_t* v_cache,
    float*          out,
    int total_len,
    int n_heads, int n_kv_heads, int head_dim)
{
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
