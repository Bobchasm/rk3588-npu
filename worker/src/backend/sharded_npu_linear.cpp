#include "backend/sharded_npu_linear.h"
#include "backend/cpu_linear.h"
#include "backend/npu_linear_i4.h"
#include "backend/npu_linear_w8.h"
#include "core/half.h"
#include "ops/op_cast.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <mutex>
#include <string>
#include <sys/time.h>
#include <thread>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace {

int align_up(int v, int align) {
    return ((v + align - 1) / align) * align;
}

int64_t now_us() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return (int64_t)tv.tv_sec * 1000000LL + tv.tv_usec;
}

double us_to_ms(int64_t us) {
    return (double)us / 1000.0;
}

bool env_enabled(const char* name) {
    const char* v = std::getenv(name);
    return v && v[0] != '\0' &&
           std::strcmp(v, "0") != 0 &&
           std::strcmp(v, "false") != 0 &&
           std::strcmp(v, "FALSE") != 0 &&
           std::strcmp(v, "off") != 0 &&
           std::strcmp(v, "OFF") != 0;
}

bool mlp_profile_enabled() {
    return env_enabled("RKLLM_MLP_PROFILE");
}

int mlp_profile_layer_filter() {
    static int cached = []() {
        const char* v = std::getenv("RKLLM_MLP_PROFILE_LAYER");
        if (!v || v[0] == '\0') {
            return -1;
        }
        char* end = nullptr;
        long parsed = std::strtol(v, &end, 10);
        if (end == v || parsed < 0) {
            return -1;
        }
        return (int)parsed;
    }();
    return cached;
}

int layer_index_from_cache_key(const std::string& key) {
    const char prefix[] = "model.layers.";
    const size_t p = key.find(prefix);
    if (p == std::string::npos) {
        return -1;
    }
    const char* s = key.c_str() + p + sizeof(prefix) - 1;
    if (!std::isdigit((unsigned char)*s)) {
        return -1;
    }
    char* end = nullptr;
    long parsed = std::strtol(s, &end, 10);
    return end == s ? -1 : (int)parsed;
}

bool mlp_profile_enabled_for_key(const std::string& key) {
    if (!mlp_profile_enabled() || key.find(".mlp.") == std::string::npos) {
        return false;
    }
    const int filter = mlp_profile_layer_filter();
    return filter < 0 || layer_index_from_cache_key(key) == filter;
}

inline uint16_t f16_order_key(uint16_t v) {
    return (v & 0x8000u) ? (uint16_t)~v : (uint16_t)(v ^ 0x8000u);
}

void add_f32_inplace(float* dst, const float* src, int n) {
    int i = 0;
#if defined(__aarch64__)
    for (; i + 4 <= n; i += 4) {
        const float32x4_t d = vld1q_f32(dst + i);
        const float32x4_t s = vld1q_f32(src + i);
        vst1q_f32(dst + i, vaddq_f32(d, s));
    }
#endif
    for (; i < n; ++i) {
        dst[i] += src[i];
    }
}

void f32_to_f16_buffer(const float* src, uint16_t* dst, int n) {
    int i = 0;
#if defined(__aarch64__)
    for (; i + 4 <= n; i += 4) {
        const float32x4_t f = vld1q_f32(src + i);
        const float16x4_t h = vcvt_f16_f32(f);
        vst1_f16(reinterpret_cast<float16_t*>(dst + i), h);
    }
#endif
    for (; i < n; ++i) {
        dst[i] = f32_to_f16(src[i]);
    }
}

class KSplitNpuLinear : public ILinearOp {
public:
    explicit KSplitNpuLinear(int chunk_k, rknn_core_mask core_mask)
        : chunk_k_(std::max(32, chunk_k)), core_mask_(core_mask) {}

    bool init(int K, int N, const uint16_t* weight_kn) override {
        destroy();
        if (K <= 0 || N <= 0 || !weight_kn) {
            std::fprintf(stderr, "[KSplitNpuLinear] invalid init args K=%d N=%d\n", K, N);
            return false;
        }
        K_ = K;
        N_ = N;

        int offset = 0;
        int chunk_index = 0;
        while (offset < K_) {
            int ck = std::min(chunk_k_, K_ - offset);
            // RK3588 FP16 matmul K requires 32-byte alignment, i.e. 16 FP16
            // elements. Keep non-final chunks aligned; Qwen down K=8960 is
            // divisible by common chunk sizes.
            if (offset + ck < K_) {
                ck = std::max(16, (ck / 16) * 16);
            }
            if (ck <= 0) {
                std::fprintf(stderr, "[KSplitNpuLinear] bad chunk split K=%d chunk_k=%d\n",
                             K_, chunk_k_);
                destroy();
                return false;
            }

            std::unique_ptr<NpuLinear> chunk(new NpuLinear());
            chunk->set_core_mask(core_mask_);
            chunk->set_output_f32(true);
            if (!cache_key_.empty()) {
                chunk->set_cache_key(cache_key_ + ".ksplit" + std::to_string(chunk_index));
            }
            if (!chunk->init(ck, N_, weight_kn + (size_t)offset * N_)) {
                std::fprintf(stderr,
                             "[KSplitNpuLinear] chunk %d init failed offset=%d K=%d N=%d\n",
                             chunk_index, offset, ck, N_);
                destroy();
                return false;
            }
            chunks_.push_back(std::move(chunk));
            offsets_.push_back(offset);
            sizes_.push_back(ck);
            offset += ck;
            ++chunk_index;
        }

        std::fprintf(stderr, "[KSplitNpuLinear] init K=%d N=%d chunks:", K_, N_);
        for (size_t i = 0; i < sizes_.size(); ++i) {
            std::fprintf(stderr, "%s%d", i == 0 ? " " : "/", sizes_[i]);
        }
        std::fprintf(stderr, " f32_accumulate\n");
        return true;
    }

    void set_cache_key(const std::string& key) override {
        cache_key_ = key;
        for (size_t i = 0; i < chunks_.size(); ++i) {
            chunks_[i]->set_cache_key(cache_key_ + ".ksplit" + std::to_string(i));
        }
    }

    bool init_from_cache(int K, int N) override {
        destroy();
        if (K <= 0 || N <= 0 || cache_key_.empty()) {
            return false;
        }
        K_ = K;
        N_ = N;

        int offset = 0;
        int chunk_index = 0;
        while (offset < K_) {
            int ck = std::min(chunk_k_, K_ - offset);
            if (offset + ck < K_) {
                ck = std::max(16, (ck / 16) * 16);
            }
            if (ck <= 0) {
                destroy();
                return false;
            }
            std::unique_ptr<NpuLinear> chunk(new NpuLinear());
            chunk->set_core_mask(core_mask_);
            chunk->set_output_f32(true);
            chunk->set_cache_key(cache_key_ + ".ksplit" + std::to_string(chunk_index));
            if (!chunk->init_from_cache(ck, N_)) {
                destroy();
                return false;
            }
            chunks_.push_back(std::move(chunk));
            offsets_.push_back(offset);
            sizes_.push_back(ck);
            offset += ck;
            ++chunk_index;
        }

        std::fprintf(stderr, "[KSplitNpuLinear] init K=%d N=%d chunks:", K_, N_);
        for (size_t i = 0; i < sizes_.size(); ++i) {
            std::fprintf(stderr, "%s%d", i == 0 ? " " : "/", sizes_[i]);
        }
        std::fprintf(stderr, " f32_accumulate cache\n");
        return true;
    }

    bool forward(const uint16_t* input_f16, int M, uint16_t* output_f16) override {
        if (!input_f16 || !output_f16 || !run_accumulate(input_f16, M)) {
            return false;
        }
        f32_to_f16_buffer(accum_.data(), output_f16, M * N_);
        return true;
    }

    uint16_t* prepare_input_f16(int M) override {
        prepared_input_.resize((size_t)M * K_);
        prepared_M_ = M;
        return prepared_input_.data();
    }

    bool forward_prepared(uint16_t* output_f16) override {
        if (!output_f16 || prepared_input_.empty() || prepared_M_ <= 0) {
            return false;
        }
        return forward(prepared_input_.data(), prepared_M_, output_f16);
    }

    const uint16_t* forward_prepared_output_f16() override {
        if (prepared_input_.empty() || prepared_M_ <= 0 ||
            !run_accumulate(prepared_input_.data(), prepared_M_)) {
            return nullptr;
        }
        prepared_output_.resize((size_t)prepared_M_ * N_);
        f32_to_f16_buffer(accum_.data(), prepared_output_.data(), prepared_M_ * N_);
        return prepared_output_.data();
    }

    const float* forward_prepared_output_f32() override {
        if (prepared_input_.empty() || prepared_M_ <= 0 ||
            !run_accumulate(prepared_input_.data(), prepared_M_)) {
            return nullptr;
        }
        return accum_.data();
    }

    bool forward_prepared_accumulate(float* accum_f32) override {
        if (!accum_f32 || prepared_input_.empty() || prepared_M_ <= 0 ||
            !run_accumulate(prepared_input_.data(), prepared_M_)) {
            return false;
        }
        add_f32_inplace(accum_f32, accum_.data(), prepared_M_ * N_);
        return true;
    }

    bool supports_batch(int M) const override {
        if (M <= 0) {
            return false;
        }
        for (const auto& chunk : chunks_) {
            if (!chunk || !chunk->supports_batch(M)) {
                return false;
            }
        }
        return true;
    }

    void destroy() override {
        for (auto& chunk : chunks_) {
            if (chunk) chunk->destroy();
        }
        chunks_.clear();
        offsets_.clear();
        sizes_.clear();
        std::vector<uint16_t>().swap(prepared_input_);
        std::vector<uint16_t>().swap(prepared_output_);
        std::vector<uint16_t>().swap(chunk_input_);
        std::vector<float>().swap(accum_);
        prepared_M_ = 0;
        K_ = 0;
        N_ = 0;
    }

private:
    bool run_accumulate(const uint16_t* input_f16, int M) {
        if (!input_f16 || M <= 0 || K_ <= 0 || N_ <= 0 || chunks_.empty()) {
            return false;
        }
        accum_.assign((size_t)M * N_, 0.0f);

        for (size_t ci = 0; ci < chunks_.size(); ++ci) {
            const int offset = offsets_[ci];
            const int ck = sizes_[ci];
            uint16_t* chunk_input = chunks_[ci]->prepare_input_f16(M);
            if (!chunk_input) {
                return false;
            }
            for (int m = 0; m < M; ++m) {
                std::memcpy(chunk_input + (size_t)m * ck,
                            input_f16 + (size_t)m * K_ + offset,
                            (size_t)ck * sizeof(uint16_t));
            }
            const float* out = chunks_[ci]->forward_prepared_output_f32();
            if (!out) {
                return false;
            }
            add_f32_inplace(accum_.data(), out, M * N_);
        }
        return true;
    }

    int K_ = 0;
    int N_ = 0;
    int prepared_M_ = 0;
    int chunk_k_ = 4096;
    rknn_core_mask core_mask_ = RKNN_NPU_CORE_AUTO;
    std::string cache_key_;
    std::vector<std::unique_ptr<NpuLinear>> chunks_;
    std::vector<int> offsets_;
    std::vector<int> sizes_;
    std::vector<uint16_t> prepared_input_;
    std::vector<uint16_t> prepared_output_;
    std::vector<uint16_t> chunk_input_;
    std::vector<float> accum_;
};

class ShardedNpuWorkerPool {
public:
    static constexpr int kNumWorkers = 3;

    struct Task {
        ILinearOp* shard = nullptr;
        const uint16_t* input = nullptr;
        int M = 0;
        uint16_t* output = nullptr;
        const uint16_t** prepared_output = nullptr;
        const float** prepared_output_f32 = nullptr;
        bool argmax = false;
        bool prepared = false;
    };

    ShardedNpuWorkerPool() {
        for (int i = 0; i < kNumWorkers; ++i) {
            workers_[i] = std::thread([this, i]() { worker_loop(i); });
        }
    }

    ~ShardedNpuWorkerPool() {
        {
            std::lock_guard<std::mutex> lock(mu_);
            stop_ = true;
            ++generation_;
        }
        cv_task_.notify_all();
        for (auto& worker : workers_) {
            if (worker.joinable()) worker.join();
        }
    }

    ShardedNpuWorkerPool(const ShardedNpuWorkerPool&) = delete;
    ShardedNpuWorkerPool& operator=(const ShardedNpuWorkerPool&) = delete;

    bool run(const std::array<Task, kNumWorkers>& tasks) {
        std::unique_lock<std::mutex> run_lock(run_mu_);
        {
            std::lock_guard<std::mutex> lock(mu_);
            tasks_ = tasks;
            ok_.fill(false);
            done_count_ = 0;
            ++generation_;
        }
        cv_task_.notify_all();

        std::unique_lock<std::mutex> lock(mu_);
        cv_done_.wait(lock, [this]() { return done_count_ == kNumWorkers; });

        for (bool ok : ok_) {
            if (!ok) return false;
        }
        return true;
    }

    bool run_argmax(const std::array<Task, kNumWorkers>& tasks,
                    std::array<int, kNumWorkers>* argmax_ids,
                    std::array<uint16_t, kNumWorkers>* argmax_values) {
        std::unique_lock<std::mutex> run_lock(run_mu_);
        {
            std::lock_guard<std::mutex> lock(mu_);
            tasks_ = tasks;
            ok_.fill(false);
            argmax_ids_.fill(0);
            argmax_values_.fill(0);
            done_count_ = 0;
            ++generation_;
        }
        cv_task_.notify_all();

        std::unique_lock<std::mutex> lock(mu_);
        cv_done_.wait(lock, [this]() { return done_count_ == kNumWorkers; });

        for (bool ok : ok_) {
            if (!ok) return false;
        }
        *argmax_ids = argmax_ids_;
        *argmax_values = argmax_values_;
        return true;
    }

private:
    void worker_loop(int worker_id) {
        uint64_t seen_generation = 0;
        while (true) {
            Task task;
            {
                std::unique_lock<std::mutex> lock(mu_);
                cv_task_.wait(lock, [this, seen_generation]() {
                    return stop_ || generation_ != seen_generation;
                });
                if (stop_) return;
                seen_generation = generation_;
                task = tasks_[worker_id];
            }

            int local_id = 0;
            uint16_t local_value = 0;
            bool ok = false;
            if (task.shard) {
                if (task.argmax) {
                    if (task.prepared) {
                        ok = task.shard->forward_prepared_argmax(&local_id, &local_value);
                    } else {
                        ok = task.shard->forward_argmax(task.input, task.M,
                                                        &local_id, &local_value);
                    }
                } else if (task.prepared) {
                    if (task.prepared_output_f32) {
                        *task.prepared_output_f32 = task.shard->forward_prepared_output_f32();
                        ok = (*task.prepared_output_f32 != nullptr);
                    } else if (task.prepared_output) {
                        *task.prepared_output = task.shard->forward_prepared_output_f16();
                        ok = (*task.prepared_output != nullptr);
                    } else {
                        ok = task.shard->forward_prepared(task.output);
                    }
                } else {
                    ok = task.shard->forward(task.input, task.M, task.output);
                }
            }

            {
                std::lock_guard<std::mutex> lock(mu_);
                ok_[worker_id] = ok;
                if (task.argmax && ok) {
                    argmax_ids_[worker_id] = local_id;
                    argmax_values_[worker_id] = local_value;
                }
                ++done_count_;
            }
            cv_done_.notify_one();
        }
    }

    std::array<std::thread, kNumWorkers> workers_;
    std::array<Task, kNumWorkers> tasks_{};
    std::array<bool, kNumWorkers> ok_{};
    std::array<int, kNumWorkers> argmax_ids_{};
    std::array<uint16_t, kNumWorkers> argmax_values_{};

    std::mutex run_mu_;
    std::mutex mu_;
    std::condition_variable cv_task_;
    std::condition_variable cv_done_;
    uint64_t generation_ = 0;
    int done_count_ = 0;
    bool stop_ = false;
};

ShardedNpuWorkerPool& worker_pool() {
    static ShardedNpuWorkerPool pool;
    return pool;
}

std::unique_ptr<ILinearOp> make_shard_linear(bool use_a4w4,
                                             bool use_a8w8,
                                             rknn_core_mask core_mask,
                                             bool use_k_split,
                                             int k_split_chunk_k) {
    if (use_k_split && !use_a4w4 && !use_a8w8) {
        return std::unique_ptr<ILinearOp>(new KSplitNpuLinear(k_split_chunk_k, core_mask));
    }
    if (use_a4w4) {
        std::unique_ptr<NpuLinearI4> i4(new NpuLinearI4());
        i4->set_core_mask(core_mask);
        return std::move(i4);
    }
    if (use_a8w8) {
        std::unique_ptr<NpuLinearW8> w8(new NpuLinearW8());
        w8->set_core_mask(core_mask);
        return std::move(w8);
    }

    std::unique_ptr<NpuLinear> fp16(new NpuLinear());
    fp16->set_core_mask(core_mask);
    return std::move(fp16);
}

}  // namespace

void ShardedNpuLinear::set_hybrid_cpu_shard(bool enabled, int percent) {
    hybrid_cpu_shard_enabled_ = enabled;
    hybrid_cpu_percent_ = std::max(1, std::min(percent, 25));
}

void ShardedNpuLinear::set_k_split_accumulate(bool enabled, int chunk_k) {
    k_split_accumulate_enabled_ = enabled;
    k_split_chunk_k_ = std::max(32, chunk_k);
}

bool ShardedNpuLinear::init(int K, int N, const uint16_t* weight_kn) {
    destroy();

    if (!weight_kn) {
        std::fprintf(stderr, "[ShardedNpuLinear] invalid init args K=%d N=%d\n", K, N);
        return false;
    }
    if (k_shard_accumulate_enabled_ && !allow_a4w4_ && !allow_a8w8_ &&
        !gate_up_pair_layout_ && !hybrid_cpu_shard_enabled_) {
        return init_k_sharded(K, N, weight_kn);
    }
    if (!configure_layout(K, N)) {
        destroy();
        return false;
    }
    const bool use_a4w4 = allow_a4w4_;
    const bool use_a8w8 = allow_a8w8_ && !use_a4w4;
    const bool use_k_split = k_split_accumulate_enabled_ && !use_a4w4 && !use_a8w8;
    k_split_accumulate_active_ = use_k_split;

    const rknn_core_mask core_masks[kNumNpuShards] = {
        RKNN_NPU_CORE_0,
        RKNN_NPU_CORE_1,
        RKNN_NPU_CORE_2,
    };

    std::fprintf(stderr,
                 "[ShardedNpuLinear] init K=%d N=%d shards:",
                 K_, N_);
    for (int i = 0; i < shard_count_; ++i) {
        std::fprintf(stderr, "%s%d", i == 0 ? " " : "/", sizes_[i]);
    }
    std::fprintf(stderr, "%s%s%s\n",
                 use_a4w4 ? " a4w4" :
                 use_a8w8 ? " a8w8" :
                 use_k_split ? " ksplit" : "",
                 gate_up_pair_layout_ ? " gate_up_pair" : "",
                 cpu_shard_index_ >= 0 ? " hybrid_cpu" : "");

    for (int i = 0; i < shard_count_; ++i) {
        const int shard_n = sizes_[i];
        const int shard_offset = offsets_[i];
        std::vector<uint16_t> shard_weight((size_t)K_ * shard_n);

        for (int k = 0; k < K_; ++k) {
            uint16_t* dst = shard_weight.data() + (size_t)k * shard_n;
            if (gate_up_pair_layout_) {
                // gate_up_pair 的每个 shard 同时保存一段 gate 和对应一段 up：
                // shard_i B = [gate[offset:size], up[offset:size]]。
                // 这样推理时 SiLU(gate)*up 可以直接消费 shard 输出，
                // 不必先拼回完整 [gate_all, up_all]。
                const int inner = N_ / 2;
                const int seg_offset = pair_offsets_[i];
                const int seg_size = pair_sizes_[i];
                const uint16_t* gate = weight_kn + (size_t)k * N_ + seg_offset;
                const uint16_t* up = weight_kn + (size_t)k * N_ + inner + seg_offset;
                std::memcpy(dst, gate, (size_t)seg_size * sizeof(uint16_t));
                std::memcpy(dst + seg_size, up, (size_t)seg_size * sizeof(uint16_t));
            } else {
                const uint16_t* src = weight_kn + (size_t)k * N_ + shard_offset;
                std::memcpy(dst, src, (size_t)shard_n * sizeof(uint16_t));
            }
        }

        std::unique_ptr<ILinearOp> shard;
        if (i == cpu_shard_index_) {
            std::unique_ptr<CpuLinear> cpu(new CpuLinear());
            cpu->set_fast_f32_weight(true);
            shard = std::move(cpu);
        } else {
            shard = make_shard_linear(use_a4w4, use_a8w8, core_masks[i],
                                      use_k_split, k_split_chunk_k_);
        }
        if (!cache_key_.empty()) {
            // 三核分片缓存按 shard 独立保存。warm-load 时只需要按同样分片
            // 读回每个 shard 的 native B，不需要构造完整大权重。
            std::string shard_key = cache_key_;
            if (gate_up_pair_layout_) {
                shard_key += ".gate_up_pair";
            }
            if (cpu_shard_index_ >= 0) {
                shard_key += ".hybrid_cpu";
            }
            shard_key += ".shard" + std::to_string(i);
            shard->set_cache_key(shard_key);
        }
        bool shard_ok = shard->init(K_, shard_n, shard_weight.data());
        if (!shard_ok && i != cpu_shard_index_) {
            if (use_a4w4 || use_a8w8) {
                std::fprintf(stderr,
                             "[ShardedNpuLinear] %s shard %d init failed, fallback FP16\n",
                             use_a4w4 ? "I4" : "W8", i);
                shard->destroy();
                shard = make_shard_linear(false, false, core_masks[i],
                                          use_k_split, k_split_chunk_k_);
                if (!cache_key_.empty()) {
                    std::string shard_key = cache_key_;
                    if (gate_up_pair_layout_) {
                        shard_key += ".gate_up_pair";
                    }
                    if (cpu_shard_index_ >= 0) {
                        shard_key += ".hybrid_cpu";
                    }
                    shard_key += ".shard" + std::to_string(i);
                    shard->set_cache_key(shard_key);
                }
                shard_ok = shard->init(K_, shard_n, shard_weight.data());
            }
        }
        if (!shard_ok) {
            std::fprintf(stderr, "[ShardedNpuLinear] shard %d init failed\n", i);
            destroy();
            return false;
        }
        shards_[i] = std::move(shard);
    }

    return true;
}

bool ShardedNpuLinear::init_from_cache(int K, int N) {
    if (cache_key_.empty()) {
        return false;
    }

    destroy();
    if (k_shard_accumulate_enabled_ && !allow_a4w4_ && !allow_a8w8_ &&
        !gate_up_pair_layout_ && !hybrid_cpu_shard_enabled_) {
        return init_k_sharded_from_cache(K, N);
    }
    if (!configure_layout(K, N)) {
        destroy();
        return false;
    }
    if (cpu_shard_index_ >= 0) {
        destroy();
        return false;
    }
    const bool use_a4w4 = allow_a4w4_;
    const bool use_a8w8 = allow_a8w8_ && !use_a4w4;
    const bool use_k_split = k_split_accumulate_enabled_ && !use_a4w4 && !use_a8w8;
    k_split_accumulate_active_ = use_k_split;

    const rknn_core_mask core_masks[kNumNpuShards] = {
        RKNN_NPU_CORE_0,
        RKNN_NPU_CORE_1,
        RKNN_NPU_CORE_2,
    };

    for (int i = 0; i < shard_count_; ++i) {
        std::unique_ptr<ILinearOp> shard =
            make_shard_linear(use_a4w4, use_a8w8, core_masks[i],
                              use_k_split, k_split_chunk_k_);

        std::string shard_key = cache_key_;
        if (gate_up_pair_layout_) {
            shard_key += ".gate_up_pair";
        }
        shard_key += ".shard" + std::to_string(i);
        shard->set_cache_key(shard_key);

        // 每个 shard 独立 cache hit 才算整个 sharded linear 命中。
        // 任意 shard miss 都销毁并让模型层回到冷加载路径。
        if (!shard->init_from_cache(K_, sizes_[i])) {
            destroy();
            return false;
        }
        shards_[i] = std::move(shard);
    }

    std::fprintf(stderr,
                 "[ShardedNpuLinear] init K=%d N=%d shards: %d/%d/%d%s%s cache\n",
                 K_, N_, sizes_[0], sizes_[1], sizes_[2],
                 use_a4w4 ? " a4w4" :
                 use_a8w8 ? " a8w8" :
                 use_k_split ? " ksplit" : "",
                 gate_up_pair_layout_ ? " gate_up_pair" : "");
    return true;
}

bool ShardedNpuLinear::configure_layout(int K, int N) {
    if (K <= 0 || N <= 0) {
        std::fprintf(stderr, "[ShardedNpuLinear] invalid init args K=%d N=%d\n", K, N);
        return false;
    }

    const bool use_a4w4 = allow_a4w4_;
    const bool use_a8w8 = allow_a8w8_ && !use_a4w4;
    const int backend_n_align = use_a4w4 ? 64 : (use_a8w8 ? 32 : kNAlign);
    if (N % backend_n_align != 0) {
        std::fprintf(stderr, "[ShardedNpuLinear] N=%d is not aligned to %d\n",
                     N, backend_n_align);
        return false;
    }

    // Start the process-wide workers during load so the first profiled forward
    // does not include thread creation.
    worker_pool();

    K_ = K;
    N_ = N;
    shard_count_ = hybrid_cpu_shard_enabled_ ? kMaxShards : kNumNpuShards;
    cpu_shard_index_ = hybrid_cpu_shard_enabled_ ? (shard_count_ - 1) : -1;
    offsets_.assign((size_t)shard_count_, 0);
    sizes_.assign((size_t)shard_count_, 0);
    pair_offsets_.clear();
    pair_sizes_.clear();
    if (gate_up_pair_layout_) {
        if ((N_ % 2) != 0) {
            std::fprintf(stderr, "[ShardedNpuLinear] gate_up pair layout needs even N=%d\n", N_);
            return false;
        }
        pair_offsets_.assign((size_t)shard_count_, 0);
        pair_sizes_.assign((size_t)shard_count_, 0);
    }
    shards_.resize((size_t)shard_count_);

    // 普通 sharding 按 N 连续切三份。gate_up_pair 先把逻辑 N 看成
    // gate/up 两半，只对 intermediate 维切三份，随后每份扩成 gate+up。
    const int split_N = gate_up_pair_layout_ ? N_ / 2 : N_;
    const int split_align = gate_up_pair_layout_
        ? std::max(1, backend_n_align / 2)
        : backend_n_align;
    int cpu_split = 0;
    if (cpu_shard_index_ >= 0) {
        cpu_split = align_up((split_N * hybrid_cpu_percent_ + 99) / 100,
                             split_align);
        if (cpu_split >= split_N) {
            cpu_split = 0;
        }
        // Keep enough aligned work for the three NPU shards.
        const int min_npu_total = split_align * kNumNpuShards;
        if (cpu_split > 0 && split_N - cpu_split < min_npu_total) {
            cpu_split = 0;
        }
    }
    int remaining = split_N;
    int offset = 0;
    for (int i = 0; i < shard_count_; ++i) {
        offsets_[i] = offset;

        int size = 0;
        if (i == cpu_shard_index_) {
            size = remaining;
        } else if (cpu_shard_index_ >= 0 && i == kNumNpuShards - 1) {
            size = remaining - cpu_split;
        } else if (i == shard_count_ - 1) {
            size = remaining;
        } else {
            int npu_end = cpu_shard_index_ >= 0 ? kNumNpuShards : shard_count_;
            int shards_left = npu_end - i;
            int target_remaining = remaining - (cpu_shard_index_ >= 0 ? cpu_split : 0);
            size = align_up(remaining / shards_left, split_align);
            if (cpu_shard_index_ >= 0) {
                size = align_up(target_remaining / shards_left, split_align);
            }
        }

        if (size <= 0 || size % split_align != 0 || size > remaining) {
            std::fprintf(stderr,
                         "[ShardedNpuLinear] bad shard split i=%d size=%d remaining=%d\n",
                         i, size, remaining);
            return false;
        }

        if (gate_up_pair_layout_) {
            pair_offsets_[i] = offset;
            pair_sizes_[i] = size;
            sizes_[i] = size * 2;
        } else {
            sizes_[i] = size;
        }
        remaining -= size;
        offset += size;
    }
    if (remaining != 0) {
        std::fprintf(stderr, "[ShardedNpuLinear] split left remaining=%d\n", remaining);
        return false;
    }
    return true;
}

bool ShardedNpuLinear::init_k_sharded(int K, int N, const uint16_t* weight_kn) {
    constexpr int kKShardAlign = 32;
    if (K <= 0 || N <= 0 || !weight_kn ||
        (K % kKShardAlign) != 0 || (N % kNAlign) != 0) {
        std::fprintf(stderr, "[ShardedNpuLinear] invalid K-shard init K=%d N=%d\n", K, N);
        return false;
    }

    K_ = K;
    N_ = N;
    shard_count_ = kNumNpuShards;
    k_shard_accumulate_active_ = true;
    k_offsets_.assign(kNumNpuShards, 0);
    k_sizes_.assign(kNumNpuShards, 0);
    offsets_.assign(kNumNpuShards, 0);
    sizes_.assign(kNumNpuShards, N_);
    shards_.resize(kNumNpuShards);
    worker_pool();

    const rknn_core_mask core_masks[kNumNpuShards] = {
        RKNN_NPU_CORE_0,
        RKNN_NPU_CORE_1,
        RKNN_NPU_CORE_2,
    };

    int offset = 0;
    for (int i = 0; i < kNumNpuShards; ++i) {
        k_offsets_[i] = offset;
        int shards_left = kNumNpuShards - i;
        int remaining = K_ - offset;
        int k_part = (i == kNumNpuShards - 1)
            ? remaining
            : align_up(remaining / shards_left, kKShardAlign);
        if (k_part <= 0 || k_part > remaining || (k_part % kKShardAlign) != 0) {
            std::fprintf(stderr,
                         "[ShardedNpuLinear] bad K-shard split i=%d Kpart=%d remaining=%d\n",
                         i, k_part, remaining);
            destroy();
            return false;
        }
        k_sizes_[i] = k_part;

        std::unique_ptr<NpuLinear> shard(new NpuLinear());
        shard->set_core_mask(core_masks[i]);
        shard->set_output_f32(true);
        if (!cache_key_.empty()) {
            shard->set_cache_key(cache_key_ + ".kshard" + std::to_string(i));
        }
        if (!shard->init(k_part, N_, weight_kn + (size_t)offset * N_)) {
            std::fprintf(stderr,
                         "[ShardedNpuLinear] K-shard %d init failed K=%d N=%d\n",
                         i, k_part, N_);
            destroy();
            return false;
        }
        shards_[i] = std::move(shard);
        offset += k_part;
    }

    std::fprintf(stderr, "[ShardedNpuLinear] init K=%d N=%d K-shards: %d/%d/%d f32_accumulate\n",
                 K_, N_, k_sizes_[0], k_sizes_[1], k_sizes_[2]);
    return true;
}

bool ShardedNpuLinear::init_k_sharded_from_cache(int K, int N) {
    constexpr int kKShardAlign = 32;
    if (K <= 0 || N <= 0 || cache_key_.empty() ||
        (K % kKShardAlign) != 0 || (N % kNAlign) != 0) {
        return false;
    }

    K_ = K;
    N_ = N;
    shard_count_ = kNumNpuShards;
    k_shard_accumulate_active_ = true;
    k_offsets_.assign(kNumNpuShards, 0);
    k_sizes_.assign(kNumNpuShards, 0);
    offsets_.assign(kNumNpuShards, 0);
    sizes_.assign(kNumNpuShards, N_);
    shards_.resize(kNumNpuShards);
    worker_pool();

    const rknn_core_mask core_masks[kNumNpuShards] = {
        RKNN_NPU_CORE_0,
        RKNN_NPU_CORE_1,
        RKNN_NPU_CORE_2,
    };

    int offset = 0;
    for (int i = 0; i < kNumNpuShards; ++i) {
        k_offsets_[i] = offset;
        int shards_left = kNumNpuShards - i;
        int remaining = K_ - offset;
        int k_part = (i == kNumNpuShards - 1)
            ? remaining
            : align_up(remaining / shards_left, kKShardAlign);
        if (k_part <= 0 || k_part > remaining || (k_part % kKShardAlign) != 0) {
            destroy();
            return false;
        }
        k_sizes_[i] = k_part;

        std::unique_ptr<NpuLinear> shard(new NpuLinear());
        shard->set_core_mask(core_masks[i]);
        shard->set_output_f32(true);
        shard->set_cache_key(cache_key_ + ".kshard" + std::to_string(i));
        if (!shard->init_from_cache(k_part, N_)) {
            destroy();
            return false;
        }
        shards_[i] = std::move(shard);
        offset += k_part;
    }

    std::fprintf(stderr,
                 "[ShardedNpuLinear] init K=%d N=%d K-shards: %d/%d/%d f32_accumulate cache\n",
                 K_, N_, k_sizes_[0], k_sizes_[1], k_sizes_[2]);
    return true;
}

bool ShardedNpuLinear::prepare_k_shard_inputs() {
    if (!prepared_input_ || prepared_M_ <= 0 || !k_shard_accumulate_active_) {
        return false;
    }
    const bool profile = mlp_profile_enabled_for_key(cache_key_);
    const int64_t t0 = profile ? now_us() : 0;
    int64_t prepare_total_us = 0;
    int64_t copy_total_us = 0;
    for (int i = 0; i < kNumNpuShards; ++i) {
        const int64_t t_prepare0 = profile ? now_us() : 0;
        uint16_t* dst = shards_[i] ? shards_[i]->prepare_input_f16(prepared_M_) : nullptr;
        const int64_t t_copy0 = profile ? now_us() : 0;
        if (!dst) {
            return false;
        }
        const int k_offset = k_offsets_[i];
        const int k_size = k_sizes_[i];
        for (int m = 0; m < prepared_M_; ++m) {
            std::memcpy(dst + (size_t)m * k_size,
                        prepared_input_ + (size_t)m * K_ + k_offset,
                        (size_t)k_size * sizeof(uint16_t));
        }
        if (profile) {
            const int64_t t1 = now_us();
            prepare_total_us += t_copy0 - t_prepare0;
            copy_total_us += t1 - t_copy0;
        }
    }
    if (profile) {
        std::fprintf(stderr,
                     "[mlp_profile] down_kshard M=%d K=%d N=%d prepare_inputs total=%.3f ms prepare_ac=%.3f ms copy_slices=%.3f ms Kparts=%d/%d/%d\n",
                     prepared_M_, K_, N_, us_to_ms(now_us() - t0),
                     us_to_ms(prepare_total_us), us_to_ms(copy_total_us),
                     k_sizes_[0], k_sizes_[1], k_sizes_[2]);
    }
    return true;
}

bool ShardedNpuLinear::run_k_shard_accumulate(float* accum_f32) {
    const bool profile = mlp_profile_enabled_for_key(cache_key_);
    const int64_t t0 = profile ? now_us() : 0;
    if (!accum_f32 || !prepare_k_shard_inputs()) {
        clear_prepared_state();
        return false;
    }
    const int64_t t_prepared = profile ? now_us() : 0;

    const int M = prepared_M_;
    last_shard_outputs_f32_.fill(nullptr);
    std::array<ShardedNpuWorkerPool::Task, kNumNpuShards> tasks{};
    for (int i = 0; i < kNumNpuShards; ++i) {
        tasks[i].shard = shards_[i].get();
        tasks[i].M = prepared_M_;
        tasks[i].prepared = true;
        tasks[i].prepared_output_f32 = &last_shard_outputs_f32_[i];
    }
    const int64_t t_run0 = profile ? now_us() : 0;
    const bool ok = worker_pool().run(tasks);
    const int64_t t_run1 = profile ? now_us() : 0;
    clear_prepared_state();
    if (!ok) {
        std::fprintf(stderr, "[ShardedNpuLinear] K-shard prepared forward failed\n");
        return false;
    }

    const int64_t t_zero0 = profile ? now_us() : 0;
    k_shard_accum_.assign((size_t)M * N_, 0.0f);
    const int64_t t_sum0 = profile ? now_us() : 0;
    for (int i = 0; i < kNumNpuShards; ++i) {
        const float* out = last_shard_outputs_f32_[i];
        if (!out) {
            return false;
        }
        add_f32_inplace(k_shard_accum_.data(), out, M * N_);
    }
    const int64_t t_add0 = profile ? now_us() : 0;
    add_f32_inplace(accum_f32, k_shard_accum_.data(), M * N_);
    if (profile) {
        const int64_t t1 = now_us();
        std::fprintf(stderr,
                     "[mlp_profile] down_kshard M=%d K=%d N=%d total=%.3f ms prepare=%.3f ms dispatch_run=%.3f ms zero=%.3f ms sum_partials=%.3f ms add_residual=%.3f ms\n",
                     M, K_, N_, us_to_ms(t1 - t0),
                     us_to_ms(t_prepared - t0),
                     us_to_ms(t_run1 - t_run0),
                     us_to_ms(t_sum0 - t_zero0),
                     us_to_ms(t_add0 - t_sum0),
                     us_to_ms(t1 - t_add0));
    }
    return true;
}

bool ShardedNpuLinear::forward(const uint16_t* input_f16, int M, uint16_t* output_f16) {
    if (!input_f16 || !output_f16 || K_ <= 0 || N_ <= 0 || M <= 0 ||
        (int)shards_.size() != shard_count_) {
        std::fprintf(stderr,
                     "[ShardedNpuLinear] invalid forward args M=%d K=%d N=%d\n",
                     M, K_, N_);
        return false;
    }

    if (k_shard_accumulate_active_) {
        prepared_k_shard_input_.assign(input_f16, input_f16 + (size_t)M * K_);
        prepared_input_ = prepared_k_shard_input_.data();
        prepared_M_ = M;
        k_shard_accum_.assign((size_t)M * N_, 0.0f);
        if (!run_k_shard_accumulate(k_shard_accum_.data())) {
            return false;
        }
        f32_to_f16_buffer(k_shard_accum_.data(), output_f16, M * N_);
        return true;
    }

    std::array<ShardedNpuWorkerPool::Task, kNumNpuShards> tasks{};
    const bool can_write_direct = (M == 1 && !gate_up_pair_layout_ && cpu_shard_index_ < 0);
    for (int i = 0; i < kNumNpuShards; ++i) {
        if (can_write_direct) {
            tasks[i].shard = shards_[i].get();
            tasks[i].input = input_f16;
            tasks[i].M = M;
            tasks[i].output = output_f16 + offsets_[i];
        } else {
            shard_outputs_[i].resize((size_t)M * sizes_[i]);
            tasks[i].shard = shards_[i].get();
            tasks[i].input = input_f16;
            tasks[i].M = M;
            tasks[i].output = shard_outputs_[i].data();
        }
    }

    std::future<bool> cpu_future;
    if (cpu_shard_index_ >= 0) {
        const int ci = cpu_shard_index_;
        shard_outputs_[ci].resize((size_t)M * sizes_[ci]);
        cpu_future = std::async(std::launch::async, [this, input_f16, M, ci]() {
            return shards_[ci] &&
                   shards_[ci]->forward(input_f16, M, shard_outputs_[ci].data());
        });
    }

    if (!worker_pool().run(tasks)) {
        std::fprintf(stderr, "[ShardedNpuLinear] shard forward failed\n");
        return false;
    }
    if (cpu_shard_index_ >= 0 && !cpu_future.get()) {
        std::fprintf(stderr, "[ShardedNpuLinear] CPU hybrid shard forward failed\n");
        return false;
    }

    if (!can_write_direct) {
        copy_shard_outputs_to(output_f16, M);
    }

    return true;
}

void ShardedNpuLinear::copy_shard_outputs_to(uint16_t* output_f16, int M) const {
    if (!output_f16 || M <= 0) {
        return;
    }
    for (int m = 0; m < M; ++m) {
        uint16_t* out_row = output_f16 + (size_t)m * N_;
        for (int i = 0; i < shard_count_; ++i) {
            const uint16_t* src = shard_outputs_[i].data() + (size_t)m * sizes_[i];
            if (gate_up_pair_layout_) {
                const int inner = N_ / 2;
                const int seg_offset = pair_offsets_[i];
                const int seg_size = pair_sizes_[i];
                std::memcpy(out_row + seg_offset,
                            src,
                            (size_t)seg_size * sizeof(uint16_t));
                std::memcpy(out_row + inner + seg_offset,
                            src + seg_size,
                            (size_t)seg_size * sizeof(uint16_t));
            } else {
                uint16_t* dst = out_row + offsets_[i];
                std::memcpy(dst, src, (size_t)sizes_[i] * sizeof(uint16_t));
            }
        }
    }
}

uint16_t* ShardedNpuLinear::prepare_input_f16(int M) {
    prepared_input_ = nullptr;
    prepared_M_ = 0;
    if (M <= 0 || K_ <= 0 || N_ <= 0 || (int)shards_.size() != shard_count_) {
        return nullptr;
    }
    if (k_shard_accumulate_active_) {
        prepared_k_shard_input_.resize((size_t)M * K_);
        prepared_input_ = prepared_k_shard_input_.data();
        prepared_M_ = M;
        return prepared_input_;
    }
    for (const auto& shard : shards_) {
        if (!shard || !shard->supports_batch(M)) {
            return nullptr;
        }
    }

    uint16_t* input = shards_[0] ? shards_[0]->prepare_input_f16(M) : nullptr;
    if (!input) {
        return nullptr;
    }

    prepared_input_ = input;
    prepared_M_ = M;
    return prepared_input_;
}

bool ShardedNpuLinear::bind_shared_prepared_inputs() {
    NpuLinear* owner = dynamic_cast<NpuLinear*>(shards_[0].get());
    const rknn_tensor_mem* shared_input = owner ? owner->prepared_input_mem() : nullptr;
    if (!shared_input) {
        return false;
    }

    if (cpu_shard_index_ >= 0) {
        return false;
    }
    for (int i = 1; i < kNumNpuShards; ++i) {
        NpuLinear* shard = dynamic_cast<NpuLinear*>(shards_[i].get());
        if (!shard || !shard->bind_external_input_f16(prepared_M_, shared_input)) {
            return false;
        }
    }

    static bool logged = false;
    if (!logged) {
        logged = true;
        std::fprintf(stderr, "[ShardedNpuLinear] shared prepared A buffer enabled\n");
    }
    return true;
}

void ShardedNpuLinear::clear_prepared_state() {
    prepared_input_ = nullptr;
    prepared_M_ = 0;
}

bool ShardedNpuLinear::run_prepared_output_shards() {
    if (!prepared_input_ || prepared_M_ <= 0 ||
        K_ <= 0 || N_ <= 0 || (int)shards_.size() != shard_count_) {
        return false;
    }

    const bool profile = mlp_profile_enabled_for_key(cache_key_);
    const int M = prepared_M_;
    const int64_t t0 = profile ? now_us() : 0;
    const bool shared = bind_shared_prepared_inputs();
    const int64_t t_bind = profile ? now_us() : 0;
    int copied_inputs = 0;
    int64_t copy_inputs_us = 0;
    if (!shared) {
        for (int i = 1; i < shard_count_; ++i) {
            uint16_t* shard_input = shards_[i] ? shards_[i]->prepare_input_f16(prepared_M_) : nullptr;
            if (!shard_input) {
                clear_prepared_state();
                return false;
            }
            if (shard_input != prepared_input_) {
                const int64_t t_copy0 = profile ? now_us() : 0;
                std::memcpy(shard_input, prepared_input_,
                            (size_t)prepared_M_ * K_ * sizeof(uint16_t));
                if (profile) {
                    copy_inputs_us += now_us() - t_copy0;
                    ++copied_inputs;
                }
            }
        }
    }
    const int64_t t_inputs = profile ? now_us() : 0;

    last_shard_outputs_.fill(nullptr);
    std::array<ShardedNpuWorkerPool::Task, kNumNpuShards> tasks{};
    for (int i = 0; i < kNumNpuShards; ++i) {
        tasks[i].shard = shards_[i].get();
        tasks[i].M = prepared_M_;
        tasks[i].prepared = true;
        tasks[i].prepared_output = &last_shard_outputs_[i];
    }
    std::future<bool> cpu_future;
    if (cpu_shard_index_ >= 0) {
        const int ci = cpu_shard_index_;
        cpu_future = std::async(std::launch::async, [this, ci]() {
            last_shard_outputs_[ci] = shards_[ci]
                ? shards_[ci]->forward_prepared_output_f16()
                : nullptr;
            return last_shard_outputs_[ci] != nullptr;
        });
    }

    const int64_t t_run0 = profile ? now_us() : 0;
    const bool ok = worker_pool().run(tasks);
    const int64_t t_run1 = profile ? now_us() : 0;
    clear_prepared_state();
    if (!ok) {
        std::fprintf(stderr, "[ShardedNpuLinear] prepared shard forward failed\n");
        return false;
    }
    if (cpu_shard_index_ >= 0 && !cpu_future.get()) {
        std::fprintf(stderr, "[ShardedNpuLinear] prepared CPU hybrid shard forward failed\n");
        return false;
    }
    for (int i = 0; i < shard_count_; ++i) {
        if (!last_shard_outputs_[i]) {
            return false;
        }
    }
    if (profile) {
        std::fprintf(stderr,
                     "[mlp_profile] sharded_prepared M=%d K=%d N=%d total=%.3f ms bind=%.3f ms input_copy=%.3f ms input_setup=%.3f ms run=%.3f ms shared=%d copied=%d gate_up_pair=%d\n",
                     M, K_, N_, us_to_ms(now_us() - t0),
                     us_to_ms(t_bind - t0),
                     us_to_ms(copy_inputs_us),
                     us_to_ms(t_inputs - t_bind),
                     us_to_ms(t_run1 - t_run0),
                     shared ? 1 : 0,
                     copied_inputs,
                     gate_up_pair_layout_ ? 1 : 0);
    }
    return true;
}

bool ShardedNpuLinear::forward_prepared_output_shards_f16(const uint16_t** outputs,
                                                          int* offsets,
                                                          int* sizes,
                                                          int max_shards,
                                                          int* num_shards) {
    if (!outputs || !offsets || !sizes || !num_shards || max_shards < shard_count_) {
        return false;
    }
    if (!run_prepared_output_shards()) {
        return false;
    }

    *num_shards = shard_count_;
    for (int i = 0; i < shard_count_; ++i) {
        outputs[i] = last_shard_outputs_[i];
        if (gate_up_pair_layout_) {
            offsets[i] = pair_offsets_[i];
            sizes[i] = pair_sizes_[i];
        } else {
            offsets[i] = offsets_[i];
            sizes[i] = sizes_[i];
        }
    }
    return true;
}

bool ShardedNpuLinear::forward_prepared_accumulate(float* accum_f32) {
    if (!accum_f32 || !prepared_input_ || prepared_M_ <= 0 ||
        K_ <= 0 || N_ <= 0 || (int)shards_.size() != shard_count_) {
        return false;
    }

    const int M = prepared_M_;
    if (k_shard_accumulate_active_) {
        return run_k_shard_accumulate(accum_f32);
    }
    if (k_split_accumulate_active_ && !gate_up_pair_layout_ && cpu_shard_index_ < 0) {
        if (!bind_shared_prepared_inputs()) {
            for (int i = 1; i < shard_count_; ++i) {
                uint16_t* shard_input =
                    shards_[i] ? shards_[i]->prepare_input_f16(prepared_M_) : nullptr;
                if (!shard_input) {
                    clear_prepared_state();
                    return false;
                }
                if (shard_input != prepared_input_) {
                    std::memcpy(shard_input, prepared_input_,
                                (size_t)prepared_M_ * K_ * sizeof(uint16_t));
                }
            }
        }

        // WorkerPool cannot call forward_prepared_accumulate directly, so use
        // async here and accumulate shard-local FP32 results after each task.
        std::array<std::future<const float*>, kNumNpuShards> futures;
        for (int i = 0; i < kNumNpuShards; ++i) {
            futures[i] = std::async(std::launch::async, [this, i]() {
                return shards_[i] ? shards_[i]->forward_prepared_output_f32() : nullptr;
            });
        }
        clear_prepared_state();
        for (int i = 0; i < kNumNpuShards; ++i) {
            const float* out = futures[i].get();
            if (!out) {
                return false;
            }
            for (int m = 0; m < M; ++m) {
                add_f32_inplace(accum_f32 + (size_t)m * N_ + offsets_[i],
                                out + (size_t)m * sizes_[i],
                                sizes_[i]);
            }
        }
        return true;
    }

    if (!run_prepared_output_shards()) {
        return false;
    }

    for (int i = 0; i < shard_count_; ++i) {
        const uint16_t* out = last_shard_outputs_[i];
        if (gate_up_pair_layout_) {
            const int inner = N_ / 2;
            const int seg_offset = pair_offsets_[i];
            const int seg_size = pair_sizes_[i];
            for (int m = 0; m < M; ++m) {
                float* dst = accum_f32 + (size_t)m * N_;
                const uint16_t* src = out + (size_t)m * sizes_[i];
                op_add_f16_to_f32_inplace(dst + seg_offset, src, seg_size);
                op_add_f16_to_f32_inplace(dst + inner + seg_offset,
                                          src + seg_size, seg_size);
            }
        } else {
            for (int m = 0; m < M; ++m) {
                op_add_f16_to_f32_inplace(accum_f32 + (size_t)m * N_ + offsets_[i],
                                          out + (size_t)m * sizes_[i],
                                          sizes_[i]);
            }
        }
    }
    return true;
}

const uint16_t* ShardedNpuLinear::forward_prepared_output_f16() {
    if (!prepared_input_ || prepared_M_ <= 0 ||
        K_ <= 0 || N_ <= 0 || (int)shards_.size() != shard_count_) {
        return nullptr;
    }

    const int M = prepared_M_;
    if (k_shard_accumulate_active_) {
        k_shard_accum_.assign((size_t)M * N_, 0.0f);
        if (!run_k_shard_accumulate(k_shard_accum_.data())) {
            return nullptr;
        }
        prepared_output_.resize((size_t)M * N_);
        f32_to_f16_buffer(k_shard_accum_.data(), prepared_output_.data(), M * N_);
        return prepared_output_.data();
    }
    if (!run_prepared_output_shards()) {
        return nullptr;
    }

    prepared_output_.resize((size_t)M * N_);
    for (int m = 0; m < M; ++m) {
        uint16_t* out_row = prepared_output_.data() + (size_t)m * N_;
        for (int i = 0; i < shard_count_; ++i) {
            const uint16_t* src = last_shard_outputs_[i] + (size_t)m * sizes_[i];
            if (gate_up_pair_layout_) {
                const int inner = N_ / 2;
                const int seg_offset = pair_offsets_[i];
                const int seg_size = pair_sizes_[i];
                std::memcpy(out_row + seg_offset,
                            src,
                            (size_t)seg_size * sizeof(uint16_t));
                std::memcpy(out_row + inner + seg_offset,
                            src + seg_size,
                            (size_t)seg_size * sizeof(uint16_t));
            } else {
                std::memcpy(out_row + offsets_[i],
                            src,
                            (size_t)sizes_[i] * sizeof(uint16_t));
            }
        }
    }
    return prepared_output_.data();
}

bool ShardedNpuLinear::forward_prepared_argmax(int* argmax_id, uint16_t* argmax_value) {
    if (!argmax_id || !prepared_input_ || prepared_M_ != 1 ||
        K_ <= 0 || N_ <= 0 || (int)shards_.size() != kNumNpuShards ||
        cpu_shard_index_ >= 0) {
        return false;
    }

    if (!bind_shared_prepared_inputs()) {
        for (int i = 1; i < kNumNpuShards; ++i) {
            uint16_t* shard_input = shards_[i] ? shards_[i]->prepare_input_f16(prepared_M_) : nullptr;
            if (!shard_input) {
                prepared_input_ = nullptr;
                prepared_M_ = 0;
                return false;
            }
            if (shard_input != prepared_input_) {
                std::memcpy(shard_input, prepared_input_,
                            (size_t)prepared_M_ * K_ * sizeof(uint16_t));
            }
        }
    }

    std::array<ShardedNpuWorkerPool::Task, kNumNpuShards> tasks{};
    for (int i = 0; i < kNumNpuShards; ++i) {
        tasks[i].shard = shards_[i].get();
        tasks[i].M = prepared_M_;
        tasks[i].argmax = true;
        tasks[i].prepared = true;
    }

    std::array<int, kNumNpuShards> local_ids{};
    std::array<uint16_t, kNumNpuShards> local_values{};
    const bool ok = worker_pool().run_argmax(tasks, &local_ids, &local_values);
    prepared_input_ = nullptr;
    prepared_M_ = 0;
    if (!ok) {
        std::fprintf(stderr, "[ShardedNpuLinear] prepared shard argmax failed\n");
        return false;
    }

    int best = offsets_[0] + local_ids[0];
    uint16_t best_value = local_values[0];
    uint16_t best_key = f16_order_key(best_value);
    for (int si = 0; si < kNumNpuShards; ++si) {
        const uint16_t key = f16_order_key(local_values[si]);
        if (key > best_key) {
            best_key = key;
            best_value = local_values[si];
            best = offsets_[si] + local_ids[si];
        }
    }

    *argmax_id = best;
    if (argmax_value) {
        *argmax_value = best_value;
    }
    return true;
}

bool ShardedNpuLinear::supports_batch(int M) const {
    if (M <= 0 || (int)shards_.size() != shard_count_) {
        return false;
    }
    for (const auto& shard : shards_) {
        if (!shard || !shard->supports_batch(M)) {
            return false;
        }
    }
    return true;
}

bool ShardedNpuLinear::forward_argmax(const uint16_t* input_f16, int M,
                                      int* argmax_id, uint16_t* argmax_value) {
    if (!argmax_id || !input_f16 || K_ <= 0 || N_ <= 0 || M != 1 ||
        (int)shards_.size() != kNumNpuShards || cpu_shard_index_ >= 0) {
        return false;
    }

    std::array<ShardedNpuWorkerPool::Task, kNumNpuShards> tasks{};
    for (int i = 0; i < kNumNpuShards; ++i) {
        tasks[i].shard = shards_[i].get();
        tasks[i].input = input_f16;
        tasks[i].M = 1;
        tasks[i].argmax = true;
    }

    std::array<int, kNumNpuShards> local_ids{};
    std::array<uint16_t, kNumNpuShards> local_values{};
    if (!worker_pool().run_argmax(tasks, &local_ids, &local_values)) {
        std::fprintf(stderr, "[ShardedNpuLinear] shard argmax forward failed\n");
        return false;
    }

    int best = offsets_[0] + local_ids[0];
    uint16_t best_value = local_values[0];
    uint16_t best_key = f16_order_key(best_value);
    for (int si = 0; si < kNumNpuShards; ++si) {
        const uint16_t key = f16_order_key(local_values[si]);
        if (key > best_key) {
            best_key = key;
            best_value = local_values[si];
            best = offsets_[si] + local_ids[si];
        }
    }

    *argmax_id = best;
    if (argmax_value) {
        *argmax_value = best_value;
    }
    return true;
}

void ShardedNpuLinear::destroy() {
    for (auto it = shards_.rbegin(); it != shards_.rend(); ++it) {
        if (*it) (*it)->destroy();
    }
    shards_.clear();
    offsets_.clear();
    sizes_.clear();
    pair_offsets_.clear();
    pair_sizes_.clear();
    for (auto& out : shard_outputs_) {
        std::vector<uint16_t>().swap(out);
    }
    std::vector<uint16_t>().swap(prepared_output_);
    last_shard_outputs_.fill(nullptr);
    last_shard_outputs_f32_.fill(nullptr);
    prepared_input_ = nullptr;
    prepared_M_ = 0;
    k_split_accumulate_active_ = false;
    k_shard_accumulate_active_ = false;
    shard_count_ = kNumNpuShards;
    cpu_shard_index_ = -1;
    K_ = 0;
    N_ = 0;
}
