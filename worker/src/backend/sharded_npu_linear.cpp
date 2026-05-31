#include "backend/sharded_npu_linear.h"
#include "backend/npu_linear_w8.h"

#include <algorithm>
#include <array>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <thread>

namespace {

int align_up(int v, int align) {
    return ((v + align - 1) / align) * align;
}

inline uint16_t f16_order_key(uint16_t v) {
    return (v & 0x8000u) ? (uint16_t)~v : (uint16_t)(v ^ 0x8000u);
}

class ShardedNpuWorkerPool {
public:
    static constexpr int kNumWorkers = 3;

    struct Task {
        ILinearOp* shard = nullptr;
        const uint16_t* input = nullptr;
        int M = 0;
        uint16_t* output = nullptr;
        bool argmax = false;
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
                    ok = task.shard->forward_argmax(task.input, task.M,
                                                    &local_id, &local_value);
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

}  // namespace

bool ShardedNpuLinear::init(int K, int N, const uint16_t* weight_kn) {
    destroy();

    if (K <= 0 || N <= 0 || !weight_kn) {
        std::fprintf(stderr, "[ShardedNpuLinear] invalid init args K=%d N=%d\n", K, N);
        return false;
    }
    const bool use_a8w8 = allow_a8w8_;
    const int n_align = use_a8w8 ? 32 : kNAlign;
    if (N % n_align != 0) {
        std::fprintf(stderr, "[ShardedNpuLinear] N=%d is not aligned to %d\n", N, n_align);
        return false;
    }

    // Start the process-wide workers during load so the first profiled forward
    // does not include thread creation.
    worker_pool();

    K_ = K;
    N_ = N;
    offsets_.resize(kNumShards);
    sizes_.resize(kNumShards);
    shards_.resize(kNumShards);

    int remaining = N_;
    int offset = 0;
    for (int i = 0; i < kNumShards; ++i) {
        offsets_[i] = offset;

        int size = 0;
        if (i == kNumShards - 1) {
            size = remaining;
        } else {
            int shards_left = kNumShards - i;
            size = align_up(remaining / shards_left, n_align);
        }

        if (size <= 0 || size % n_align != 0 || size > remaining) {
            std::fprintf(stderr,
                         "[ShardedNpuLinear] bad shard split i=%d size=%d remaining=%d\n",
                         i, size, remaining);
            destroy();
            return false;
        }

        sizes_[i] = size;
        remaining -= size;
        offset += size;
    }

    const rknn_core_mask core_masks[kNumShards] = {
        RKNN_NPU_CORE_0,
        RKNN_NPU_CORE_1,
        RKNN_NPU_CORE_2,
    };

    std::fprintf(stderr,
                 "[ShardedNpuLinear] init K=%d N=%d shards: %d/%d/%d%s\n",
                 K_, N_, sizes_[0], sizes_[1], sizes_[2],
                 use_a8w8 ? " a8w8" : "");

    for (int i = 0; i < kNumShards; ++i) {
        const int shard_n = sizes_[i];
        const int shard_offset = offsets_[i];
        std::vector<uint16_t> shard_weight((size_t)K_ * shard_n);

        for (int k = 0; k < K_; ++k) {
            const uint16_t* src = weight_kn + (size_t)k * N_ + shard_offset;
            uint16_t* dst = shard_weight.data() + (size_t)k * shard_n;
            std::memcpy(dst, src, (size_t)shard_n * sizeof(uint16_t));
        }

        std::unique_ptr<ILinearOp> shard;
        if (use_a8w8) {
            std::unique_ptr<NpuLinearW8> w8(new NpuLinearW8());
            w8->set_core_mask(core_masks[i]);
            shard = std::move(w8);
        } else {
            std::unique_ptr<NpuLinear> fp16(new NpuLinear());
            fp16->set_core_mask(core_masks[i]);
            shard = std::move(fp16);
        }
        bool shard_ok = shard->init(K_, shard_n, shard_weight.data());
        if (!shard_ok) {
            if (use_a8w8) {
                std::fprintf(stderr,
                             "[ShardedNpuLinear] W8 shard %d init failed, fallback FP16\n",
                             i);
                shard->destroy();
                std::unique_ptr<NpuLinear> fp16(new NpuLinear());
                fp16->set_core_mask(core_masks[i]);
                shard = std::move(fp16);
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

bool ShardedNpuLinear::forward(const uint16_t* input_f16, int M, uint16_t* output_f16) {
    if (!input_f16 || !output_f16 || K_ <= 0 || N_ <= 0 || M <= 0 ||
        (int)shards_.size() != kNumShards) {
        std::fprintf(stderr,
                     "[ShardedNpuLinear] invalid forward args M=%d K=%d N=%d\n",
                     M, K_, N_);
        return false;
    }

    for (int i = 0; i < kNumShards; ++i) {
        shard_outputs_[i].resize((size_t)M * sizes_[i]);
    }

    std::array<ShardedNpuWorkerPool::Task, kNumShards> tasks{};
    for (int i = 0; i < kNumShards; ++i) {
        tasks[i].shard = shards_[i].get();
        tasks[i].input = input_f16;
        tasks[i].M = M;
        tasks[i].output = shard_outputs_[i].data();
    }

    if (!worker_pool().run(tasks)) {
        std::fprintf(stderr, "[ShardedNpuLinear] shard forward failed\n");
        return false;
    }

    for (int m = 0; m < M; ++m) {
        uint16_t* out_row = output_f16 + (size_t)m * N_;
        for (int i = 0; i < kNumShards; ++i) {
            const uint16_t* src = shard_outputs_[i].data() + (size_t)m * sizes_[i];
            uint16_t* dst = out_row + offsets_[i];
            std::memcpy(dst, src, (size_t)sizes_[i] * sizeof(uint16_t));
        }
    }

    return true;
}

bool ShardedNpuLinear::supports_batch(int M) const {
    return M == 1;
}

bool ShardedNpuLinear::forward_argmax(const uint16_t* input_f16, int M,
                                      int* argmax_id, uint16_t* argmax_value) {
    if (!argmax_id || !input_f16 || K_ <= 0 || N_ <= 0 || M != 1 ||
        (int)shards_.size() != kNumShards) {
        return false;
    }

    std::array<ShardedNpuWorkerPool::Task, kNumShards> tasks{};
    for (int i = 0; i < kNumShards; ++i) {
        tasks[i].shard = shards_[i].get();
        tasks[i].input = input_f16;
        tasks[i].M = 1;
        tasks[i].argmax = true;
    }

    std::array<int, kNumShards> local_ids{};
    std::array<uint16_t, kNumShards> local_values{};
    if (!worker_pool().run_argmax(tasks, &local_ids, &local_values)) {
        std::fprintf(stderr, "[ShardedNpuLinear] shard argmax forward failed\n");
        return false;
    }

    int best = offsets_[0] + local_ids[0];
    uint16_t best_value = local_values[0];
    uint16_t best_key = f16_order_key(best_value);
    for (int si = 0; si < kNumShards; ++si) {
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
    for (auto& shard : shards_) {
        if (shard) shard->destroy();
    }
    shards_.clear();
    offsets_.clear();
    sizes_.clear();
    for (auto& out : shard_outputs_) {
        std::vector<uint16_t>().swap(out);
    }
    K_ = 0;
    N_ = 0;
}
