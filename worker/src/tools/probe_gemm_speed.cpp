#include "core/half.h"
#include "rknn_matmul_api.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace {

enum class OutKind {
    F16,
    F32,
};

struct BenchConfig {
    int M = 1;
    int K = 1536;
    int N = 5984;
    int iters = 50;
    int warmup = 3;
    bool has_core_mask = false;
    rknn_core_mask core_mask = RKNN_NPU_CORE_AUTO;
    const char* core_name = "auto";
    OutKind out_kind = OutKind::F16;
};

struct ShapeCase {
    const char* name;
    int M;
    int K;
    int N;
    int iters;
};

double now_ms() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(
        clock::now().time_since_epoch()).count();
}

bool parse_int(const char* s, int* out) {
    if (!s || !out) return false;
    char* end = nullptr;
    long v = std::strtol(s, &end, 10);
    if (end == s || *end != '\0' || v <= 0 || v > 1000000000L) {
        return false;
    }
    *out = (int)v;
    return true;
}

bool parse_core(const char* s, bool* has_mask, rknn_core_mask* mask, const char** name) {
    if (!s || !has_mask || !mask || !name) return false;
    std::string v(s);
    if (v == "auto") {
        *has_mask = false;
        *mask = RKNN_NPU_CORE_AUTO;
        *name = "auto";
        return true;
    }
    if (v == "0") {
        *has_mask = true;
        *mask = RKNN_NPU_CORE_0;
        *name = "0";
        return true;
    }
    if (v == "1") {
        *has_mask = true;
        *mask = RKNN_NPU_CORE_1;
        *name = "1";
        return true;
    }
    if (v == "2") {
        *has_mask = true;
        *mask = RKNN_NPU_CORE_2;
        *name = "2";
        return true;
    }
    if (v == "01") {
        *has_mask = true;
        *mask = RKNN_NPU_CORE_0_1;
        *name = "01";
        return true;
    }
    if (v == "012") {
        *has_mask = true;
        *mask = RKNN_NPU_CORE_0_1_2;
        *name = "012";
        return true;
    }
    if (v == "all") {
        *has_mask = true;
        *mask = RKNN_NPU_CORE_ALL;
        *name = "all";
        return true;
    }
    return false;
}

bool parse_out_kind(const char* s, OutKind* out) {
    if (!s || !out) return false;
    std::string v(s);
    if (v == "f16") {
        *out = OutKind::F16;
        return true;
    }
    if (v == "f32") {
        *out = OutKind::F32;
        return true;
    }
    return false;
}

const char* out_kind_name(OutKind kind) {
    return kind == OutKind::F32 ? "f32" : "f16";
}

void print_usage(const char* argv0) {
    std::fprintf(stderr,
                 "Usage:\n"
                 "  %s M K N [iters] [core] [out]\n"
                 "  %s --suite [iters] [core] [out]\n"
                 "\n"
                 "core: auto | 0 | 1 | 2 | 01 | 012 | all   (default: auto)\n"
                 "out : f16 | f32                            (default: f16)\n"
                 "\n"
                 "Examples:\n"
                 "  %s 1 1536 5984 100 0 f16\n"
                 "  %s 23 1536 5984 20 0 f16\n"
                 "  %s 1 8960 512 100 0 f16\n",
                 argv0, argv0, argv0, argv0, argv0);
}

void parse_optional_args(int start, int argc, char** argv, BenchConfig* cfg) {
    for (int i = start; i < argc; ++i) {
        int parsed = 0;
        if (parse_core(argv[i], &cfg->has_core_mask, &cfg->core_mask, &cfg->core_name)) {
            continue;
        }
        if (parse_out_kind(argv[i], &cfg->out_kind)) {
            continue;
        }
        if (parse_int(argv[i], &parsed)) {
            cfg->iters = parsed;
            continue;
        }
        std::fprintf(stderr, "Unknown argument: %s\n", argv[i]);
        print_usage(argv[0]);
        std::exit(2);
    }
}

void fill_inputs(std::vector<uint16_t>* A, std::vector<uint16_t>* B, int M, int K, int N) {
    A->resize((size_t)M * K);
    B->resize((size_t)K * N);
    for (int m = 0; m < M; ++m) {
        for (int k = 0; k < K; ++k) {
            const int v = (m * 13 + k * 7) % 31 - 15;
            (*A)[(size_t)m * K + k] = f32_to_f16((float)v * 0.015625f);
        }
    }
    for (int k = 0; k < K; ++k) {
        for (int n = 0; n < N; ++n) {
            const int v = (k * 11 + n * 5) % 29 - 14;
            (*B)[(size_t)k * N + n] = f32_to_f16((float)v * 0.0078125f);
        }
    }
}

void convert_b_to_f32(const std::vector<uint16_t>& B, std::vector<float>* B_f32) {
    B_f32->resize(B.size());
    for (size_t i = 0; i < B.size(); ++i) {
        (*B_f32)[i] = f16_to_f32(B[i]);
    }
}

void cpu_gemm_once(const uint16_t* A,
                   const float* B_f32,
                   int M,
                   int K,
                   int N,
                   OutKind out_kind,
                   uint16_t* C_f16,
                   float* C_f32,
                   std::vector<float>* scratch) {
    for (int m = 0; m < M; ++m) {
        float* acc = nullptr;
        if (out_kind == OutKind::F32) {
            acc = C_f32 + (size_t)m * N;
            std::fill(acc, acc + N, 0.0f);
        } else {
            scratch->assign((size_t)N, 0.0f);
            acc = scratch->data();
        }

        const uint16_t* a_row = A + (size_t)m * K;
        for (int k = 0; k < K; ++k) {
            const float x = f16_to_f32(a_row[k]);
            const float* b_row = B_f32 + (size_t)k * N;
            int n = 0;
#if defined(__aarch64__)
            const float32x4_t xv = vdupq_n_f32(x);
            for (; n + 4 <= N; n += 4) {
                float32x4_t av = vld1q_f32(acc + n);
                const float32x4_t bv = vld1q_f32(b_row + n);
                av = vfmaq_f32(av, xv, bv);
                vst1q_f32(acc + n, av);
            }
#endif
            for (; n < N; ++n) {
                acc[n] += x * b_row[n];
            }
        }

        if (out_kind == OutKind::F16) {
            uint16_t* c_row = C_f16 + (size_t)m * N;
            int n = 0;
#if defined(__aarch64__)
            for (; n + 4 <= N; n += 4) {
                const float32x4_t v = vld1q_f32(acc + n);
                const float16x4_t h = vcvt_f16_f32(v);
                vst1_f16(reinterpret_cast<float16_t*>(c_row + n), h);
            }
#endif
            for (; n < N; ++n) {
                c_row[n] = f32_to_f16(acc[n]);
            }
        }
    }
}

double checksum_f16(const uint16_t* data, size_t count) {
    double sum = 0.0;
    const size_t step = std::max<size_t>(1, count / 4096);
    for (size_t i = 0; i < count; i += step) {
        sum += f16_to_f32(data[i]) * (double)((i % 17) + 1);
    }
    return sum;
}

double checksum_f32(const float* data, size_t count) {
    double sum = 0.0;
    const size_t step = std::max<size_t>(1, count / 4096);
    for (size_t i = 0; i < count; i += step) {
        sum += (double)data[i] * (double)((i % 17) + 1);
    }
    return sum;
}

struct CpuBenchResult {
    double avg_ms = 0.0;
    double gops = 0.0;
    double checksum = 0.0;
    std::vector<uint16_t> out_f16;
    std::vector<float> out_f32;
};

CpuBenchResult bench_cpu(const BenchConfig& cfg,
                         const std::vector<uint16_t>& A,
                         const std::vector<float>& B_f32) {
    CpuBenchResult result;
    const size_t c_count = (size_t)cfg.M * cfg.N;
    if (cfg.out_kind == OutKind::F32) {
        result.out_f32.resize(c_count);
    } else {
        result.out_f16.resize(c_count);
    }
    std::vector<float> scratch((size_t)cfg.N);

    for (int i = 0; i < cfg.warmup; ++i) {
        cpu_gemm_once(A.data(), B_f32.data(), cfg.M, cfg.K, cfg.N, cfg.out_kind,
                      result.out_f16.empty() ? nullptr : result.out_f16.data(),
                      result.out_f32.empty() ? nullptr : result.out_f32.data(),
                      &scratch);
    }

    const double t0 = now_ms();
    for (int i = 0; i < cfg.iters; ++i) {
        cpu_gemm_once(A.data(), B_f32.data(), cfg.M, cfg.K, cfg.N, cfg.out_kind,
                      result.out_f16.empty() ? nullptr : result.out_f16.data(),
                      result.out_f32.empty() ? nullptr : result.out_f32.data(),
                      &scratch);
    }
    const double t1 = now_ms();
    result.avg_ms = (t1 - t0) / std::max(1, cfg.iters);
    const double ops = 2.0 * (double)cfg.M * (double)cfg.K * (double)cfg.N;
    result.gops = ops / (result.avg_ms / 1000.0) / 1.0e9;
    result.checksum = cfg.out_kind == OutKind::F32
        ? checksum_f32(result.out_f32.data(), c_count)
        : checksum_f16(result.out_f16.data(), c_count);
    return result;
}

class NpuMatmulBench {
public:
    ~NpuMatmulBench() {
        destroy();
    }

    bool init(const BenchConfig& cfg,
              const std::vector<uint16_t>& A,
              const std::vector<uint16_t>& B) {
        cfg_ = cfg;
        destroy();

        info_ = {};
        info_.M = cfg.M;
        info_.K = cfg.K;
        info_.N = cfg.N;
        info_.type = cfg.out_kind == OutKind::F32
            ? RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32
            : RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT16;
        info_.B_layout = RKNN_MM_LAYOUT_NATIVE;
        info_.AC_layout = RKNN_MM_LAYOUT_NORM;
        info_.B_quant_type = RKNN_QUANT_TYPE_PER_LAYER_SYM;
        info_.AC_quant_type = RKNN_QUANT_TYPE_PER_LAYER_SYM;

        int ret = 0;
        if (cfg.M > 1) {
            info_.M = 1;
            rknn_matmul_shape shapes[2] = {
                {1, cfg.K, cfg.N},
                {cfg.M, cfg.K, cfg.N},
            };
            rknn_matmul_io_attr attrs[2] = {};
            ret = rknn_matmul_create_dynamic_shape(&ctx_, &info_, 2, shapes, attrs);
            if (ret < 0) {
                std::fprintf(stderr,
                             "[probe_gemm_speed] dynamic create failed ret=%d M=%d K=%d N=%d\n",
                             ret, cfg.M, cfg.K, cfg.N);
                return false;
            }
            io_attr_ = attrs[1];
            b_attr_ = attrs[0].B;
            ret = rknn_matmul_set_dynamic_shape(ctx_, &shapes[1]);
            if (ret < 0) {
                std::fprintf(stderr,
                             "[probe_gemm_speed] set dynamic shape failed ret=%d M=%d K=%d N=%d\n",
                             ret, cfg.M, cfg.K, cfg.N);
                return false;
            }
        } else {
            ret = rknn_matmul_create(&ctx_, &info_, &io_attr_);
            if (ret < 0) {
                std::fprintf(stderr,
                             "[probe_gemm_speed] create failed ret=%d M=%d K=%d N=%d\n",
                             ret, cfg.M, cfg.K, cfg.N);
                return false;
            }
            b_attr_ = io_attr_.B;
        }

        if (cfg.has_core_mask) {
            ret = rknn_matmul_set_core_mask(ctx_, cfg.core_mask);
            if (ret < 0) {
                std::fprintf(stderr,
                             "[probe_gemm_speed] set_core_mask(%s) failed ret=%d\n",
                             cfg.core_name, ret);
                return false;
            }
        }

        A_mem_ = rknn_create_mem(ctx_, io_attr_.A.size);
        B_mem_ = rknn_create_mem(ctx_, b_attr_.size);
        C_mem_ = rknn_create_mem(ctx_, io_attr_.C.size);
        if (!A_mem_ || !B_mem_ || !C_mem_) {
            std::fprintf(stderr,
                         "[probe_gemm_speed] create_mem failed A=%u B=%u C=%u\n",
                         io_attr_.A.size, b_attr_.size, io_attr_.C.size);
            return false;
        }

        std::memset(A_mem_->virt_addr, 0, io_attr_.A.size);
        std::memset(C_mem_->virt_addr, 0, io_attr_.C.size);
        std::memcpy(A_mem_->virt_addr, A.data(), A.size() * sizeof(uint16_t));
        ret = rknn_B_normal_layout_to_native_layout(
            (void*)B.data(), B_mem_->virt_addr, cfg.K, cfg.N, &info_);
        if (ret < 0) {
            std::fprintf(stderr,
                         "[probe_gemm_speed] B native layout failed ret=%d\n",
                         ret);
            return false;
        }

        ret = rknn_matmul_set_io_mem(ctx_, A_mem_, &io_attr_.A);
        if (ret >= 0) ret = rknn_matmul_set_io_mem(ctx_, B_mem_, &b_attr_);
        if (ret >= 0) ret = rknn_matmul_set_io_mem(ctx_, C_mem_, &io_attr_.C);
        if (ret < 0) {
            std::fprintf(stderr,
                         "[probe_gemm_speed] set_io_mem failed ret=%d\n",
                         ret);
            return false;
        }

        ret = rknn_mem_sync(ctx_, A_mem_, RKNN_MEMORY_SYNC_TO_DEVICE);
        if (ret >= 0) ret = rknn_mem_sync(ctx_, B_mem_, RKNN_MEMORY_SYNC_TO_DEVICE);
        if (ret < 0) {
            std::fprintf(stderr,
                         "[probe_gemm_speed] initial mem_sync TO_DEVICE failed ret=%d\n",
                         ret);
            return false;
        }
        return true;
    }

    bool run_only() {
        const int ret = rknn_matmul_run(ctx_);
        if (ret < 0) {
            std::fprintf(stderr, "[probe_gemm_speed] rknn_matmul_run failed ret=%d\n", ret);
            return false;
        }
        return true;
    }

    bool run_with_ac_sync() {
        int ret = rknn_mem_sync(ctx_, A_mem_, RKNN_MEMORY_SYNC_TO_DEVICE);
        if (ret >= 0) ret = rknn_matmul_run(ctx_);
        if (ret >= 0) ret = rknn_mem_sync(ctx_, C_mem_, RKNN_MEMORY_SYNC_FROM_DEVICE);
        if (ret < 0) {
            std::fprintf(stderr,
                         "[probe_gemm_speed] synced run failed ret=%d\n",
                         ret);
            return false;
        }
        return true;
    }

    bool sync_output() {
        const int ret = rknn_mem_sync(ctx_, C_mem_, RKNN_MEMORY_SYNC_FROM_DEVICE);
        if (ret < 0) {
            std::fprintf(stderr,
                         "[probe_gemm_speed] output mem_sync FROM_DEVICE failed ret=%d\n",
                         ret);
            return false;
        }
        return true;
    }

    const void* output_data() const {
        return C_mem_ ? C_mem_->virt_addr : nullptr;
    }

    uint32_t a_bytes() const { return io_attr_.A.size; }
    uint32_t b_bytes() const { return b_attr_.size; }
    uint32_t c_bytes() const { return io_attr_.C.size; }

private:
    void destroy() {
        if (ctx_) {
            if (A_mem_) rknn_destroy_mem(ctx_, A_mem_);
            if (B_mem_) rknn_destroy_mem(ctx_, B_mem_);
            if (C_mem_) rknn_destroy_mem(ctx_, C_mem_);
            rknn_matmul_destroy(ctx_);
        }
        ctx_ = 0;
        A_mem_ = nullptr;
        B_mem_ = nullptr;
        C_mem_ = nullptr;
        io_attr_ = {};
        b_attr_ = {};
        info_ = {};
    }

    BenchConfig cfg_;
    rknn_matmul_ctx ctx_ = 0;
    rknn_matmul_info info_{};
    rknn_matmul_io_attr io_attr_{};
    rknn_matmul_tensor_attr b_attr_{};
    rknn_tensor_mem* A_mem_ = nullptr;
    rknn_tensor_mem* B_mem_ = nullptr;
    rknn_tensor_mem* C_mem_ = nullptr;
};

struct NpuBenchResult {
    double avg_ms = 0.0;
    double gops = 0.0;
    double checksum = 0.0;
};

NpuBenchResult bench_npu_run_only(const BenchConfig& cfg, NpuMatmulBench* npu) {
    NpuBenchResult result;
    for (int i = 0; i < cfg.warmup; ++i) {
        if (!npu->run_only()) return result;
    }
    const double t0 = now_ms();
    for (int i = 0; i < cfg.iters; ++i) {
        if (!npu->run_only()) return result;
    }
    const double t1 = now_ms();
    npu->sync_output();

    result.avg_ms = (t1 - t0) / std::max(1, cfg.iters);
    const double ops = 2.0 * (double)cfg.M * (double)cfg.K * (double)cfg.N;
    result.gops = ops / (result.avg_ms / 1000.0) / 1.0e9;
    const size_t count = (size_t)cfg.M * cfg.N;
    result.checksum = cfg.out_kind == OutKind::F32
        ? checksum_f32(reinterpret_cast<const float*>(npu->output_data()), count)
        : checksum_f16(reinterpret_cast<const uint16_t*>(npu->output_data()), count);
    return result;
}

NpuBenchResult bench_npu_with_sync(const BenchConfig& cfg, NpuMatmulBench* npu) {
    NpuBenchResult result;
    for (int i = 0; i < cfg.warmup; ++i) {
        if (!npu->run_with_ac_sync()) return result;
    }
    const double t0 = now_ms();
    for (int i = 0; i < cfg.iters; ++i) {
        if (!npu->run_with_ac_sync()) return result;
    }
    const double t1 = now_ms();

    result.avg_ms = (t1 - t0) / std::max(1, cfg.iters);
    const double ops = 2.0 * (double)cfg.M * (double)cfg.K * (double)cfg.N;
    result.gops = ops / (result.avg_ms / 1000.0) / 1.0e9;
    const size_t count = (size_t)cfg.M * cfg.N;
    result.checksum = cfg.out_kind == OutKind::F32
        ? checksum_f32(reinterpret_cast<const float*>(npu->output_data()), count)
        : checksum_f16(reinterpret_cast<const uint16_t*>(npu->output_data()), count);
    return result;
}

void compare_outputs(const BenchConfig& cfg,
                     const CpuBenchResult& cpu,
                     const void* npu_output) {
    if (!npu_output) return;
    const size_t count = (size_t)cfg.M * cfg.N;
    double max_abs = 0.0;
    double max_rel = 0.0;
    size_t max_idx = 0;
    double max_cpu = 0.0;
    double max_npu = 0.0;
    for (size_t i = 0; i < count; ++i) {
        const double c = cfg.out_kind == OutKind::F32
            ? (double)cpu.out_f32[i]
            : (double)f16_to_f32(cpu.out_f16[i]);
        const double n = cfg.out_kind == OutKind::F32
            ? (double)reinterpret_cast<const float*>(npu_output)[i]
            : (double)f16_to_f32(reinterpret_cast<const uint16_t*>(npu_output)[i]);
        const double diff = std::fabs(c - n);
        const double rel = diff / std::max(std::fabs(c), 1e-9);
        if (diff > max_abs) {
            max_abs = diff;
            max_rel = rel;
            max_idx = i;
            max_cpu = c;
            max_npu = n;
        }
    }
    std::printf("verify cpu_vs_npu max_abs=%.6g max_rel=%.6g idx=%zu cpu=%.6g npu=%.6g\n",
                max_abs, max_rel, max_idx, max_cpu, max_npu);
}

bool run_one(const BenchConfig& cfg, const char* case_name) {
    std::printf("\nshape%s%s M=%d K=%d N=%d iters=%d core=%s out=%s\n",
                case_name ? " " : "",
                case_name ? case_name : "",
                cfg.M, cfg.K, cfg.N, cfg.iters, cfg.core_name, out_kind_name(cfg.out_kind));
    const double ops_m = 2.0 * (double)cfg.M * (double)cfg.K * (double)cfg.N / 1.0e6;
    std::printf("ops_per_iter=%.3f Mops\n", ops_m);

    std::vector<uint16_t> A;
    std::vector<uint16_t> B;
    fill_inputs(&A, &B, cfg.M, cfg.K, cfg.N);

    std::vector<float> B_f32;
    const double cpu_init0 = now_ms();
    convert_b_to_f32(B, &B_f32);
    const double cpu_init1 = now_ms();

    const CpuBenchResult cpu = bench_cpu(cfg, A, B_f32);
    std::printf("CPU f32_weight_neon avg=%.3f ms  speed=%.2f GOPS  checksum=%.6f  init_b_f32=%.2f ms\n",
                cpu.avg_ms, cpu.gops, cpu.checksum, cpu_init1 - cpu_init0);

    NpuMatmulBench npu;
    const double npu_init0 = now_ms();
    if (!npu.init(cfg, A, B)) {
        std::fprintf(stderr, "NPU init failed\n");
        return false;
    }
    const double npu_init1 = now_ms();
    std::printf("NPU buffers A=%u B_native=%u C=%u init_native_b=%.2f ms\n",
                npu.a_bytes(), npu.b_bytes(), npu.c_bytes(), npu_init1 - npu_init0);

    const NpuBenchResult npu_run = bench_npu_run_only(cfg, &npu);
    std::printf("NPU run_only       avg=%.3f ms  speed=%.2f GOPS  checksum=%.6f\n",
                npu_run.avg_ms, npu_run.gops, npu_run.checksum);

    const NpuBenchResult npu_sync = bench_npu_with_sync(cfg, &npu);
    std::printf("NPU with_A/C_sync  avg=%.3f ms  speed=%.2f GOPS  checksum=%.6f\n",
                npu_sync.avg_ms, npu_sync.gops, npu_sync.checksum);
    compare_outputs(cfg, cpu, npu.output_data());
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    BenchConfig cfg;
    if (argc < 2) {
        print_usage(argv[0]);
        return 2;
    }

    if (std::strcmp(argv[1], "--suite") == 0) {
        const bool has_iters_override = argc >= 3 && argv[2][0] >= '0' && argv[2][0] <= '9';
        parse_optional_args(2, argc, argv, &cfg);
        const ShapeCase cases[] = {
            {"decode_gate_up_shard", 1, 1536, 5984, 100},
            {"prefill23_gate_up_shard", 23, 1536, 5984, 20},
            {"decode_down_shard", 1, 8960, 512, 100},
            {"prefill23_down_shard", 23, 8960, 512, 20},
            {"decode_qkv", 1, 1536, 2048, 100},
            {"prefill23_qkv", 23, 1536, 2048, 20},
        };
        bool ok = true;
        const int override_iters = cfg.iters;
        for (const auto& c : cases) {
            BenchConfig one = cfg;
            one.M = c.M;
            one.K = c.K;
            one.N = c.N;
            one.iters = has_iters_override ? override_iters : c.iters;
            ok = run_one(one, c.name) && ok;
        }
        return ok ? 0 : 1;
    }

    if (argc < 4 ||
        !parse_int(argv[1], &cfg.M) ||
        !parse_int(argv[2], &cfg.K) ||
        !parse_int(argv[3], &cfg.N)) {
        print_usage(argv[0]);
        return 2;
    }
    parse_optional_args(4, argc, argv, &cfg);
    return run_one(cfg, nullptr) ? 0 : 1;
}
