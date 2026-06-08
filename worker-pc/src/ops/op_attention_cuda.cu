#include "ops/op_attention.h"

#if defined(WORKER_PC_ENABLE_CUDA)

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <vector>

namespace {

struct AttentionCudaWorkspace {
    float* q_dev = nullptr;
    float* scores_dev = nullptr;
    float* out_dev = nullptr;
    int q_capacity = 0;
    int scores_capacity = 0;
    int out_capacity = 0;

    ~AttentionCudaWorkspace() {
        if (q_dev) cudaFree(q_dev);
        if (scores_dev) cudaFree(scores_dev);
        if (out_dev) cudaFree(out_dev);
    }

    bool reserve(int q_elems, int score_elems, int out_elems) {
        if (q_elems > q_capacity) {
            if (q_dev) cudaFree(q_dev);
            if (cudaMalloc(&q_dev, static_cast<size_t>(q_elems) * sizeof(float)) != cudaSuccess) {
                q_dev = nullptr;
                q_capacity = 0;
                return false;
            }
            q_capacity = q_elems;
        }
        if (score_elems > scores_capacity) {
            if (scores_dev) cudaFree(scores_dev);
            if (cudaMalloc(&scores_dev, static_cast<size_t>(score_elems) * sizeof(float)) != cudaSuccess) {
                scores_dev = nullptr;
                scores_capacity = 0;
                return false;
            }
            scores_capacity = score_elems;
        }
        if (out_elems > out_capacity) {
            if (out_dev) cudaFree(out_dev);
            if (cudaMalloc(&out_dev, static_cast<size_t>(out_elems) * sizeof(float)) != cudaSuccess) {
                out_dev = nullptr;
                out_capacity = 0;
                return false;
            }
            out_capacity = out_elems;
        }
        return true;
    }
};

AttentionCudaWorkspace& workspace() {
    static AttentionCudaWorkspace ws;
    return ws;
}

__global__ void attention_decode_kernel(const float* q,
                                        const uint16_t* k_cache,
                                        const uint16_t* v_cache,
                                        float* scores,
                                        float* out,
                                        int total_len,
                                        int n_heads,
                                        int n_kv_heads,
                                        int head_dim) {
    const int h = blockIdx.x;
    const int kv_h = h / (n_heads / n_kv_heads);
    const int kv_dim = n_kv_heads * head_dim;
    const float* qh = q + h * head_dim;
    const half* v_half = reinterpret_cast<const half*>(v_cache);
    const half* k_half = reinterpret_cast<const half*>(k_cache);
    float* score_row = scores + static_cast<size_t>(h) * total_len;
    const float scale = rsqrtf(static_cast<float>(head_dim));

    if (threadIdx.x == 0) {
        float max_v = -1e30f;
        for (int sk = 0; sk < total_len; ++sk) {
            const half* kh = k_half + static_cast<size_t>(sk) * kv_dim + kv_h * head_dim;
            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                dot += qh[d] * __half2float(kh[d]);
            }
            const float score = dot * scale;
            score_row[sk] = score;
            max_v = fmaxf(max_v, score);
        }

        float sum_v = 0.0f;
        for (int sk = 0; sk < total_len; ++sk) {
            const float v = __expf(score_row[sk] - max_v);
            score_row[sk] = v;
            sum_v += v;
        }
        const float inv_sum = 1.0f / sum_v;
        for (int sk = 0; sk < total_len; ++sk) {
            score_row[sk] *= inv_sum;
        }
    }
    __syncthreads();

    const int d = threadIdx.x;
    if (d >= head_dim) {
        return;
    }
    float acc = 0.0f;
    for (int sk = 0; sk < total_len; ++sk) {
        const half* vh = v_half + static_cast<size_t>(sk) * kv_dim + kv_h * head_dim;
        acc += score_row[sk] * __half2float(vh[d]);
    }
    out[h * head_dim + d] = acc;
}

__global__ void attention_prefill_row_kernel(const float* q,
                                             const uint16_t* k_cache,
                                             const uint16_t* v_cache,
                                             float* score_row,
                                             float* out,
                                             int sq,
                                             int total_len,
                                             int n_heads,
                                             int n_kv_heads,
                                             int head_dim,
                                             int pos_base) {
    const int h = blockIdx.x;
    const int kv_h = h / (n_heads / n_kv_heads);
    const int kv_dim = n_kv_heads * head_dim;
    const int hidden = n_heads * head_dim;
    const int abs_sq = pos_base + sq;
    const float* qh = q + static_cast<size_t>(sq) * hidden + h * head_dim;
    const half* v_half = reinterpret_cast<const half*>(v_cache);
    const half* k_half = reinterpret_cast<const half*>(k_cache);
    const float scale = rsqrtf(static_cast<float>(head_dim));

    if (threadIdx.x == 0) {
        float max_v = -1e30f;
        for (int sk = 0; sk < total_len; ++sk) {
            float score = -1e9f;
            if (sk <= abs_sq) {
                const half* kh = k_half + static_cast<size_t>(sk) * kv_dim + kv_h * head_dim;
                float dot = 0.0f;
                for (int d = 0; d < head_dim; ++d) {
                    dot += qh[d] * __half2float(kh[d]);
                }
                score = dot * scale;
            }
            score_row[sk] = score;
            max_v = fmaxf(max_v, score);
        }

        float sum_v = 0.0f;
        for (int sk = 0; sk < total_len; ++sk) {
            const float v = __expf(score_row[sk] - max_v);
            score_row[sk] = v;
            sum_v += v;
        }
        const float inv_sum = 1.0f / sum_v;
        for (int sk = 0; sk < total_len; ++sk) {
            score_row[sk] *= inv_sum;
        }
    }
    __syncthreads();

    const int d = threadIdx.x;
    if (d >= head_dim) {
        return;
    }
    float acc = 0.0f;
    for (int sk = 0; sk < total_len; ++sk) {
        const half* vh = v_half + static_cast<size_t>(sk) * kv_dim + kv_h * head_dim;
        acc += score_row[sk] * __half2float(vh[d]);
    }
    out[static_cast<size_t>(sq) * hidden + h * head_dim + d] = acc;
}

bool cuda_ok(cudaError_t err, const char* where) {
    if (err == cudaSuccess) {
        return true;
    }
    std::fprintf(stderr, "[worker-pc/op_attention_cuda] %s failed: %s\n",
                 where, cudaGetErrorString(err));
    return false;
}

}  // namespace

bool op_attention_decode_cuda(const float* q_host,
                              const uint16_t* k_cache_dev,
                              const uint16_t* v_cache_dev,
                              float* out_host,
                              int total_len,
                              int n_heads,
                              int n_kv_heads,
                              int head_dim) {
    if (!q_host || !k_cache_dev || !v_cache_dev || !out_host ||
        total_len <= 0 || n_heads <= 0 || n_kv_heads <= 0 || head_dim <= 0) {
        return false;
    }

    AttentionCudaWorkspace& ws = workspace();
    const int hidden = n_heads * head_dim;
    if (!ws.reserve(hidden, total_len * n_heads, hidden)) {
        std::fprintf(stderr, "[worker-pc/op_attention_cuda] reserve workspace failed\n");
        return false;
    }

    if (!cuda_ok(cudaMemcpy(ws.q_dev, q_host, static_cast<size_t>(hidden) * sizeof(float),
                            cudaMemcpyHostToDevice), "cudaMemcpy(q)")) {
        return false;
    }

    attention_decode_kernel<<<n_heads, head_dim>>>(ws.q_dev, k_cache_dev, v_cache_dev,
                                                   ws.scores_dev, ws.out_dev,
                                                   total_len, n_heads, n_kv_heads, head_dim);
    if (!cuda_ok(cudaGetLastError(), "attention_decode_kernel")) {
        return false;
    }

    if (!cuda_ok(cudaMemcpy(out_host, ws.out_dev, static_cast<size_t>(hidden) * sizeof(float),
                            cudaMemcpyDeviceToHost), "cudaMemcpy(out)")) {
        return false;
    }
    return cuda_ok(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
}

bool op_attention_cuda(const float* q_host,
                       const uint16_t* k_cache_dev,
                       const uint16_t* v_cache_dev,
                       float* out_host,
                       int seq,
                       int total_len,
                       int n_heads,
                       int n_kv_heads,
                       int head_dim,
                       int pos_base) {
    if (!q_host || !k_cache_dev || !v_cache_dev || !out_host ||
        seq <= 0 || total_len <= 0 || n_heads <= 0 || n_kv_heads <= 0 || head_dim <= 0) {
        return false;
    }

    AttentionCudaWorkspace& ws = workspace();
    const int hidden = n_heads * head_dim;
    if (!ws.reserve(seq * hidden, total_len, seq * hidden)) {
        std::fprintf(stderr, "[worker-pc/op_attention_cuda] reserve prefill workspace failed\n");
        return false;
    }

    if (!cuda_ok(cudaMemcpy(ws.q_dev, q_host, static_cast<size_t>(seq) * hidden * sizeof(float),
                            cudaMemcpyHostToDevice), "cudaMemcpy(prefill_q)")) {
        return false;
    }

    for (int sq = 0; sq < seq; ++sq) {
        attention_prefill_row_kernel<<<n_heads, head_dim>>>(
            ws.q_dev, k_cache_dev, v_cache_dev, ws.scores_dev, ws.out_dev,
            sq, total_len, n_heads, n_kv_heads, head_dim, pos_base);
        if (!cuda_ok(cudaGetLastError(), "attention_prefill_row_kernel")) {
            return false;
        }
    }

    if (!cuda_ok(cudaMemcpy(out_host, ws.out_dev, static_cast<size_t>(seq) * hidden * sizeof(float),
                            cudaMemcpyDeviceToHost), "cudaMemcpy(prefill_out)")) {
        return false;
    }
    return cuda_ok(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
}

#else

bool op_attention_decode_cuda(const float*,
                              const uint16_t*,
                              const uint16_t*,
                              float*,
                              int,
                              int,
                              int,
                              int) {
    return false;
}

bool op_attention_cuda(const float*,
                       const uint16_t*,
                       const uint16_t*,
                       float*,
                       int,
                       int,
                       int,
                       int,
                       int,
                       int) {
    return false;
}

#endif
