#include "backend/gpu_linear.h"

#include <cstdio>
#include <cstring>

#if defined(WORKER_PC_ENABLE_CUDA)
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#endif

namespace {

bool& gpu_backend_logged() {
    static bool logged = false;
    return logged;
}

#if defined(WORKER_PC_ENABLE_CUDA)
bool cuda_ok(cudaError_t err, const char* where) {
    if (err == cudaSuccess) {
        return true;
    }
    std::fprintf(stderr, "[worker-pc/GpuLinear] %s failed: %s\n", where, cudaGetErrorString(err));
    return false;
}

bool cublas_ok(cublasStatus_t status, const char* where) {
    if (status == CUBLAS_STATUS_SUCCESS) {
        return true;
    }
    std::fprintf(stderr, "[worker-pc/GpuLinear] %s failed: cublas status=%d\n",
                 where, static_cast<int>(status));
    return false;
}
#endif

}  // namespace

bool GpuLinear::init(int K, int N, const uint16_t* weight_kn) {
    destroy();

    if (K <= 0 || N <= 0 || !weight_kn) {
        std::fprintf(stderr, "[worker-pc/GpuLinear] invalid init args K=%d N=%d\n", K, N);
        return false;
    }

    K_ = K;
    N_ = N;

    const bool cpu_ready = cpu_fallback_.init(K, N, weight_kn);
    if (!cpu_ready) {
        return false;
    }

#if defined(WORKER_PC_ENABLE_CUDA)
    if (init_cuda()) {
        const size_t weight_bytes = static_cast<size_t>(K_) * N_ * sizeof(uint16_t);
        if (cuda_ok(cudaMalloc(&d_weight_, weight_bytes), "cudaMalloc(weight)") &&
            cuda_ok(cudaMemcpy(d_weight_, weight_kn, weight_bytes, cudaMemcpyHostToDevice),
                    "cudaMemcpy(weight)")) {
            using_cuda_ = true;
            if (!gpu_backend_logged()) {
                gpu_backend_logged() = true;
                int device = 0;
                cudaGetDevice(&device);
                cudaDeviceProp prop{};
                const cudaError_t prop_err = cudaGetDeviceProperties(&prop, device);
                if (prop_err == cudaSuccess) {
                    std::fprintf(stderr,
                                 "[worker-pc/GpuLinear] CUDA backend enabled on device=%d name=%s sm=%d.%d mem_mb=%zu\n",
                                 device, prop.name, prop.major, prop.minor,
                                 prop.totalGlobalMem / (1024 * 1024));
                } else {
                    std::fprintf(stderr, "[worker-pc/GpuLinear] CUDA backend enabled\n");
                }
            }
            return true;
        }

        destroy_cuda();
    }
#endif

    if (!warned_once_) {
        warned_once_ = true;
        std::fprintf(stderr,
                     "[worker-pc/GpuLinear] CUDA backend unavailable; fallback to CPU backend.\n");
    }
    return cpu_fallback_.init(K, N, weight_kn);
}

bool GpuLinear::forward(const uint16_t* input_f16, int M, uint16_t* output_f16) {
    if (using_cuda_) {
        return forward_cuda(input_f16, M, output_f16);
    }
    return cpu_fallback_.forward(input_f16, M, output_f16);
}

void GpuLinear::destroy() {
    cpu_fallback_.destroy();
    destroy_cuda();
    K_ = 0;
    N_ = 0;
    workspace_rows_ = 0;
    using_cuda_ = false;
}

bool GpuLinear::init_cuda() {
#if defined(WORKER_PC_ENABLE_CUDA)
    int device_count = 0;
    if (!cuda_ok(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount") || device_count <= 0) {
        return false;
    }
    if (!cublas_ok(cublasCreate(&cublas_), "cublasCreate")) {
        cublas_ = nullptr;
        return false;
    }
    return true;
#else
    return false;
#endif
}

bool GpuLinear::reserve_workspace(int M) {
#if defined(WORKER_PC_ENABLE_CUDA)
    if (M <= workspace_rows_) {
        return true;
    }

    if (d_input_) {
        cudaFree(d_input_);
        d_input_ = nullptr;
    }
    if (d_output_) {
        cudaFree(d_output_);
        d_output_ = nullptr;
    }

    const size_t input_bytes = static_cast<size_t>(M) * K_ * sizeof(uint16_t);
    const size_t output_bytes = static_cast<size_t>(M) * N_ * sizeof(uint16_t);
    if (!cuda_ok(cudaMalloc(&d_input_, input_bytes), "cudaMalloc(input)") ||
        !cuda_ok(cudaMalloc(&d_output_, output_bytes), "cudaMalloc(output)")) {
        return false;
    }
    workspace_rows_ = M;
    return true;
#else
    (void)M;
    return false;
#endif
}

bool GpuLinear::forward_cuda(const uint16_t* input_f16, int M, uint16_t* output_f16) {
#if defined(WORKER_PC_ENABLE_CUDA)
    if (!input_f16 || !output_f16 || M <= 0 || K_ <= 0 || N_ <= 0 || !cublas_ || !d_weight_) {
        std::fprintf(stderr, "[worker-pc/GpuLinear] invalid forward args M=%d K=%d N=%d\n",
                     M, K_, N_);
        return false;
    }

    if (!reserve_workspace(M)) {
        std::fprintf(stderr, "[worker-pc/GpuLinear] reserve workspace failed, fallback to CPU.\n");
        return cpu_fallback_.forward(input_f16, M, output_f16);
    }

    const size_t input_bytes = static_cast<size_t>(M) * K_ * sizeof(uint16_t);
    const size_t output_bytes = static_cast<size_t>(M) * N_ * sizeof(uint16_t);
    if (!cuda_ok(cudaMemcpy(d_input_, input_f16, input_bytes, cudaMemcpyHostToDevice),
                 "cudaMemcpy(input)")) {
        return cpu_fallback_.forward(input_f16, M, output_f16);
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    const cublasStatus_t status = cublasGemmEx(
        cublas_,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N_, M, K_,
        &alpha,
        d_weight_, CUDA_R_16F, N_,
        d_input_, CUDA_R_16F, K_,
        &beta,
        d_output_, CUDA_R_16F, N_,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT);
    if (!cublas_ok(status, "cublasGemmEx")) {
        return cpu_fallback_.forward(input_f16, M, output_f16);
    }

    if (!cuda_ok(cudaMemcpy(output_f16, d_output_, output_bytes, cudaMemcpyDeviceToHost),
                 "cudaMemcpy(output)") ||
        !cuda_ok(cudaDeviceSynchronize(), "cudaDeviceSynchronize")) {
        return cpu_fallback_.forward(input_f16, M, output_f16);
    }
    return true;
#else
    (void)input_f16;
    (void)M;
    (void)output_f16;
    return false;
#endif
}

void GpuLinear::destroy_cuda() {
#if defined(WORKER_PC_ENABLE_CUDA)
    if (d_input_) {
        cudaFree(d_input_);
        d_input_ = nullptr;
    }
    if (d_output_) {
        cudaFree(d_output_);
        d_output_ = nullptr;
    }
    if (d_weight_) {
        cudaFree(d_weight_);
        d_weight_ = nullptr;
    }
    if (cublas_) {
        cublasDestroy(cublas_);
        cublas_ = nullptr;
    }
#endif
}
