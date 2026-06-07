#include "core/half.h"
#include "rknn_matmul_api.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static int8_t i4_value(int v) {
    v = std::max(-7, std::min(7, v));
    return (int8_t)v;
}

static uint8_t pack_i4_pair(int8_t lo, int8_t hi) {
    return ((uint8_t)lo & 0x0f) | (((uint8_t)hi & 0x0f) << 4);
}

static int8_t unpack_i4(uint8_t byte, bool high) {
    uint8_t v = high ? (byte >> 4) : (byte & 0x0f);
    return (v & 0x08) ? (int8_t)(v | 0xf0) : (int8_t)v;
}

static void set_i4(std::vector<uint8_t>& data, size_t idx, int8_t value) {
    uint8_t& byte = data[idx / 2];
    if ((idx & 1) == 0) {
        byte = (byte & 0xf0) | ((uint8_t)value & 0x0f);
    } else {
        byte = (byte & 0x0f) | (((uint8_t)value & 0x0f) << 4);
    }
}

static int8_t get_i4(const std::vector<uint8_t>& data, size_t idx) {
    return unpack_i4(data[idx / 2], (idx & 1) != 0);
}

static bool run_probe(rknn_matmul_type type) {
    constexpr int M = 1;
    constexpr int K = 64;
    constexpr int N = 32;

    std::vector<uint16_t> A((size_t)M * K);
    std::vector<int8_t> B((size_t)K * N);
    for (int k = 0; k < K; ++k) {
        A[k] = f32_to_f16(((k % 13) - 6) * 0.125f);
        for (int n = 0; n < N; ++n) {
            B[(size_t)k * N + n] = (int8_t)(((k * 7 + n * 3) % 31) - 15);
        }
    }

    rknn_matmul_info info{};
    info.M = M;
    info.K = K;
    info.N = N;
    info.type = type;
    info.B_layout = 1;
    info.AC_layout = 0;
    info.B_quant_type = 0;
    info.AC_quant_type = 0;

    rknn_matmul_ctx ctx = 0;
    rknn_matmul_io_attr attr{};
    int ret = rknn_matmul_create(&ctx, &info, &attr);
    if (ret < 0) {
        std::printf("create %s failed: %d\n", get_matmul_type_string(type), ret);
        return false;
    }

    rknn_tensor_mem* A_mem = rknn_create_mem(ctx, attr.A.size);
    rknn_tensor_mem* B_mem = rknn_create_mem(ctx, attr.B.size);
    rknn_tensor_mem* C_mem = rknn_create_mem(ctx, attr.C.size);
    if (!A_mem || !B_mem || !C_mem) {
        std::printf("create mem failed A=%u B=%u C=%u\n", attr.A.size, attr.B.size, attr.C.size);
        if (A_mem) rknn_destroy_mem(ctx, A_mem);
        if (B_mem) rknn_destroy_mem(ctx, B_mem);
        if (C_mem) rknn_destroy_mem(ctx, C_mem);
        rknn_matmul_destroy(ctx);
        return false;
    }

    std::memcpy(A_mem->virt_addr, A.data(), A.size() * sizeof(uint16_t));
    ret = rknn_B_normal_layout_to_native_layout(B.data(), B_mem->virt_addr, K, N, &info);
    if (ret < 0) {
        std::printf("B layout failed: %d\n", ret);
        return false;
    }
    ret = rknn_matmul_set_io_mem(ctx, A_mem, &attr.A);
    ret |= rknn_matmul_set_io_mem(ctx, B_mem, &attr.B);
    ret |= rknn_matmul_set_io_mem(ctx, C_mem, &attr.C);
    if (ret < 0) {
        std::printf("set io failed: %d\n", ret);
        return false;
    }
    ret = rknn_matmul_run(ctx);
    if (ret < 0) {
        std::printf("run failed: %d\n", ret);
        return false;
    }

    std::printf("%s sizes A=%u B=%u C=%u Ctype=%d\n",
                get_matmul_type_string(type), attr.A.size, attr.B.size, attr.C.size, attr.C.type);
    for (int n = 0; n < 8; ++n) {
        float ref = 0.0f;
        for (int k = 0; k < K; ++k) {
            ref += f16_to_f32(A[k]) * (float)B[(size_t)k * N + n];
        }
        float got = 0.0f;
        if (type == RKNN_FLOAT16_MM_INT8_TO_FLOAT32) {
            got = reinterpret_cast<float*>(C_mem->virt_addr)[n];
        } else {
            got = f16_to_f32(reinterpret_cast<uint16_t*>(C_mem->virt_addr)[n]);
        }
        std::printf("n=%d ref=%.6f got=%.6f diff=%.6f\n",
                    n, ref, got, std::fabs(ref - got));
    }

    rknn_destroy_mem(ctx, A_mem);
    rknn_destroy_mem(ctx, B_mem);
    rknn_destroy_mem(ctx, C_mem);
    rknn_matmul_destroy(ctx);
    return true;
}

static bool run_i8_probe(rknn_matmul_type type) {
    constexpr int M = 1;
    constexpr int K = 64;
    constexpr int N = 32;

    std::vector<int8_t> A((size_t)M * K);
    std::vector<int8_t> B((size_t)K * N);
    for (int k = 0; k < K; ++k) {
        A[k] = (int8_t)(((k * 5) % 29) - 14);
        for (int n = 0; n < N; ++n) {
            B[(size_t)k * N + n] = (int8_t)(((k * 7 + n * 3) % 31) - 15);
        }
    }

    rknn_matmul_info info{};
    info.M = M;
    info.K = K;
    info.N = N;
    info.type = type;
    info.B_layout = 1;
    info.AC_layout = 0;
    info.B_quant_type = 0;
    info.AC_quant_type = 0;

    rknn_matmul_ctx ctx = 0;
    rknn_matmul_io_attr attr{};
    int ret = rknn_matmul_create(&ctx, &info, &attr);
    if (ret < 0) {
        std::printf("create %s failed: %d\n", get_matmul_type_string(type), ret);
        return false;
    }

    rknn_tensor_mem* A_mem = rknn_create_mem(ctx, attr.A.size);
    rknn_tensor_mem* B_mem = rknn_create_mem(ctx, attr.B.size);
    rknn_tensor_mem* C_mem = rknn_create_mem(ctx, attr.C.size);
    if (!A_mem || !B_mem || !C_mem) {
        std::printf("create mem failed A=%u B=%u C=%u\n", attr.A.size, attr.B.size, attr.C.size);
        if (A_mem) rknn_destroy_mem(ctx, A_mem);
        if (B_mem) rknn_destroy_mem(ctx, B_mem);
        if (C_mem) rknn_destroy_mem(ctx, C_mem);
        rknn_matmul_destroy(ctx);
        return false;
    }

    std::memcpy(A_mem->virt_addr, A.data(), A.size() * sizeof(int8_t));
    ret = rknn_B_normal_layout_to_native_layout(B.data(), B_mem->virt_addr, K, N, &info);
    if (ret < 0) {
        std::printf("B layout failed: %d\n", ret);
        return false;
    }
    ret = rknn_matmul_set_io_mem(ctx, A_mem, &attr.A);
    ret |= rknn_matmul_set_io_mem(ctx, B_mem, &attr.B);
    ret |= rknn_matmul_set_io_mem(ctx, C_mem, &attr.C);
    if (ret < 0) {
        std::printf("set io failed: %d\n", ret);
        return false;
    }
    ret = rknn_matmul_run(ctx);
    if (ret < 0) {
        std::printf("run failed: %d\n", ret);
        return false;
    }

    std::printf("%s sizes A=%u B=%u C=%u Ctype=%d\n",
                get_matmul_type_string(type), attr.A.size, attr.B.size, attr.C.size, attr.C.type);
    for (int n = 0; n < 8; ++n) {
        int32_t ref = 0;
        for (int k = 0; k < K; ++k) {
            ref += (int32_t)A[k] * (int32_t)B[(size_t)k * N + n];
        }
        double got = 0.0;
        if (type == RKNN_INT8_MM_INT8_TO_INT32) {
            got = reinterpret_cast<int32_t*>(C_mem->virt_addr)[n];
        } else if (type == RKNN_INT8_MM_INT8_TO_FLOAT32) {
            got = reinterpret_cast<float*>(C_mem->virt_addr)[n];
        } else {
            got = reinterpret_cast<int8_t*>(C_mem->virt_addr)[n];
        }
        std::printf("n=%d ref=%d got=%.6f diff=%.6f\n",
                    n, ref, got, std::fabs((double)ref - got));
    }

    rknn_destroy_mem(ctx, A_mem);
    rknn_destroy_mem(ctx, B_mem);
    rknn_destroy_mem(ctx, C_mem);
    rknn_matmul_destroy(ctx);
    return true;
}

static bool run_i4_probe(rknn_matmul_type type) {
    constexpr int M = 1;
    constexpr int K = 64;
    constexpr int N = 64;

    std::vector<uint8_t> A((size_t)M * K / 2, 0);
    std::vector<uint8_t> B((size_t)K * N / 2, 0);
    for (int k = 0; k < K; ++k) {
        set_i4(A, k, i4_value(((k * 5) % 15) - 7));
        for (int n = 0; n < N; ++n) {
            set_i4(B, (size_t)k * N + n, i4_value(((k * 7 + n * 3) % 15) - 7));
        }
    }

    rknn_matmul_info info{};
    info.M = M;
    info.K = K;
    info.N = N;
    info.type = type;
    info.B_layout = 1;
    info.AC_layout = 0;
    info.B_quant_type = 0;
    info.AC_quant_type = 0;

    rknn_matmul_ctx ctx = 0;
    rknn_matmul_io_attr attr{};
    int ret = rknn_matmul_create(&ctx, &info, &attr);
    if (ret < 0) {
        std::printf("create %s failed: %d\n", get_matmul_type_string(type), ret);
        return false;
    }

    rknn_tensor_mem* A_mem = rknn_create_mem(ctx, attr.A.size);
    rknn_tensor_mem* B_mem = rknn_create_mem(ctx, attr.B.size);
    rknn_tensor_mem* C_mem = rknn_create_mem(ctx, attr.C.size);
    if (!A_mem || !B_mem || !C_mem) {
        std::printf("create mem failed A=%u B=%u C=%u\n", attr.A.size, attr.B.size, attr.C.size);
        if (A_mem) rknn_destroy_mem(ctx, A_mem);
        if (B_mem) rknn_destroy_mem(ctx, B_mem);
        if (C_mem) rknn_destroy_mem(ctx, C_mem);
        rknn_matmul_destroy(ctx);
        return false;
    }

    std::memcpy(A_mem->virt_addr, A.data(), A.size());
    ret = rknn_B_normal_layout_to_native_layout(B.data(), B_mem->virt_addr, K, N, &info);
    if (ret < 0) {
        std::printf("B layout failed: %d\n", ret);
        return false;
    }
    ret = rknn_matmul_set_io_mem(ctx, A_mem, &attr.A);
    ret |= rknn_matmul_set_io_mem(ctx, B_mem, &attr.B);
    ret |= rknn_matmul_set_io_mem(ctx, C_mem, &attr.C);
    if (ret < 0) {
        std::printf("set io failed: %d\n", ret);
        return false;
    }
    ret = rknn_matmul_run(ctx);
    if (ret < 0) {
        std::printf("run failed: %d\n", ret);
        return false;
    }

    std::printf("%s sizes A=%u B=%u C=%u Ctype=%d\n",
                get_matmul_type_string(type), attr.A.size, attr.B.size, attr.C.size, attr.C.type);
    for (int n = 0; n < 8; ++n) {
        int32_t ref = 0;
        for (int k = 0; k < K; ++k) {
            ref += (int32_t)get_i4(A, k) * (int32_t)get_i4(B, (size_t)k * N + n);
        }
        double got = 0.0;
        if (type == RKNN_INT4_MM_INT4_TO_INT16) {
            got = reinterpret_cast<int16_t*>(C_mem->virt_addr)[n];
        } else {
            got = reinterpret_cast<int32_t*>(C_mem->virt_addr)[n];
        }
        std::printf("n=%d ref=%d got=%.6f diff=%.6f\n",
                    n, ref, got, std::fabs((double)ref - got));
    }

    rknn_destroy_mem(ctx, A_mem);
    rknn_destroy_mem(ctx, B_mem);
    rknn_destroy_mem(ctx, C_mem);
    rknn_matmul_destroy(ctx);
    return true;
}

static bool run_i8_i4_probe() {
    constexpr int M = 1;
    constexpr int K = 64;
    constexpr int N = 64;

    std::vector<int8_t> A((size_t)M * K);
    std::vector<uint8_t> B((size_t)K * N / 2, 0);
    for (int k = 0; k < K; ++k) {
        A[k] = (int8_t)(((k * 5) % 29) - 14);
        for (int n = 0; n < N; ++n) {
            set_i4(B, (size_t)k * N + n, i4_value(((k * 7 + n * 3) % 15) - 7));
        }
    }

    rknn_matmul_info info{};
    info.M = M;
    info.K = K;
    info.N = N;
    info.type = RKNN_INT8_MM_INT4_TO_INT32;
    info.B_layout = 1;
    info.AC_layout = 0;
    info.B_quant_type = 0;
    info.AC_quant_type = 0;

    rknn_matmul_ctx ctx = 0;
    rknn_matmul_io_attr attr{};
    int ret = rknn_matmul_create(&ctx, &info, &attr);
    if (ret < 0) {
        std::printf("create %s failed: %d\n", get_matmul_type_string(info.type), ret);
        return false;
    }

    rknn_tensor_mem* A_mem = rknn_create_mem(ctx, attr.A.size);
    rknn_tensor_mem* B_mem = rknn_create_mem(ctx, attr.B.size);
    rknn_tensor_mem* C_mem = rknn_create_mem(ctx, attr.C.size);
    if (!A_mem || !B_mem || !C_mem) {
        std::printf("create mem failed A=%u B=%u C=%u\n", attr.A.size, attr.B.size, attr.C.size);
        if (A_mem) rknn_destroy_mem(ctx, A_mem);
        if (B_mem) rknn_destroy_mem(ctx, B_mem);
        if (C_mem) rknn_destroy_mem(ctx, C_mem);
        rknn_matmul_destroy(ctx);
        return false;
    }

    std::memcpy(A_mem->virt_addr, A.data(), A.size());
    ret = rknn_B_normal_layout_to_native_layout(B.data(), B_mem->virt_addr, K, N, &info);
    if (ret < 0) {
        std::printf("B layout failed: %d\n", ret);
        return false;
    }
    ret = rknn_matmul_set_io_mem(ctx, A_mem, &attr.A);
    ret |= rknn_matmul_set_io_mem(ctx, B_mem, &attr.B);
    ret |= rknn_matmul_set_io_mem(ctx, C_mem, &attr.C);
    if (ret < 0) {
        std::printf("set io failed: %d\n", ret);
        return false;
    }
    ret = rknn_matmul_run(ctx);
    if (ret < 0) {
        std::printf("run failed: %d\n", ret);
        return false;
    }

    std::printf("%s sizes A=%u B=%u C=%u Ctype=%d\n",
                get_matmul_type_string(info.type), attr.A.size, attr.B.size, attr.C.size, attr.C.type);
    for (int n = 0; n < 8; ++n) {
        int32_t ref = 0;
        for (int k = 0; k < K; ++k) {
            ref += (int32_t)A[k] * (int32_t)get_i4(B, (size_t)k * N + n);
        }
        double got = reinterpret_cast<int32_t*>(C_mem->virt_addr)[n];
        std::printf("n=%d ref=%d got=%.6f diff=%.6f\n",
                    n, ref, got, std::fabs((double)ref - got));
    }

    rknn_destroy_mem(ctx, A_mem);
    rknn_destroy_mem(ctx, B_mem);
    rknn_destroy_mem(ctx, C_mem);
    rknn_matmul_destroy(ctx);
    return true;
}

int main() {
    run_probe(RKNN_FLOAT16_MM_INT8_TO_FLOAT32);
    run_probe(RKNN_FLOAT16_MM_INT8_TO_FLOAT16);
    run_i8_probe(RKNN_INT8_MM_INT8_TO_INT32);
    run_i8_probe(RKNN_INT8_MM_INT8_TO_FLOAT32);
    run_i8_probe(RKNN_INT8_MM_INT8_TO_INT8);
    run_i4_probe(RKNN_INT4_MM_INT4_TO_INT16);
    run_i8_i4_probe();
    return 0;
}
