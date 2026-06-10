#include "core/half.h"
#include "rknn_matmul_api.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

enum class ElemKind {
    F16,
    BF16,
    F32,
    I4,
    I8,
    I16,
    I32,
};

struct ProbeSpec {
    rknn_matmul_type type;
    const char* name;
    ElemKind a_kind;
    ElemKind b_kind;
    ElemKind c_kind;
};

static int8_t clamp_i4(int v) {
    return (int8_t)std::max(-7, std::min(7, v));
}

static void set_i4(std::vector<uint8_t>& data, size_t idx, int8_t value) {
    uint8_t& byte = data[idx / 2];
    const uint8_t packed = (uint8_t)value & 0x0f;
    if ((idx & 1) == 0) {
        byte = (byte & 0xf0u) | packed;
    } else {
        byte = (byte & 0x0fu) | (uint8_t)(packed << 4);
    }
}

static int8_t get_i4(const std::vector<uint8_t>& data, size_t idx) {
    const uint8_t byte = data[idx / 2];
    uint8_t v = ((idx & 1) == 0) ? (byte & 0x0fu) : (byte >> 4);
    return (v & 0x08u) ? (int8_t)(v | 0xf0u) : (int8_t)v;
}

static size_t compact_bytes(ElemKind kind, size_t elements) {
    switch (kind) {
        case ElemKind::F16:
        case ElemKind::BF16:
        case ElemKind::I16:
            return elements * 2;
        case ElemKind::F32:
        case ElemKind::I32:
            return elements * 4;
        case ElemKind::I8:
            return elements;
        case ElemKind::I4:
            return (elements + 1) / 2;
    }
    return 0;
}

static double a_value(ElemKind kind, const std::vector<uint8_t>& data, size_t idx) {
    switch (kind) {
        case ElemKind::F16:
            return f16_to_f32(reinterpret_cast<const uint16_t*>(data.data())[idx]);
        case ElemKind::I8:
            return reinterpret_cast<const int8_t*>(data.data())[idx];
        case ElemKind::I4:
            return get_i4(data, idx);
        default:
            return 0.0;
    }
}

static double c_value(ElemKind kind, const void* data, size_t idx) {
    switch (kind) {
        case ElemKind::F32:
            return reinterpret_cast<const float*>(data)[idx];
        case ElemKind::F16:
            return f16_to_f32(reinterpret_cast<const uint16_t*>(data)[idx]);
        case ElemKind::BF16: {
            const uint32_t bits = (uint32_t)reinterpret_cast<const uint16_t*>(data)[idx] << 16;
            float out = 0.0f;
            std::memcpy(&out, &bits, sizeof(out));
            return out;
        }
        case ElemKind::I32:
            return reinterpret_cast<const int32_t*>(data)[idx];
        case ElemKind::I16:
            return reinterpret_cast<const int16_t*>(data)[idx];
        case ElemKind::I8:
            return reinterpret_cast<const int8_t*>(data)[idx];
        default:
            return 0.0;
    }
}

static void fill_a(ElemKind kind, std::vector<uint8_t>& data, int K) {
    if (kind == ElemKind::F16) {
        uint16_t* out = reinterpret_cast<uint16_t*>(data.data());
        for (int k = 0; k < K; ++k) {
            out[k] = f32_to_f16(((k * 5) % 17 - 8) * 0.125f);
        }
        return;
    }
    if (kind == ElemKind::I8) {
        int8_t* out = reinterpret_cast<int8_t*>(data.data());
        for (int k = 0; k < K; ++k) {
            out[k] = (int8_t)(((k * 5) % 29) - 14);
        }
        return;
    }
    if (kind == ElemKind::I4) {
        for (int k = 0; k < K; ++k) {
            set_i4(data, (size_t)k, clamp_i4(((k * 5) % 15) - 7));
        }
    }
}

static void fill_b(ElemKind kind, std::vector<uint8_t>& data, int K, int N) {
    if (kind == ElemKind::F16) {
        uint16_t* out = reinterpret_cast<uint16_t*>(data.data());
        for (int k = 0; k < K; ++k) {
            for (int n = 0; n < N; ++n) {
                out[(size_t)k * N + n] =
                    f32_to_f16(((k * 7 + n * 3) % 19 - 9) * 0.0625f);
            }
        }
        return;
    }
    if (kind == ElemKind::I8) {
        int8_t* out = reinterpret_cast<int8_t*>(data.data());
        for (int k = 0; k < K; ++k) {
            for (int n = 0; n < N; ++n) {
                out[(size_t)k * N + n] = (int8_t)(((k * 7 + n * 3) % 31) - 15);
            }
        }
        return;
    }
    if (kind == ElemKind::I4) {
        for (int k = 0; k < K; ++k) {
            for (int n = 0; n < N; ++n) {
                set_i4(data, (size_t)k * N + n, clamp_i4(((k * 7 + n * 3) % 15) - 7));
            }
        }
    }
}

static void unpack_i4_to_i8_bytes(const std::vector<uint8_t>& compact,
                                  std::vector<uint8_t>* unpacked,
                                  int K, int N) {
    unpacked->assign((size_t)K * N, 0);
    for (int k = 0; k < K; ++k) {
        for (int n = 0; n < N; ++n) {
            const size_t idx = (size_t)k * N + n;
            (*unpacked)[idx] = (uint8_t)get_i4(compact, idx);
        }
    }
}

static bool probe_one(const ProbeSpec& spec) {
    constexpr int M = 1;
    constexpr int K = 64;
    constexpr int N = 64;

    std::vector<uint8_t> A(compact_bytes(spec.a_kind, (size_t)M * K), 0);
    std::vector<uint8_t> B(compact_bytes(spec.b_kind, (size_t)K * N), 0);
    fill_a(spec.a_kind, A, K);
    fill_b(spec.b_kind, B, K, N);

    rknn_matmul_info info{};
    info.M = M;
    info.K = K;
    info.N = N;
    info.type = spec.type;
    info.B_layout = RKNN_MM_LAYOUT_NATIVE;
    info.AC_layout = RKNN_MM_LAYOUT_NORM;
    info.B_quant_type = RKNN_QUANT_TYPE_PER_LAYER_SYM;
    info.AC_quant_type = RKNN_QUANT_TYPE_PER_LAYER_SYM;

    rknn_matmul_ctx ctx = 0;
    rknn_matmul_io_attr attr{};
    int ret = rknn_matmul_create(&ctx, &info, &attr);
    if (ret < 0) {
        std::printf("FAIL %-36s create ret=%d\n", spec.name, ret);
        return false;
    }

    rknn_tensor_mem* A_mem = rknn_create_mem(ctx, attr.A.size);
    rknn_tensor_mem* B_mem = rknn_create_mem(ctx, attr.B.size);
    rknn_tensor_mem* C_mem = rknn_create_mem(ctx, attr.C.size);
    auto cleanup = [&]() {
        if (A_mem) rknn_destroy_mem(ctx, A_mem);
        if (B_mem) rknn_destroy_mem(ctx, B_mem);
        if (C_mem) rknn_destroy_mem(ctx, C_mem);
        if (ctx) rknn_matmul_destroy(ctx);
    };

    if (!A_mem || !B_mem || !C_mem) {
        std::printf("FAIL %-36s alloc A=%u B=%u C=%u\n",
                    spec.name, attr.A.size, attr.B.size, attr.C.size);
        cleanup();
        return false;
    }

    std::memset(A_mem->virt_addr, 0, attr.A.size);
    std::memcpy(A_mem->virt_addr, A.data(), A.size());
    std::vector<uint8_t> B_unpacked;
    void* b_normal = B.data();
    if (spec.b_kind == ElemKind::I4) {
        unpack_i4_to_i8_bytes(B, &B_unpacked, K, N);
        b_normal = B_unpacked.data();
    }
    ret = rknn_B_normal_layout_to_native_layout(b_normal, B_mem->virt_addr, K, N, &info);
    if (ret < 0) {
        std::printf("FAIL %-36s B_layout ret=%d\n", spec.name, ret);
        cleanup();
        return false;
    }

    ret = rknn_matmul_set_io_mem(ctx, A_mem, &attr.A);
    if (ret >= 0) ret = rknn_matmul_set_io_mem(ctx, B_mem, &attr.B);
    if (ret >= 0) ret = rknn_matmul_set_io_mem(ctx, C_mem, &attr.C);
    if (ret < 0) {
        std::printf("FAIL %-36s set_io ret=%d\n", spec.name, ret);
        cleanup();
        return false;
    }

    ret = rknn_mem_sync(ctx, A_mem, RKNN_MEMORY_SYNC_TO_DEVICE);
    if (ret >= 0) ret = rknn_mem_sync(ctx, B_mem, RKNN_MEMORY_SYNC_TO_DEVICE);
    if (ret < 0) {
        std::printf("FAIL %-36s mem_sync TO_DEVICE ret=%d\n", spec.name, ret);
        cleanup();
        return false;
    }
    ret = rknn_matmul_run(ctx);
    if (ret < 0) {
        std::printf("FAIL %-36s run ret=%d\n", spec.name, ret);
        cleanup();
        return false;
    }
    ret = rknn_mem_sync(ctx, C_mem, RKNN_MEMORY_SYNC_FROM_DEVICE);
    if (ret < 0) {
        std::printf("FAIL %-36s mem_sync FROM_DEVICE ret=%d\n", spec.name, ret);
        cleanup();
        return false;
    }

    double ref0 = 0.0;
    for (int k = 0; k < K; ++k) {
        ref0 += a_value(spec.a_kind, A, k) * a_value(spec.b_kind, B, (size_t)k * N);
    }
    const double got0 = c_value(spec.c_kind, C_mem->virt_addr, 0);
    std::printf("PASS %-36s A=%u B=%u C=%u Ctype=%d ref0=%.6f got0=%.6f diff0=%.6f\n",
                spec.name, attr.A.size, attr.B.size, attr.C.size, attr.C.type,
                ref0, got0, std::fabs(ref0 - got0));
    cleanup();
    return true;
}

int main() {
    const ProbeSpec specs[] = {
        {RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32, "RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32", ElemKind::F16, ElemKind::F16, ElemKind::F32},
        {RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT16, "RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT16", ElemKind::F16, ElemKind::F16, ElemKind::F16},
        {RKNN_INT8_MM_INT8_TO_INT32,         "RKNN_INT8_MM_INT8_TO_INT32",         ElemKind::I8,  ElemKind::I8,  ElemKind::I32},
        {RKNN_INT8_MM_INT8_TO_INT8,          "RKNN_INT8_MM_INT8_TO_INT8",          ElemKind::I8,  ElemKind::I8,  ElemKind::I8},
        {RKNN_INT8_MM_INT8_TO_FLOAT32,       "RKNN_INT8_MM_INT8_TO_FLOAT32",       ElemKind::I8,  ElemKind::I8,  ElemKind::F32},
        {RKNN_FLOAT16_MM_INT8_TO_FLOAT32,    "RKNN_FLOAT16_MM_INT8_TO_FLOAT32",    ElemKind::F16, ElemKind::I8,  ElemKind::F32},
        {RKNN_FLOAT16_MM_INT8_TO_FLOAT16,    "RKNN_FLOAT16_MM_INT8_TO_FLOAT16",    ElemKind::F16, ElemKind::I8,  ElemKind::F16},
        {RKNN_FLOAT16_MM_INT4_TO_FLOAT32,    "RKNN_FLOAT16_MM_INT4_TO_FLOAT32",    ElemKind::F16, ElemKind::I4,  ElemKind::F32},
        {RKNN_FLOAT16_MM_INT4_TO_FLOAT16,    "RKNN_FLOAT16_MM_INT4_TO_FLOAT16",    ElemKind::F16, ElemKind::I4,  ElemKind::F16},
        {RKNN_FLOAT16_MM_INT4_TO_BFLOAT16,   "RKNN_FLOAT16_MM_INT4_TO_BFLOAT16",   ElemKind::F16, ElemKind::I4,  ElemKind::BF16},
        {RKNN_INT4_MM_INT4_TO_INT16,         "RKNN_INT4_MM_INT4_TO_INT16",         ElemKind::I4,  ElemKind::I4,  ElemKind::I16},
        {RKNN_INT8_MM_INT4_TO_INT32,         "RKNN_INT8_MM_INT4_TO_INT32",         ElemKind::I8,  ElemKind::I4,  ElemKind::I32},
        {RKNN_INT8_MM_INT4_TO_FLOAT16,       "RKNN_INT8_MM_INT4_TO_FLOAT16",       ElemKind::I8,  ElemKind::I4,  ElemKind::F16},
    };

    int pass = 0;
    for (const auto& spec : specs) {
        if (probe_one(spec)) {
            ++pass;
        }
    }
    std::printf("SUMMARY pass=%d total=%zu\n", pass, sizeof(specs) / sizeof(specs[0]));
    return pass == (int)(sizeof(specs) / sizeof(specs[0])) ? 0 : 1;
}
