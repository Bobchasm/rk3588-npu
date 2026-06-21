#ifndef WORKER_PC_ENABLE_CUDA

#include "ops/op_attention.h"

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
