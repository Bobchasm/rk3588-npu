#pragma once
#include <cstdint>
#include <vector>

// ============================================================
// op_embedding: token id -> FP32 hidden state
//   table : [vocab, hidden]  FP16 查表权重
//   ids   : 当前要编码的 token id 列表
//   out   : [seq, hidden]    FP32 输出
// ============================================================

enum class EmbeddingStorageDType {
    FP16,
    BF16,
    FP32,
};

void op_embedding_lookup(const uint16_t* table, const std::vector<int>& ids,
                         float* out, int hidden);

void op_embedding_lookup_typed(const void* table,
                               EmbeddingStorageDType dtype,
                               const std::vector<int>& ids,
                               float* out,
                               int hidden);
