#include "ops/op_embedding.h"
#include "core/half.h"

void op_embedding_lookup(const uint16_t* table, const std::vector<int>& ids,
                         float* out, int hidden)
{
    // embedding 权重以 FP16 存储节省内存；查表后转 FP32，
    // 因为后续 RMSNorm/residual/attention 都以 FP32 累加。
    for (int i = 0; i < (int)ids.size(); ++i) {
        const uint16_t* row = table + ids[i] * hidden;
        for (int d = 0; d < hidden; ++d)
            out[i * hidden + d] = f16_to_f32(row[d]);
    }
}

void op_embedding_lookup_typed(const void* table,
                               EmbeddingStorageDType dtype,
                               const std::vector<int>& ids,
                               float* out,
                               int hidden)
{
    if (dtype == EmbeddingStorageDType::FP16) {
        op_embedding_lookup(reinterpret_cast<const uint16_t*>(table), ids, out, hidden);
        return;
    }

    if (dtype == EmbeddingStorageDType::BF16) {
        // 保持和 load_tensor_f16() 一致的数值路径：BF16 -> FP16 -> FP32。
        const uint16_t* table_u16 = reinterpret_cast<const uint16_t*>(table);
        for (int i = 0; i < (int)ids.size(); ++i) {
            const uint16_t* row = table_u16 + ids[i] * hidden;
            for (int d = 0; d < hidden; ++d)
                out[i * hidden + d] = f16_to_f32(bf16_to_f16(row[d]));
        }
        return;
    }

    const float* table_f32 = reinterpret_cast<const float*>(table);
    for (int i = 0; i < (int)ids.size(); ++i) {
        const float* row = table_f32 + ids[i] * hidden;
        for (int d = 0; d < hidden; ++d)
            out[i * hidden + d] = f16_to_f32(f32_to_f16(row[d]));
    }
}
