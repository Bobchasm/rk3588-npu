#include "ops/op_embedding.h"
#include "core/half.h"

void op_embedding_lookup(const uint16_t* table, const std::vector<int>& ids,
                         float* out, int hidden)
{
    for (int i = 0; i < (int)ids.size(); ++i) {
        const uint16_t* row = table + ids[i] * hidden;
        for (int d = 0; d < hidden; ++d)
            out[i * hidden + d] = f16_to_f32(row[d]);
    }
}
