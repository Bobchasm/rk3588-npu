#include "ops/op_attention.h"
#include "ops/op_softmax.h"
#include "core/half.h"

#include <cmath>
#include <vector>
#include <cstring>

void op_attention(
    const float*    q,
    const uint16_t* k_cache,
    const uint16_t* v_cache,
    float*          out,
    int seq, int total_len,
    int n_heads, int n_kv_heads, int head_dim,
    int pos_base)
{
    const int hidden = n_heads * head_dim;
    const int kv_dim = n_kv_heads * head_dim;
    const float scale = 1.0f / std::sqrt((float)head_dim);

    std::memset(out, 0, sizeof(float) * seq * hidden);

    std::vector<float> scores((size_t)seq * total_len);

    for (int h = 0; h < n_heads; ++h) {
        int kv_h = h / (n_heads / n_kv_heads);  // GQA: 多个 query head 共享一个 kv head

        // scores = (q · k^T) * scale，并应用 causal mask
        for (int sq = 0; sq < seq; ++sq) {
            int abs_sq = pos_base + sq;
            for (int sk = 0; sk < total_len; ++sk) {
                if (sk > abs_sq) { scores[sq * total_len + sk] = -1e9f; continue; }
                float dot = 0.0f;
                const float*    qh = q       + sq * hidden + h    * head_dim;
                const uint16_t* kh = k_cache + sk * kv_dim + kv_h * head_dim;
                for (int d = 0; d < head_dim; ++d)
                    dot += qh[d] * f16_to_f32(kh[d]);
                scores[sq * total_len + sk] = dot * scale;
            }
        }

        op_softmax(scores.data(), seq, total_len);

        // out += scores · v
        for (int sq = 0; sq < seq; ++sq) {
            float* out_row = out + sq * hidden + h * head_dim;
            for (int sk = 0; sk < total_len; ++sk) {
                const uint16_t* vh = v_cache + sk * kv_dim + kv_h * head_dim;
                float w = scores[sq * total_len + sk];
                for (int d = 0; d < head_dim; ++d)
                    out_row[d] += w * f16_to_f32(vh[d]);
            }
        }
    }
}
