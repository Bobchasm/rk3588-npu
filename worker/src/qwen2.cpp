#include "qwen2.h"
#include "weight_loader.h"
#include "cpu_ops.h"
#include "half.h"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <cassert>
#include <cfloat>
#include <algorithm>

// 工具：FP16 embed lookup
static void embed_lookup(const uint16_t* table, const std::vector<int>& ids,
                         float* out, int hidden) {
    for (int i = 0; i < (int)ids.size(); ++i) {
        const uint16_t* row = table + ids[i] * hidden;
        for (int d = 0; d < hidden; ++d)
            out[i * hidden + d] = f16_to_f32(row[d]);
    }
}

// 加载模型
bool Qwen2Model::load(const std::string& model_dir) {
    std::string sf_path = model_dir + "/model.safetensors";
    printf("[load] 解析 safetensors 文件头...\n");
    TensorMap meta;
    try {
        meta = load_safetensors_meta(sf_path);
    } catch (const std::exception& e) {
        fprintf(stderr, "[load] 解析失败: %s\n", e.what());
        return false;
    }
    printf("[load] 找到 %zu 个张量\n", meta.size());

    const auto& c = config;
    int H  = c.hidden_size;
    int IL = c.num_hidden_layers;
    int kv_dim = c.num_kv_heads * c.head_dim;  // 2 * 128 = 256
    int IS = c.intermediate_size;
    int V  = c.vocab_size;

    // ---- Embedding ----
    printf("[load] embed_tokens...\n");
    embed_tokens = load_tensor_f16(sf_path, meta.at("model.embed_tokens.weight"));

    // ---- 每层 ----
    layers.resize(IL);
    for (int i = 0; i < IL; ++i) {
        auto* L = new TransformerLayer();
        std::string pfx = "model.layers." + std::to_string(i) + ".";

        printf("[load] layer %d/%d\r", i+1, IL);
        fflush(stdout);

        // --- input_layernorm ---
        L->input_layernorm = load_tensor_f32(sf_path,
            meta.at(pfx + "input_layernorm.weight"));

        // --- q_proj: weight [H, H], bias [H] ---
        {
            auto w = load_tensor_f16(sf_path, meta.at(pfx + "self_attn.q_proj.weight"), /*transpose=*/true);
            if (!L->q_proj.init(H, H, w.data())) return false;
            L->q_bias = load_tensor_f32(sf_path, meta.at(pfx + "self_attn.q_proj.bias"));
        }
        // --- k_proj: weight [kv_dim, H] ---
        {
            auto w = load_tensor_f16(sf_path, meta.at(pfx + "self_attn.k_proj.weight"), true);
            if (!L->k_proj.init(H, kv_dim, w.data())) return false;
            L->k_bias = load_tensor_f32(sf_path, meta.at(pfx + "self_attn.k_proj.bias"));
        }
        // --- v_proj: weight [kv_dim, H] ---
        {
            auto w = load_tensor_f16(sf_path, meta.at(pfx + "self_attn.v_proj.weight"), true);
            if (!L->v_proj.init(H, kv_dim, w.data())) return false;
            L->v_bias = load_tensor_f32(sf_path, meta.at(pfx + "self_attn.v_proj.bias"));
        }
        // --- o_proj: weight [H, H]，无 bias ---
        {
            auto w = load_tensor_f16(sf_path, meta.at(pfx + "self_attn.o_proj.weight"), true);
            if (!L->o_proj.init(H, H, w.data())) return false;
        }

        // --- post_attention_layernorm ---
        L->post_attention_layernorm = load_tensor_f32(sf_path,
            meta.at(pfx + "post_attention_layernorm.weight"));

        // --- gate_proj: weight [IS, H] ---
        {
            auto w = load_tensor_f16(sf_path, meta.at(pfx + "mlp.gate_proj.weight"), true);
            if (!L->gate_proj.init(H, IS, w.data())) return false;
        }
        // --- up_proj: weight [IS, H] ---
        {
            auto w = load_tensor_f16(sf_path, meta.at(pfx + "mlp.up_proj.weight"), true);
            if (!L->up_proj.init(H, IS, w.data())) return false;
        }
        // --- down_proj: weight [H, IS] ---
        {
            auto w = load_tensor_f16(sf_path, meta.at(pfx + "mlp.down_proj.weight"), true);
            if (!L->down_proj.init(IS, H, w.data())) return false;
        }

        layers[i].reset(L);
    }
    printf("\n[load] 所有层加载完毕\n");

    // ---- final norm ----
    norm_weight = load_tensor_f32(sf_path, meta.at("model.norm.weight"));

    // ---- lm_head（tied weights = embed_tokens 转置：[H, V]）----
    printf("[load] lm_head...\n");
    {
        // embed_tokens 存储为 [V, H]，需要转置为 [H, V]
        auto w = load_tensor_f16(sf_path, meta.at("model.embed_tokens.weight"), /*transpose=*/true);
        if (!lm_head.init(H, V, w.data())) return false;
    }

    // ---- KV Cache ----
    int max_pos = config.max_position;
    kv_cache.capacity = max_pos;
    kv_cache.kv_dim   = kv_dim;
    kv_cache.k_cache.assign(IL, std::vector<uint16_t>((size_t)max_pos * kv_dim, 0));
    kv_cache.v_cache.assign(IL, std::vector<uint16_t>((size_t)max_pos * kv_dim, 0));

    printf("[load] 加载完成\n");
    return true;
}

void Qwen2Model::reset_kv_cache() {
    for (auto& v : kv_cache.k_cache) std::fill(v.begin(), v.end(), 0);
    for (auto& v : kv_cache.v_cache) std::fill(v.begin(), v.end(), 0);
    kv_cache.cur_pos = 0;
}

// 前向推理（支持多 token prefill + 逐步 decode）
std::vector<float> Qwen2Model::forward(const std::vector<int>& tokens) {
    const auto& c  = config;
    int H          = c.hidden_size;
    int n_heads    = c.num_attention_heads;
    int n_kv_heads = c.num_kv_heads;
    int head_dim   = c.head_dim;
    int kv_dim     = n_kv_heads * head_dim;
    int IS         = c.intermediate_size;
    int seq        = (int)tokens.size();
    int pos        = kv_cache.cur_pos;    // 历史 KV 长度
    int total_len  = pos + seq;           // 本次处理后的总长度

    // ---------- hidden_state [seq, H] ----------
    std::vector<float> hidden(seq * H);
    embed_lookup(embed_tokens.data(), tokens, hidden.data(), H);

    std::vector<float> buf(seq * H);           // 临时缓冲
    std::vector<uint16_t> npu_in(IS);           // 单 token FP16 输入（最大 IS）
    std::vector<uint16_t> npu_out;

    for (int li = 0; li < c.num_hidden_layers; ++li) {
        TransformerLayer& L = *layers[li];

        // ---- 1. Input LayerNorm ----
        rms_norm(hidden.data(), L.input_layernorm.data(), buf.data(),
                 seq, H, c.rms_norm_eps);

        // ---- 2. Q / K / V proj（NPU，每 token 单独调用 M=1）----
        // 注意：NpuLinear 创建时固定 M=1，不能批量调用
        std::vector<uint16_t> q_f16(seq * H);
        std::vector<uint16_t> k_f16(seq * kv_dim);
        std::vector<uint16_t> v_f16(seq * kv_dim);

        for (int s = 0; s < seq; ++s) {
            f32_to_f16_vec(buf.data() + s * H, npu_in.data(), H);
            L.q_proj.forward(npu_in.data(), 1, q_f16.data() + s * H);
            L.k_proj.forward(npu_in.data(), 1, k_f16.data() + s * kv_dim);
            L.v_proj.forward(npu_in.data(), 1, v_f16.data() + s * kv_dim);
        }

        // FP16 -> FP32 + 加 bias
        std::vector<float> q(seq * H), k(seq * kv_dim), v(seq * kv_dim);
        f16_to_f32_vec(q_f16.data(), q.data(), seq * H);
        f16_to_f32_vec(k_f16.data(), k.data(), seq * kv_dim);
        f16_to_f32_vec(v_f16.data(), v.data(), seq * kv_dim);
        vec_add_bias(q.data(), L.q_bias.data(), seq, H);
        vec_add_bias(k.data(), L.k_bias.data(), seq, kv_dim);
        vec_add_bias(v.data(), L.v_bias.data(), seq, kv_dim);

        // ---- 3. RoPE（使用绝对位置）----
        for (int s = 0; s < seq; ++s) {
            apply_rope(q.data() + s * H,
                       k.data() + s * kv_dim,
                       n_heads, n_kv_heads, head_dim,
                       pos + s, c.rope_theta);
        }

        // ---- 4. 写入 KV Cache ----
        uint16_t* kc = kv_cache.k_cache[li].data();
        uint16_t* vc = kv_cache.v_cache[li].data();
        for (int s = 0; s < seq; ++s) {
            const float* ksrc = k.data() + s * kv_dim;
            const float* vsrc = v.data() + s * kv_dim;
            uint16_t* kdst = kc + (pos + s) * kv_dim;
            uint16_t* vdst = vc + (pos + s) * kv_dim;
            for (int d = 0; d < kv_dim; ++d) {
                kdst[d] = f32_to_f16(ksrc[d]);
                vdst[d] = f32_to_f16(vsrc[d]);
            }
        }

        // ---- 5. Attention（使用 KV Cache，scores[seq, total_len]）----
        float scale = 1.0f / sqrtf((float)head_dim);
        std::vector<float> attn_out(seq * H, 0.0f);

        for (int h = 0; h < n_heads; ++h) {
            int kv_h = h / (n_heads / n_kv_heads);

            // scores[sq, sk]：sq 是当前新 token，sk 遍历全部历史 + 当前
            std::vector<float> scores(seq * total_len);
            for (int sq = 0; sq < seq; ++sq) {
                int abs_sq = pos + sq;  // 当前 query 的绝对位置
                for (int sk = 0; sk < total_len; ++sk) {
                    if (sk > abs_sq) { scores[sq * total_len + sk] = -1e9f; continue; }
                    float dot = 0.0f;
                    const float* qh = q.data() + sq * H + h * head_dim;
                    const uint16_t* kh = kc + sk * kv_dim + kv_h * head_dim;
                    for (int d = 0; d < head_dim; ++d)
                        dot += qh[d] * f16_to_f32(kh[d]);
                    scores[sq * total_len + sk] = dot * scale;
                }
            }
            softmax(scores.data(), seq, total_len);

            for (int sq = 0; sq < seq; ++sq) {
                float* out_row = attn_out.data() + sq * H + h * head_dim;
                for (int sk = 0; sk < total_len; ++sk) {
                    const uint16_t* vh = vc + sk * kv_dim + kv_h * head_dim;
                    float w = scores[sq * total_len + sk];
                    for (int d = 0; d < head_dim; ++d)
                        out_row[d] += w * f16_to_f32(vh[d]);
                }
            }
        }

        // ---- 5. O proj（NPU，每 token 单独调用）----
        npu_out.resize(seq * H);
        for (int s = 0; s < seq; ++s) {
            f32_to_f16_vec(attn_out.data() + s * H, npu_in.data(), H);
            L.o_proj.forward(npu_in.data(), 1, npu_out.data() + s * H);
        }
        f16_to_f32_vec(npu_out.data(), attn_out.data(), seq * H);

        // ---- 6. Residual ----
        vec_add(hidden.data(), attn_out.data(), seq * H);

        // ---- 7. Post-Attention LayerNorm ----
        rms_norm(hidden.data(), L.post_attention_layernorm.data(), buf.data(),
                 seq, H, c.rms_norm_eps);

        // ---- 8. FFN: gate_proj & up_proj（NPU，每 token 单独调用）----
        std::vector<uint16_t> gate_f16(seq * IS), up_f16(seq * IS);
        for (int s = 0; s < seq; ++s) {
            f32_to_f16_vec(buf.data() + s * H, npu_in.data(), H);
            L.gate_proj.forward(npu_in.data(), 1, gate_f16.data() + s * IS);
            L.up_proj.forward(npu_in.data(),  1, up_f16.data()   + s * IS);
        }

        std::vector<float> gate(seq * IS), up(seq * IS);
        f16_to_f32_vec(gate_f16.data(), gate.data(), seq * IS);
        f16_to_f32_vec(up_f16.data(),   up.data(),   seq * IS);

        // SiLU(gate) * up
        silu(gate.data(), seq * IS);
        for (int i = 0; i < seq * IS; ++i) gate[i] *= up[i];

        // ---- 9. down_proj（NPU，每 token 单独调用）----
        std::vector<uint16_t> ffn_in_f16(IS);
        std::vector<uint16_t> ffn_out_f16(seq * H);
        for (int s = 0; s < seq; ++s) {
            f32_to_f16_vec(gate.data() + s * IS, ffn_in_f16.data(), IS);
            L.down_proj.forward(ffn_in_f16.data(), 1, ffn_out_f16.data() + s * H);
        }

        std::vector<float> ffn_out(seq * H);
        f16_to_f32_vec(ffn_out_f16.data(), ffn_out.data(), seq * H);

        // ---- 10. Residual ----
        vec_add(hidden.data(), ffn_out.data(), seq * H);
    }

    // ---- Final LayerNorm（取最后一个 token）----
    std::vector<float> last(H);
    rms_norm(hidden.data() + (seq - 1) * H, norm_weight.data(), last.data(),
             1, H, c.rms_norm_eps);

    // ---- lm_head（NPU）----
    std::vector<uint16_t> lm_in(H), lm_out(config.vocab_size);
    f32_to_f16_vec(last.data(), lm_in.data(), H);
    lm_head.forward(lm_in.data(), 1, lm_out.data());

    // 更新 KV Cache 位置指针
    kv_cache.cur_pos = total_len;

    // FP16 logits -> FP32
    std::vector<float> logits(config.vocab_size);
    f16_to_f32_vec(lm_out.data(), logits.data(), config.vocab_size);
    return logits;
}

// 贪心采样
int greedy_sample(const std::vector<float>& logits) {
    int best = 0;
    float best_val = -FLT_MAX;
    for (int i = 0; i < (int)logits.size(); ++i) {
        if (logits[i] > best_val) { best_val = logits[i]; best = i; }
    }
    return best;
}
