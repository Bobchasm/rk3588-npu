#include "model/qwen2_model.h"
#include "model/weight_loader.h"

#include "ops/op_rmsnorm.h"
#include "ops/op_rope.h"
#include "ops/op_silu.h"
#include "ops/op_elementwise.h"
#include "ops/op_cast.h"
#include "ops/op_embedding.h"
#include "ops/op_attention.h"

#include "core/half.h"

#include <cstdio>
#include <cstring>
#include <stdexcept>

// ============================================================
// 辅助：创建并初始化一个 Linear（抽象后端）
//   weight_name 是 safetensors 中的名字
//   transpose=true 表示从 PyTorch [out, in] 转到 [in, out]（= [K, N]）
// ============================================================
static bool init_linear(std::unique_ptr<ILinearOp>& linear,
                        LinearBackend backend,
                        const std::string& sf_path,
                        const TensorMap&   meta,
                        const std::string& weight_name,
                        int K, int N)
{
    auto w = load_tensor_f16(sf_path, meta.at(weight_name), /*transpose=*/true);
    linear = make_linear(backend);
    return linear->init(K, N, w.data());
}

Qwen2Model::Qwen2Model()  = default;
Qwen2Model::~Qwen2Model() { destroy(); }

void Qwen2Model::destroy() {
    // 显式释放所有线性层（顺序：先 lm_head，再逐层），确保 NPU handle 全部归还
    if (lm_head_) { lm_head_->destroy(); lm_head_.reset(); }
    layers_.clear();  // unique_ptr 析构链会调用每个 ILinearOp 的析构
}

void Qwen2Model::reset_kv_cache() {
    kv_cache_.reset();
}

// ============================================================
// load: 解析 safetensors、创建每层后端、填充 KV Cache
// ============================================================
bool Qwen2Model::load(const std::string& model_dir, LinearBackend backend) {
    std::string sf_path = model_dir + "/model.safetensors";
    std::printf("[load] 解析 safetensors 文件头...\n");
    TensorMap meta;
    try {
        meta = load_safetensors_meta(sf_path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[load] 解析失败: %s\n", e.what());
        return false;
    }
    std::printf("[load] 找到 %zu 个张量\n", meta.size());

    const auto& c  = config_;
    const int H    = c.hidden_size;
    const int IL   = c.num_hidden_layers;
    const int kvd  = c.kv_dim();
    const int IS   = c.intermediate_size;
    const int V    = c.vocab_size;

    // ---- Embedding ----
    std::printf("[load] embed_tokens...\n");
    embed_tokens_ = load_tensor_f16(sf_path, meta.at("model.embed_tokens.weight"));

    // ---- 每层 ----
    layers_.resize(IL);
    for (int i = 0; i < IL; ++i) {
        auto* L = new TransformerLayer();
        std::string pfx = "model.layers." + std::to_string(i) + ".";

        std::printf("[load] layer %d/%d\r", i+1, IL);
        std::fflush(stdout);

        L->input_layernorm = load_tensor_f32(sf_path, meta.at(pfx + "input_layernorm.weight"));

        if (!init_linear(L->q_proj, backend, sf_path, meta, pfx + "self_attn.q_proj.weight", H, H)) return false;
        L->q_bias = load_tensor_f32(sf_path, meta.at(pfx + "self_attn.q_proj.bias"));

        if (!init_linear(L->k_proj, backend, sf_path, meta, pfx + "self_attn.k_proj.weight", H, kvd)) return false;
        L->k_bias = load_tensor_f32(sf_path, meta.at(pfx + "self_attn.k_proj.bias"));

        if (!init_linear(L->v_proj, backend, sf_path, meta, pfx + "self_attn.v_proj.weight", H, kvd)) return false;
        L->v_bias = load_tensor_f32(sf_path, meta.at(pfx + "self_attn.v_proj.bias"));

        if (!init_linear(L->o_proj, backend, sf_path, meta, pfx + "self_attn.o_proj.weight", H, H)) return false;

        L->post_attention_layernorm = load_tensor_f32(sf_path, meta.at(pfx + "post_attention_layernorm.weight"));

        if (!init_linear(L->gate_proj, backend, sf_path, meta, pfx + "mlp.gate_proj.weight", H,  IS)) return false;
        if (!init_linear(L->up_proj,   backend, sf_path, meta, pfx + "mlp.up_proj.weight",   H,  IS)) return false;
        if (!init_linear(L->down_proj, backend, sf_path, meta, pfx + "mlp.down_proj.weight", IS, H )) return false;

        layers_[i].reset(L);
    }
    std::printf("\n[load] 所有层加载完毕\n");

    // ---- final norm ----
    norm_weight_ = load_tensor_f32(sf_path, meta.at("model.norm.weight"));

    // ---- lm_head（tied weights，复用 embed_tokens 的转置 = [H, V]）----
    std::printf("[load] lm_head...\n");
    if (!init_linear(lm_head_, backend, sf_path, meta, "model.embed_tokens.weight", H, V))
        return false;

    // ---- KV Cache ----
    kv_cache_.init(IL, c.max_position, kvd);

    std::printf("[load] 加载完成\n");
    return true;
}

// ============================================================
// forward: 28 层 Transformer Block + final norm + lm_head
//
// 输入：tokens（本次要处理的 token id 序列）
// 输出：最后一个位置的 logits
//
// KV Cache 约定：
//   - 进入时：kv_cache_.cur_pos() 是历史已写入的位置数
//   - 返回时：kv_cache_.cur_pos() += tokens.size()
// ============================================================
std::vector<float> Qwen2Model::forward(const std::vector<int>& tokens) {
    const auto& c = config_;
    const int H          = c.hidden_size;
    const int n_heads    = c.num_attention_heads;
    const int n_kv_heads = c.num_kv_heads;
    const int head_dim   = c.head_dim;
    const int kv_dim     = c.kv_dim();
    const int IS         = c.intermediate_size;
    const int seq        = (int)tokens.size();
    const int pos        = kv_cache_.cur_pos();
    const int total_len  = pos + seq;

    // ---------- embedding ----------
    std::vector<float> hidden((size_t)seq * H);
    op_embedding_lookup(embed_tokens_.data(), tokens, hidden.data(), H);

    std::vector<float>    buf((size_t)seq * H);   // norm 输出 & NPU 前的 FP32 缓冲
    std::vector<uint16_t> npu_in(IS);             // 单 token FP16 输入（最大 IS）
    std::vector<uint16_t> npu_out;

    for (int li = 0; li < c.num_hidden_layers; ++li) {
        TransformerLayer& L = *layers_[li];

        // ---- 1. Input LayerNorm (CPU) ----
        op_rmsnorm(hidden.data(), L.input_layernorm.data(), buf.data(),
                   seq, H, c.rms_norm_eps);

        // ---- 2. Q / K / V proj (NPU, M=1) ----
        std::vector<uint16_t> q_f16((size_t)seq * H);
        std::vector<uint16_t> k_f16((size_t)seq * kv_dim);
        std::vector<uint16_t> v_f16((size_t)seq * kv_dim);

        for (int s = 0; s < seq; ++s) {
            op_f32_to_f16(buf.data() + s * H, npu_in.data(), H);
            L.q_proj->forward(npu_in.data(), 1, q_f16.data() + s * H);
            L.k_proj->forward(npu_in.data(), 1, k_f16.data() + s * kv_dim);
            L.v_proj->forward(npu_in.data(), 1, v_f16.data() + s * kv_dim);
        }

        std::vector<float> q((size_t)seq * H);
        std::vector<float> k((size_t)seq * kv_dim);
        std::vector<float> v((size_t)seq * kv_dim);
        op_f16_to_f32(q_f16.data(), q.data(), seq * H);
        op_f16_to_f32(k_f16.data(), k.data(), seq * kv_dim);
        op_f16_to_f32(v_f16.data(), v.data(), seq * kv_dim);
        op_vec_add_bias(q.data(), L.q_bias.data(), seq, H);
        op_vec_add_bias(k.data(), L.k_bias.data(), seq, kv_dim);
        op_vec_add_bias(v.data(), L.v_bias.data(), seq, kv_dim);

        // ---- 3. RoPE (CPU) ----
        for (int s = 0; s < seq; ++s) {
            op_rope(q.data() + s * H,
                    k.data() + s * kv_dim,
                    n_heads, n_kv_heads, head_dim,
                    pos + s, c.rope_theta);
        }

        // ---- 4. 写入 KV Cache ----
        uint16_t* kc = kv_cache_.k_ptr(li);
        uint16_t* vc = kv_cache_.v_ptr(li);
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

        // ---- 5. Attention (CPU) ----
        std::vector<float> attn_out((size_t)seq * H);
        op_attention(q.data(), kc, vc, attn_out.data(),
                     seq, total_len, n_heads, n_kv_heads, head_dim,
                     /*pos_base=*/pos);

        // ---- 6. O proj (NPU, M=1) ----
        npu_out.resize((size_t)seq * H);
        for (int s = 0; s < seq; ++s) {
            op_f32_to_f16(attn_out.data() + s * H, npu_in.data(), H);
            L.o_proj->forward(npu_in.data(), 1, npu_out.data() + s * H);
        }
        op_f16_to_f32(npu_out.data(), attn_out.data(), seq * H);

        // ---- 7. Residual ----
        op_vec_add(hidden.data(), attn_out.data(), seq * H);

        // ---- 8. Post-Attention LayerNorm ----
        op_rmsnorm(hidden.data(), L.post_attention_layernorm.data(), buf.data(),
                   seq, H, c.rms_norm_eps);

        // ---- 9. FFN: gate & up proj (NPU, M=1) ----
        std::vector<uint16_t> gate_f16((size_t)seq * IS);
        std::vector<uint16_t> up_f16((size_t)seq * IS);
        for (int s = 0; s < seq; ++s) {
            op_f32_to_f16(buf.data() + s * H, npu_in.data(), H);
            L.gate_proj->forward(npu_in.data(), 1, gate_f16.data() + s * IS);
            L.up_proj->forward(  npu_in.data(), 1, up_f16.data()   + s * IS);
        }

        std::vector<float> gate((size_t)seq * IS);
        std::vector<float> up  ((size_t)seq * IS);
        op_f16_to_f32(gate_f16.data(), gate.data(), seq * IS);
        op_f16_to_f32(up_f16.data(),   up.data(),   seq * IS);

        // SiLU(gate) * up
        op_silu(gate.data(), seq * IS);
        for (int i = 0; i < seq * IS; ++i) gate[i] *= up[i];

        // ---- 10. down proj (NPU, M=1) ----
        std::vector<uint16_t> ffn_in_f16(IS);
        std::vector<uint16_t> ffn_out_f16((size_t)seq * H);
        for (int s = 0; s < seq; ++s) {
            op_f32_to_f16(gate.data() + s * IS, ffn_in_f16.data(), IS);
            L.down_proj->forward(ffn_in_f16.data(), 1, ffn_out_f16.data() + s * H);
        }

        std::vector<float> ffn_out((size_t)seq * H);
        op_f16_to_f32(ffn_out_f16.data(), ffn_out.data(), seq * H);

        // ---- 11. Residual ----
        op_vec_add(hidden.data(), ffn_out.data(), seq * H);
    }

    // ---- Final LayerNorm（仅取最后一个 token，因为只需要下一个 token 的 logits）----
    std::vector<float> last(H);
    op_rmsnorm(hidden.data() + (seq - 1) * H, norm_weight_.data(), last.data(),
               1, H, c.rms_norm_eps);

    // ---- lm_head (NPU) ----
    std::vector<uint16_t> lm_in(H);
    std::vector<uint16_t> lm_out(c.vocab_size);
    op_f32_to_f16(last.data(), lm_in.data(), H);
    lm_head_->forward(lm_in.data(), 1, lm_out.data());

    // 更新 KV Cache 位置
    kv_cache_.set_cur_pos(total_len);

    // FP16 logits -> FP32
    std::vector<float> logits(c.vocab_size);
    op_f16_to_f32(lm_out.data(), logits.data(), c.vocab_size);
    return logits;
}
