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

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <sys/time.h>

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
    if (!linear || !linear->init(K, N, w.data())) {
        if (linear) linear->destroy();
        linear.reset();
        return false;
    }
    return true;
}

static void run_linear_or_throw(const std::unique_ptr<ILinearOp>& linear,
                                const char* name,
                                const uint16_t* input_f16,
                                int M,
                                uint16_t* output_f16)
{
    if (!linear || !linear->forward(input_f16, M, output_f16)) {
        throw std::runtime_error(std::string("Linear forward failed: ") + name);
    }
}

static int64_t qwen2_now_us() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return (int64_t)tv.tv_sec * 1000000LL + tv.tv_usec;
}

static bool qwen2_profile_enabled() {
    const char* v = std::getenv("RKLLM_PROFILE");
    return v && v[0] != '\0' && !(v[0] == '0' && v[1] == '\0');
}

static LinearBackend select_lm_head_backend() {
    const char* v = std::getenv("RKLLM_LM_HEAD_BACKEND");
    if (v && (std::strcmp(v, "NPU") == 0 || std::strcmp(v, "npu") == 0)) {
        return LinearBackend::NPU;
    }
    return LinearBackend::CPU;
}

static const char* linear_backend_name(LinearBackend backend) {
    return backend == LinearBackend::NPU ? "NPU" : "CPU";
}

struct ForwardProfile {
    int64_t embedding_us = 0;
    int64_t rmsnorm_us = 0;
    int64_t qkv_proj_us = 0;
    int64_t rope_us = 0;
    int64_t kv_write_us = 0;
    int64_t attention_us = 0;
    int64_t o_proj_us = 0;
    int64_t gate_up_proj_us = 0;
    int64_t silu_mul_us = 0;
    int64_t down_proj_us = 0;
    int64_t lm_head_us = 0;
    int64_t logits_cast_us = 0;
    int64_t total_us = 0;
};

static double us_to_ms(int64_t us) {
    return (double)us / 1000.0;
}

static void print_forward_profile(int seq, int pos, int total_len,
                                  const ForwardProfile& p) {
    std::fprintf(stderr,
        "[profile] forward seq=%d pos=%d total_len=%d total=%.2f ms | "
        "embedding=%.2f rmsnorm=%.2f qkv=%.2f rope=%.2f kv_write=%.2f "
        "attention=%.2f o_proj=%.2f gate_up=%.2f silu_mul=%.2f "
        "down=%.2f lm_head=%.2f logits_cast=%.2f other=%.2f\n",
        seq, pos, total_len, us_to_ms(p.total_us),
        us_to_ms(p.embedding_us), us_to_ms(p.rmsnorm_us),
        us_to_ms(p.qkv_proj_us), us_to_ms(p.rope_us),
        us_to_ms(p.kv_write_us), us_to_ms(p.attention_us),
        us_to_ms(p.o_proj_us), us_to_ms(p.gate_up_proj_us),
        us_to_ms(p.silu_mul_us), us_to_ms(p.down_proj_us),
        us_to_ms(p.lm_head_us), us_to_ms(p.logits_cast_us),
        us_to_ms(p.total_us - p.embedding_us - p.rmsnorm_us -
                 p.qkv_proj_us - p.rope_us - p.kv_write_us -
                 p.attention_us - p.o_proj_us - p.gate_up_proj_us -
                 p.silu_mul_us - p.down_proj_us - p.lm_head_us -
                 p.logits_cast_us));
}

Qwen2Model::Qwen2Model()  = default;
Qwen2Model::~Qwen2Model() { destroy(); }

void Qwen2Model::destroy() {
    // 显式释放所有线性层（顺序：先 lm_head，再逐层），确保 NPU handle 全部归还
    if (lm_head_) { lm_head_->destroy(); lm_head_.reset(); }
    std::vector<std::unique_ptr<TransformerLayer>>().swap(layers_);
    std::vector<uint16_t>().swap(embed_tokens_);
    std::vector<float>().swap(norm_weight_);
    kv_cache_ = KVCache();
}

void Qwen2Model::reset_kv_cache() {
    kv_cache_.reset();
}

// ============================================================
// load: 解析 safetensors、创建每层后端、填充 KV Cache
// ============================================================
bool Qwen2Model::load(const std::string& model_dir, LinearBackend backend) {
    destroy();

    std::string sf_path = model_dir + "/model.safetensors";
    try {
        std::printf("[load] 解析 safetensors 文件头...\n");
        TensorMap meta = load_safetensors_meta(sf_path);
        std::printf("[load] 找到 %zu 个张量\n", meta.size());

        const auto& c  = config_;
        const int H    = c.hidden_size;
        const int IL   = c.num_hidden_layers;
        const int kvd  = c.kv_dim();
        const int IS   = c.intermediate_size;
        const int V    = c.vocab_size;

        auto fail = [this]() {
            destroy();
            return false;
        };

        // ---- Embedding ----
        std::printf("[load] embed_tokens...\n");
        embed_tokens_ = load_tensor_f16(sf_path, meta.at("model.embed_tokens.weight"));

        // ---- 每层 ----
        layers_.resize(IL);
        for (int i = 0; i < IL; ++i) {
            std::unique_ptr<TransformerLayer> L(new TransformerLayer());
            std::string pfx = "model.layers." + std::to_string(i) + ".";

            std::printf("[load] layer %d/%d\r", i+1, IL);
            std::fflush(stdout);

            L->input_layernorm = load_tensor_f32(sf_path, meta.at(pfx + "input_layernorm.weight"));

            if (!init_linear(L->q_proj, backend, sf_path, meta, pfx + "self_attn.q_proj.weight", H, H)) return fail();
            L->q_bias = load_tensor_f32(sf_path, meta.at(pfx + "self_attn.q_proj.bias"));

            if (!init_linear(L->k_proj, backend, sf_path, meta, pfx + "self_attn.k_proj.weight", H, kvd)) return fail();
            L->k_bias = load_tensor_f32(sf_path, meta.at(pfx + "self_attn.k_proj.bias"));

            if (!init_linear(L->v_proj, backend, sf_path, meta, pfx + "self_attn.v_proj.weight", H, kvd)) return fail();
            L->v_bias = load_tensor_f32(sf_path, meta.at(pfx + "self_attn.v_proj.bias"));

            if (!init_linear(L->o_proj, backend, sf_path, meta, pfx + "self_attn.o_proj.weight", H, H)) return fail();

            L->post_attention_layernorm = load_tensor_f32(sf_path, meta.at(pfx + "post_attention_layernorm.weight"));

            if (!init_linear(L->gate_proj, backend, sf_path, meta, pfx + "mlp.gate_proj.weight", H,  IS)) return fail();
            if (!init_linear(L->up_proj,   backend, sf_path, meta, pfx + "mlp.up_proj.weight",   H,  IS)) return fail();
            if (!init_linear(L->down_proj, backend, sf_path, meta, pfx + "mlp.down_proj.weight", IS, H )) return fail();

            layers_[i] = std::move(L);
        }
        std::printf("\n[load] 所有层加载完毕\n");

        // ---- final norm ----
        norm_weight_ = load_tensor_f32(sf_path, meta.at("model.norm.weight"));

        // ---- lm_head（tied weights，复用 embed_tokens 的转置 = [H, V]）----
        LinearBackend lm_head_backend = select_lm_head_backend();
        std::printf("[load] lm_head (%s)...\n", linear_backend_name(lm_head_backend));
        if (!init_linear(lm_head_, lm_head_backend, sf_path, meta, "model.embed_tokens.weight", H, V))
            return fail();

        // ---- KV Cache ----
        kv_cache_.init(IL, c.max_position, kvd);

        std::printf("[load] 加载完成\n");
        return true;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[load] 加载失败: %s\n", e.what());
        destroy();
        return false;
    }
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
    if (tokens.empty()) {
        throw std::runtime_error("Qwen2Model::forward received empty token list");
    }

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

    if (total_len > kv_cache_.capacity()) {
        std::fprintf(stderr,
                     "[KVCache] capacity exceeded: pos=%d seq=%d total=%d capacity=%d\n",
                     pos, seq, total_len, kv_cache_.capacity());
        throw std::runtime_error("KVCache capacity exceeded");
    }

    ForwardProfile prof;
    const bool profile = qwen2_profile_enabled();
    const int64_t total_t0 = profile ? qwen2_now_us() : 0;
    auto profile_block = [&](int64_t& bucket, auto&& fn) {
        if (profile) {
            int64_t t0 = qwen2_now_us();
            fn();
            bucket += qwen2_now_us() - t0;
        } else {
            fn();
        }
    };

    // ---------- embedding ----------
    std::vector<float> hidden((size_t)seq * H);
    profile_block(prof.embedding_us, [&]() {
        op_embedding_lookup(embed_tokens_.data(), tokens, hidden.data(), H);
    });

    std::vector<float>    buf((size_t)seq * H);   // norm 输出 & NPU 前的 FP32 缓冲
    std::vector<uint16_t> npu_in(IS);             // 单 token FP16 输入（最大 IS）
    std::vector<uint16_t> npu_out;

    for (int li = 0; li < c.num_hidden_layers; ++li) {
        TransformerLayer& L = *layers_[li];

        // ---- 1. Input LayerNorm (CPU) ----
        profile_block(prof.rmsnorm_us, [&]() {
            op_rmsnorm(hidden.data(), L.input_layernorm.data(), buf.data(),
                       seq, H, c.rms_norm_eps);
        });

        // ---- 2. Q / K / V proj (NPU, M=1) ----
        std::vector<uint16_t> q_f16((size_t)seq * H);
        std::vector<uint16_t> k_f16((size_t)seq * kv_dim);
        std::vector<uint16_t> v_f16((size_t)seq * kv_dim);

        std::vector<float> q((size_t)seq * H);
        std::vector<float> k((size_t)seq * kv_dim);
        std::vector<float> v((size_t)seq * kv_dim);
        profile_block(prof.qkv_proj_us, [&]() {
            for (int s = 0; s < seq; ++s) {
                op_f32_to_f16(buf.data() + s * H, npu_in.data(), H);
                run_linear_or_throw(L.q_proj, "q_proj", npu_in.data(), 1, q_f16.data() + s * H);
                run_linear_or_throw(L.k_proj, "k_proj", npu_in.data(), 1, k_f16.data() + s * kv_dim);
                run_linear_or_throw(L.v_proj, "v_proj", npu_in.data(), 1, v_f16.data() + s * kv_dim);
            }
            op_f16_to_f32(q_f16.data(), q.data(), seq * H);
            op_f16_to_f32(k_f16.data(), k.data(), seq * kv_dim);
            op_f16_to_f32(v_f16.data(), v.data(), seq * kv_dim);
            op_vec_add_bias(q.data(), L.q_bias.data(), seq, H);
            op_vec_add_bias(k.data(), L.k_bias.data(), seq, kv_dim);
            op_vec_add_bias(v.data(), L.v_bias.data(), seq, kv_dim);
        });

        // ---- 3. RoPE (CPU) ----
        profile_block(prof.rope_us, [&]() {
            for (int s = 0; s < seq; ++s) {
                op_rope(q.data() + s * H,
                        k.data() + s * kv_dim,
                        n_heads, n_kv_heads, head_dim,
                        pos + s, c.rope_theta);
            }
        });

        // ---- 4. 写入 KV Cache ----
        uint16_t* kc = kv_cache_.k_ptr(li);
        uint16_t* vc = kv_cache_.v_ptr(li);
        profile_block(prof.kv_write_us, [&]() {
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
        });

        // ---- 5. Attention (CPU) ----
        std::vector<float> attn_out((size_t)seq * H);
        profile_block(prof.attention_us, [&]() {
            op_attention(q.data(), kc, vc, attn_out.data(),
                         seq, total_len, n_heads, n_kv_heads, head_dim,
                         /*pos_base=*/pos);
        });

        // ---- 6. O proj (NPU, M=1) ----
        npu_out.resize((size_t)seq * H);
        profile_block(prof.o_proj_us, [&]() {
            for (int s = 0; s < seq; ++s) {
                op_f32_to_f16(attn_out.data() + s * H, npu_in.data(), H);
                run_linear_or_throw(L.o_proj, "o_proj", npu_in.data(), 1, npu_out.data() + s * H);
            }
            op_f16_to_f32(npu_out.data(), attn_out.data(), seq * H);
        });

        // ---- 7. Residual ----
        op_vec_add(hidden.data(), attn_out.data(), seq * H);

        // ---- 8. Post-Attention LayerNorm ----
        profile_block(prof.rmsnorm_us, [&]() {
            op_rmsnorm(hidden.data(), L.post_attention_layernorm.data(), buf.data(),
                       seq, H, c.rms_norm_eps);
        });

        // ---- 9. FFN: gate & up proj (NPU, M=1) ----
        std::vector<uint16_t> gate_f16((size_t)seq * IS);
        std::vector<uint16_t> up_f16((size_t)seq * IS);
        std::vector<float> gate((size_t)seq * IS);
        std::vector<float> up  ((size_t)seq * IS);
        profile_block(prof.gate_up_proj_us, [&]() {
            for (int s = 0; s < seq; ++s) {
                op_f32_to_f16(buf.data() + s * H, npu_in.data(), H);
                run_linear_or_throw(L.gate_proj, "gate_proj", npu_in.data(), 1, gate_f16.data() + s * IS);
                run_linear_or_throw(L.up_proj,   "up_proj",   npu_in.data(), 1, up_f16.data()   + s * IS);
            }
            op_f16_to_f32(gate_f16.data(), gate.data(), seq * IS);
            op_f16_to_f32(up_f16.data(),   up.data(),   seq * IS);
        });

        // SiLU(gate) * up
        profile_block(prof.silu_mul_us, [&]() {
            op_silu(gate.data(), seq * IS);
            for (int i = 0; i < seq * IS; ++i) gate[i] *= up[i];
        });

        // ---- 10. down proj (NPU, M=1) ----
        std::vector<uint16_t> ffn_in_f16(IS);
        std::vector<uint16_t> ffn_out_f16((size_t)seq * H);
        std::vector<float> ffn_out((size_t)seq * H);
        profile_block(prof.down_proj_us, [&]() {
            for (int s = 0; s < seq; ++s) {
                op_f32_to_f16(gate.data() + s * IS, ffn_in_f16.data(), IS);
                run_linear_or_throw(L.down_proj, "down_proj", ffn_in_f16.data(), 1, ffn_out_f16.data() + s * H);
            }
            op_f16_to_f32(ffn_out_f16.data(), ffn_out.data(), seq * H);
        });

        // ---- 11. Residual ----
        op_vec_add(hidden.data(), ffn_out.data(), seq * H);
    }

    // ---- Final LayerNorm（仅取最后一个 token，因为只需要下一个 token 的 logits）----
    std::vector<float> last(H);
    profile_block(prof.rmsnorm_us, [&]() {
        op_rmsnorm(hidden.data() + (seq - 1) * H, norm_weight_.data(), last.data(),
                   1, H, c.rms_norm_eps);
    });

    // ---- lm_head (CPU fallback) ----
    std::vector<uint16_t> lm_in(H);
    std::vector<uint16_t> lm_out(c.vocab_size);
    profile_block(prof.lm_head_us, [&]() {
        op_f32_to_f16(last.data(), lm_in.data(), H);
        run_linear_or_throw(lm_head_, "lm_head", lm_in.data(), 1, lm_out.data());
    });

    // 更新 KV Cache 位置
    kv_cache_.set_cur_pos(total_len);

    // FP16 logits -> FP32
    std::vector<float> logits(c.vocab_size);
    profile_block(prof.logits_cast_us, [&]() {
        op_f16_to_f32(lm_out.data(), logits.data(), c.vocab_size);
    });

    if (profile) {
        prof.total_us = qwen2_now_us() - total_t0;
        print_forward_profile(seq, pos, total_len, prof);
    }

    return logits;
}
