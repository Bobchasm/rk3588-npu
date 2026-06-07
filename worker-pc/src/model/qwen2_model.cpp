#include "model/qwen2_model.h"

#include "core/half.h"
#include "model/weight_loader.h"
#include "ops/op_attention.h"
#include "ops/op_cast.h"
#include "ops/op_embedding.h"
#include "ops/op_linear.h"
#include "ops/op_rmsnorm.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

bool init_linear(std::unique_ptr<ILinearOp>& linear,
                 ComputeDevice device,
                 const std::string& sf_path,
                 const TensorMap& meta,
                 const std::string& weight_name,
                 int K,
                 int N) {
    auto w = load_tensor_f16(sf_path, meta.at(weight_name), true);
    linear = make_linear(device);
    if (!linear || !linear->init(K, N, w.data())) {
        if (linear) {
            linear->destroy();
        }
        linear.reset();
        return false;
    }
    return true;
}

int argmax_f16(const std::vector<uint16_t>& values) {
    int best_id = 0;
    float best_value = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < static_cast<int>(values.size()); ++i) {
        const float v = f16_to_f32(values[static_cast<size_t>(i)]);
        if (v > best_value) {
            best_value = v;
            best_id = i;
        }
    }
    return best_id;
}

}  // namespace

Qwen2Model::Qwen2Model() = default;
Qwen2Model::~Qwen2Model() { destroy(); }

void Qwen2Model::destroy() {
    if (lm_head_) {
        lm_head_->destroy();
        lm_head_.reset();
    }
    for (auto& layer : layers_) {
        if (!layer) {
            continue;
        }
        if (layer->q_proj) layer->q_proj->destroy();
        if (layer->k_proj) layer->k_proj->destroy();
        if (layer->v_proj) layer->v_proj->destroy();
        if (layer->o_proj) layer->o_proj->destroy();
        if (layer->gate_proj) layer->gate_proj->destroy();
        if (layer->up_proj) layer->up_proj->destroy();
        if (layer->down_proj) layer->down_proj->destroy();
    }
    std::vector<std::unique_ptr<TransformerLayer>>().swap(layers_);
    std::vector<uint16_t>().swap(embed_tokens_);
    std::vector<float>().swap(norm_weight_);
    kv_cache_ = KVCache();
    scratch_ = ForwardScratch{};
}

bool Qwen2Model::load(const std::string& model_dir, ComputeDevice device) {
    destroy();
    device_ = device;

    const std::string sf_path = model_dir + "/model.safetensors";
    try {
        const TensorMap meta = load_safetensors_meta(sf_path);
        const auto& c = config_;
        const int H = c.hidden_size;
        const int IL = c.num_hidden_layers;
        const int kvd = c.kv_dim();
        const int IS = c.intermediate_size;
        const int V = c.vocab_size;

        embed_tokens_ = load_tensor_f16(sf_path, meta.at("model.embed_tokens.weight"));

        layers_.resize(IL);
        for (int i = 0; i < IL; ++i) {
            std::unique_ptr<TransformerLayer> layer(new TransformerLayer());
            const std::string pfx = "model.layers." + std::to_string(i) + ".";

            layer->input_layernorm = load_tensor_f32(sf_path, meta.at(pfx + "input_layernorm.weight"));
            if (!init_linear(layer->q_proj, device_, sf_path, meta, pfx + "self_attn.q_proj.weight", H, H)) return false;
            if (!init_linear(layer->k_proj, device_, sf_path, meta, pfx + "self_attn.k_proj.weight", H, kvd)) return false;
            if (!init_linear(layer->v_proj, device_, sf_path, meta, pfx + "self_attn.v_proj.weight", H, kvd)) return false;
            if (!init_linear(layer->o_proj, device_, sf_path, meta, pfx + "self_attn.o_proj.weight", H, H)) return false;

            layer->q_bias = load_tensor_f32(sf_path, meta.at(pfx + "self_attn.q_proj.bias"));
            layer->k_bias = load_tensor_f32(sf_path, meta.at(pfx + "self_attn.k_proj.bias"));
            layer->v_bias = load_tensor_f32(sf_path, meta.at(pfx + "self_attn.v_proj.bias"));

            layer->post_attention_layernorm =
                load_tensor_f32(sf_path, meta.at(pfx + "post_attention_layernorm.weight"));
            if (!init_linear(layer->gate_proj, device_, sf_path, meta, pfx + "mlp.gate_proj.weight", H, IS)) return false;
            if (!init_linear(layer->up_proj, device_, sf_path, meta, pfx + "mlp.up_proj.weight", H, IS)) return false;
            if (!init_linear(layer->down_proj, device_, sf_path, meta, pfx + "mlp.down_proj.weight", IS, H)) return false;

            layers_[i] = std::move(layer);
        }

        norm_weight_ = load_tensor_f32(sf_path, meta.at("model.norm.weight"));
        if (!init_linear(lm_head_, device_, sf_path, meta, "model.embed_tokens.weight", H, V)) return false;

        kv_cache_.init(IL, c.max_position, kvd);
        return true;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[worker-pc/Qwen2Model] load failed: %s\n", e.what());
        destroy();
        return false;
    }
}

void Qwen2Model::reset_kv_cache() {
    kv_cache_.reset();
}

void Qwen2Model::ensure_scratch(int seq) {
    const auto& c = config_;
    const int H = c.hidden_size;
    const int kvd = c.kv_dim();
    const int IS = c.intermediate_size;
    const int V = c.vocab_size;

    scratch_.hidden.resize(static_cast<size_t>(seq) * H);
    scratch_.norm_buf.resize(static_cast<size_t>(seq) * H);
    scratch_.q.resize(static_cast<size_t>(seq) * H);
    scratch_.k.resize(static_cast<size_t>(seq) * kvd);
    scratch_.v.resize(static_cast<size_t>(seq) * kvd);
    scratch_.attn_out.resize(static_cast<size_t>(seq) * H);
    scratch_.gate.resize(static_cast<size_t>(seq) * IS);
    scratch_.up.resize(static_cast<size_t>(seq) * IS);
    scratch_.ffn_out.resize(static_cast<size_t>(seq) * H);
    scratch_.last.resize(H);
    scratch_.in_f16.resize(static_cast<size_t>(seq) * H);
    scratch_.q_f16.resize(static_cast<size_t>(seq) * H);
    scratch_.k_f16.resize(static_cast<size_t>(seq) * kvd);
    scratch_.v_f16.resize(static_cast<size_t>(seq) * kvd);
    scratch_.out_f16.resize(static_cast<size_t>(seq) * std::max(H, IS));
    scratch_.gate_f16.resize(static_cast<size_t>(seq) * IS);
    scratch_.up_f16.resize(static_cast<size_t>(seq) * IS);
    scratch_.ffn_in_f16.resize(static_cast<size_t>(seq) * IS);
    scratch_.ffn_out_f16.resize(static_cast<size_t>(seq) * H);
    scratch_.lm_in.resize(H);
    scratch_.lm_out.resize(V);
}

void Qwen2Model::apply_rope(float* q_row, float* k_row, int pos) const {
    const int head_dim = config_.head_dim;
    const int half = head_dim / 2;
    const float theta = config_.rope_theta;

    auto rope_one_head = [&](float* v) {
        for (int i = 0; i < half; ++i) {
            const float freq = 1.0f / std::pow(theta, static_cast<float>(2 * i) / head_dim);
            const float angle = static_cast<float>(pos) * freq;
            const float cos_a = std::cos(angle);
            const float sin_a = std::sin(angle);
            const float v0 = v[i];
            const float v1 = v[i + half];
            v[i] = v0 * cos_a - v1 * sin_a;
            v[i + half] = v0 * sin_a + v1 * cos_a;
        }
    };

    for (int h = 0; h < config_.num_attention_heads; ++h) {
        rope_one_head(q_row + h * head_dim);
    }
    for (int h = 0; h < config_.num_kv_heads; ++h) {
        rope_one_head(k_row + h * head_dim);
    }
}

void Qwen2Model::add_bias(float* x, const float* bias, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        float* row = x + static_cast<size_t>(r) * cols;
        for (int c = 0; c < cols; ++c) {
            row[c] += bias[c];
        }
    }
}

void Qwen2Model::add_residual(float* dst, const float* src, int n) {
    for (int i = 0; i < n; ++i) {
        dst[i] += src[i];
    }
}

void Qwen2Model::silu_inplace(float* x, int n) {
    for (int i = 0; i < n; ++i) {
        x[i] = x[i] / (1.0f + std::exp(-x[i]));
    }
}

int Qwen2Model::forward_next_token(const std::vector<int>& tokens) {
    if (tokens.empty()) {
        throw std::runtime_error("worker-pc/Qwen2Model received empty token list");
    }

    const auto& c = config_;
    const int H = c.hidden_size;
    const int kvd = c.kv_dim();
    const int IS = c.intermediate_size;
    const int seq = static_cast<int>(tokens.size());
    const int pos = kv_cache_.cur_pos();
    const int total_len = pos + seq;

    if (total_len > kv_cache_.capacity()) {
        throw std::runtime_error("worker-pc/KVCache capacity exceeded");
    }

    ensure_scratch(seq);
    op_embedding_lookup(embed_tokens_.data(), tokens, scratch_.hidden.data(), H);

    for (int li = 0; li < c.num_hidden_layers; ++li) {
        TransformerLayer& layer = *layers_[li];

        op_rmsnorm(scratch_.hidden.data(),
                   layer.input_layernorm.data(),
                   scratch_.norm_buf.data(),
                   seq, H, c.rms_norm_eps);

        op_f32_to_f16(scratch_.norm_buf.data(), scratch_.in_f16.data(), seq * H);
        if (!layer.q_proj->forward(scratch_.in_f16.data(), seq, scratch_.q_f16.data()) ||
            !layer.k_proj->forward(scratch_.in_f16.data(), seq, scratch_.k_f16.data()) ||
            !layer.v_proj->forward(scratch_.in_f16.data(), seq, scratch_.v_f16.data())) {
            throw std::runtime_error("worker-pc linear forward failed in qkv");
        }

        for (int i = 0; i < seq * H; ++i) {
            scratch_.q[i] = f16_to_f32(scratch_.q_f16[static_cast<size_t>(i)]);
        }
        for (int i = 0; i < seq * kvd; ++i) {
            scratch_.k[i] = f16_to_f32(scratch_.k_f16[static_cast<size_t>(i)]);
            scratch_.v[i] = f16_to_f32(scratch_.v_f16[static_cast<size_t>(i)]);
        }

        add_bias(scratch_.q.data(), layer.q_bias.data(), seq, H);
        add_bias(scratch_.k.data(), layer.k_bias.data(), seq, kvd);
        add_bias(scratch_.v.data(), layer.v_bias.data(), seq, kvd);

        for (int s = 0; s < seq; ++s) {
            apply_rope(scratch_.q.data() + static_cast<size_t>(s) * H,
                       scratch_.k.data() + static_cast<size_t>(s) * kvd,
                       pos + s);
        }

        uint16_t* kc = kv_cache_.k_ptr(li);
        uint16_t* vc = kv_cache_.v_ptr(li);
        for (int s = 0; s < seq; ++s) {
            const float* ksrc = scratch_.k.data() + static_cast<size_t>(s) * kvd;
            const float* vsrc = scratch_.v.data() + static_cast<size_t>(s) * kvd;
            uint16_t* kdst = kc + static_cast<size_t>(pos + s) * kvd;
            uint16_t* vdst = vc + static_cast<size_t>(pos + s) * kvd;
            for (int d = 0; d < kvd; ++d) {
                kdst[d] = f32_to_f16(ksrc[d]);
                vdst[d] = f32_to_f16(vsrc[d]);
            }
        }

        std::fill(scratch_.attn_out.begin(), scratch_.attn_out.end(), 0.0f);
        if (seq == 1) {
            op_attention_decode(
                scratch_.q.data(),
                kc,
                vc,
                scratch_.attn_out.data(),
                total_len,
                c.num_attention_heads,
                c.num_kv_heads,
                c.head_dim);
        } else {
            op_attention(
                scratch_.q.data(),
                kc,
                vc,
                scratch_.attn_out.data(),
                seq,
                total_len,
                c.num_attention_heads,
                c.num_kv_heads,
                c.head_dim,
                pos);
        }

        op_f32_to_f16(scratch_.attn_out.data(), scratch_.in_f16.data(), seq * H);
        if (!layer.o_proj->forward(scratch_.in_f16.data(), seq, scratch_.out_f16.data())) {
            throw std::runtime_error("worker-pc linear forward failed in o_proj");
        }
        for (int i = 0; i < seq * H; ++i) {
            scratch_.attn_out[i] = f16_to_f32(scratch_.out_f16[static_cast<size_t>(i)]);
        }
        add_residual(scratch_.hidden.data(), scratch_.attn_out.data(), seq * H);

        op_rmsnorm(scratch_.hidden.data(),
                   layer.post_attention_layernorm.data(),
                   scratch_.norm_buf.data(),
                   seq, H, c.rms_norm_eps);

        op_f32_to_f16(scratch_.norm_buf.data(), scratch_.in_f16.data(), seq * H);
        if (!layer.gate_proj->forward(scratch_.in_f16.data(), seq, scratch_.gate_f16.data()) ||
            !layer.up_proj->forward(scratch_.in_f16.data(), seq, scratch_.up_f16.data())) {
            throw std::runtime_error("worker-pc linear forward failed in ffn up/gate");
        }
        for (int i = 0; i < seq * IS; ++i) {
            scratch_.gate[i] = f16_to_f32(scratch_.gate_f16[static_cast<size_t>(i)]);
            scratch_.up[i] = f16_to_f32(scratch_.up_f16[static_cast<size_t>(i)]);
        }

        silu_inplace(scratch_.gate.data(), seq * IS);
        for (int i = 0; i < seq * IS; ++i) {
            scratch_.gate[i] *= scratch_.up[i];
        }

        op_f32_to_f16(scratch_.gate.data(), scratch_.ffn_in_f16.data(), seq * IS);
        if (!layer.down_proj->forward(scratch_.ffn_in_f16.data(), seq, scratch_.ffn_out_f16.data())) {
            throw std::runtime_error("worker-pc linear forward failed in down_proj");
        }
        for (int i = 0; i < seq * H; ++i) {
            scratch_.ffn_out[i] = f16_to_f32(scratch_.ffn_out_f16[static_cast<size_t>(i)]);
        }
        add_residual(scratch_.hidden.data(), scratch_.ffn_out.data(), seq * H);
    }

    op_rmsnorm(scratch_.hidden.data() + static_cast<size_t>(seq - 1) * H,
               norm_weight_.data(),
               scratch_.last.data(),
               1, H, c.rms_norm_eps);
    op_f32_to_f16(scratch_.last.data(), scratch_.lm_in.data(), H);
    if (!lm_head_->forward(scratch_.lm_in.data(), 1, scratch_.lm_out.data())) {
        throw std::runtime_error("worker-pc linear forward failed in lm_head");
    }

    kv_cache_.set_cur_pos(total_len);
    return argmax_f16(scratch_.lm_out);
}

