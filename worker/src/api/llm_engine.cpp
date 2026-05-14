#include "api/llm_engine.h"
#include "model/qwen2_model.h"
#include "ops/op_sampling.h"

#include <algorithm>
#include <cstdio>
#include <sys/time.h>

static int64_t now_us() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return (int64_t)tv.tv_sec * 1000000LL + tv.tv_usec;
}

LLMEngine::LLMEngine()  : model_(new Qwen2Model()) {}
LLMEngine::~LLMEngine() { destroy(); }

bool LLMEngine::load(const std::string& model_dir, LinearBackend backend) {
    return model_->load(model_dir, backend);
}

void LLMEngine::destroy() {
    if (model_) model_->destroy();
}

void LLMEngine::reset() {
    model_->reset_kv_cache();
}

GenerationResult LLMEngine::generate(
    const std::vector<int>& input_ids,
    const GenerationConfig& cfg,
    TokenCallback on_token)
{
    GenerationResult r;
    if (input_ids.empty()) return r;

    auto is_stop = [&](int id) {
        return std::find(cfg.stop_tokens.begin(), cfg.stop_tokens.end(), id)
               != cfg.stop_tokens.end();
    };

    // ---- Prefill ----
    int64_t t0 = now_us();
    auto logits = model_->forward(input_ids);
    r.prefill_ms     = (now_us() - t0) / 1e3f;
    r.prefill_tokens = (int)input_ids.size();

    int next_id = op_greedy_sample(logits);   // 贪心；预留：其他采样分支

    // ---- Decode ----
    // 循环语义与原 main.cpp 完全一致：每一轮先检查 EOS，再推 next_id，再驱动下一次 forward
    int64_t t_decode_start = now_us();
    for (int step = 0; step < cfg.max_new_tokens; ++step) {
        if (is_stop(next_id)) { r.hit_stop = true; break; }

        int64_t ts = now_us();
        r.output_ids.push_back(next_id);

        logits = model_->forward({next_id});
        float elapsed_ms = (now_us() - ts) / 1e3f;

        if (on_token) on_token(step, next_id, elapsed_ms);

        // 采样（预留其他分支）
        next_id = op_greedy_sample(logits);
    }

    r.decode_ms     = (now_us() - t_decode_start) / 1e3f;
    r.decode_tokens = (int)r.output_ids.size();
    return r;
}
