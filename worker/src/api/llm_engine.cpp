#include "api/llm_engine.h"
#include "model/qwen2_model.h"
#include "ops/op_sampling.h"

#include <algorithm>
#include <cstdio>
#include <sys/time.h>
#include <vector>

static int64_t now_us() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return (int64_t)tv.tv_sec * 1000000LL + tv.tv_usec;
}

// ============================================================
// 重复序列检测
//   检查 ids 末尾的 window 个 token，看是否存在长度 [1, window/2] 的
//   n-gram 连续重复至少 2 次。
//   例如 window=6: [40,198,40,198,40,198] → 周期 2，重复 3 次 → 触发
// ============================================================
static bool detect_repetition(const std::vector<int>& ids, int window) {
    if (window <= 0 || (int)ids.size() < window) return false;
    int n = (int)ids.size();
    // 取末尾 window 个 token
    // 枚举周期 p = 1 .. window/2
    for (int p = 1; p <= window / 2; ++p) {
        bool rep = true;
        for (int i = n - 1; i >= n - window; --i) {
            if (ids[i] != ids[i - p]) { rep = false; break; }
            if (i - p < 0) { rep = false; break; }
        }
        if (rep) return true;
    }
    return false;
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

    int next_id = op_greedy_sample(logits);

    // ---- Decode ----
    int64_t t_decode_start = now_us();
    for (int step = 0; step < cfg.max_new_tokens; ++step) {
        // 1. stop token 检测
        if (is_stop(next_id)) { r.hit_stop = true; break; }

        int64_t ts = now_us();
        r.output_ids.push_back(next_id);

        // 2. 重复检测（在 push 之后立即判断，避免无效 forward）
        if (cfg.repetition_window > 0 &&
            detect_repetition(r.output_ids, cfg.repetition_window))
        {
            std::fprintf(stderr, "[LLMEngine] 检测到重复序列，提前停止 (step=%d)\n", step);
            r.hit_repetition = true;
            break;
        }

        logits = model_->forward({next_id});
        float elapsed_ms = (now_us() - ts) / 1e3f;

        if (on_token) on_token(step, next_id, elapsed_ms);

        next_id = op_greedy_sample(logits);
    }

    r.decode_ms     = (now_us() - t_decode_start) / 1e3f;
    r.decode_tokens = (int)r.output_ids.size();
    return r;
}

