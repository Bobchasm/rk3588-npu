#include "api/llm_engine.h"
#include "model/qwen2_model.h"

#include <algorithm>
#include <cstdio>
#include <exception>
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
            if (i - p < 0) { rep = false; break; }
            if (ids[i] != ids[i - p]) { rep = false; break; }
        }
        if (rep) return true;
    }
    return false;
}

LLMEngine::LLMEngine()  : model_(new Qwen2Model()) {}
LLMEngine::~LLMEngine() { destroy(); }

bool LLMEngine::load(const std::string& model_dir, LinearBackend backend) {
    return load(model_dir, backend, PartitionConfig{});
}

bool LLMEngine::load(const std::string& model_dir,
                     LinearBackend backend,
                     const PartitionConfig& partition) {
    bool ok = model_->load(model_dir, backend, partition);
    if (!ok) model_->destroy();
    return ok;
}

void LLMEngine::destroy() {
    if (model_) model_->destroy();
}

void LLMEngine::reset() {
    model_->reset_kv_cache();
}

LLMEngine::KvState LLMEngine::snapshot_kv_state() const {
    return model_->snapshot_kv_state();
}

bool LLMEngine::restore_kv_state(const KvState& state) {
    return model_->restore_kv_state(state);
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
    try {
        int64_t t0 = now_us();
        int next_id = model_->forward_next_token(input_ids);
        r.prefill_ms     = (now_us() - t0) / 1e3f;
        r.prefill_tokens = (int)input_ids.size();

        std::vector<int> one_token(1);

        // ---- Decode ----
        int64_t t_decode_start = now_us();
        for (int step = 0; step < cfg.max_new_tokens; ++step) {
            // 1. stop token 检测
            if (is_stop(next_id)) { r.hit_stop = true; break; }

            int64_t ts = now_us();
            const int emit_id = next_id;
            r.output_ids.push_back(emit_id);

            // 2. 重复检测（在 push 之后立即判断，避免无效 forward）
            if (cfg.repetition_window > 0 &&
                detect_repetition(r.output_ids, cfg.repetition_window))
            {
                std::fprintf(stderr, "[LLMEngine] 检测到重复序列，提前停止 (step=%d)\n", step);
                r.hit_repetition = true;
                break;
            }

            if (step + 1 >= cfg.max_new_tokens) {
                break;
            }

            one_token[0] = emit_id;
            next_id = model_->forward_next_token(one_token);
            float elapsed_ms = (now_us() - ts) / 1e3f;

            if (on_token) on_token(step, emit_id, elapsed_ms);

        }

        r.decode_ms     = (now_us() - t_decode_start) / 1e3f;
        r.decode_tokens = (int)r.output_ids.size();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[LLMEngine] generate failed: %s\n", e.what());
        model_->reset_kv_cache();
    }
    return r;
}

bool LLMEngine::forward_tokens_to_hidden(const std::vector<int>& input_ids,
                                         std::vector<uint16_t>& output_f16,
                                         std::string* error) {
    if (!model_) {
        if (error) *error = "rk3588 engine not initialized";
        return false;
    }
    try {
        if (!model_->forward_tokens_to_hidden(input_ids, output_f16)) {
            if (error) *error = "rk3588 tokens_to_hidden returned false";
            return false;
        }
        return true;
    } catch (const std::exception& e) {
        if (error) *error = e.what();
        model_->reset_kv_cache();
        return false;
    }
}

bool LLMEngine::forward_hidden_states(const std::vector<uint16_t>& input_f16,
                                      int seq,
                                      int pos_base,
                                      std::vector<uint16_t>& output_f16,
                                      std::string* error) {
    if (!model_) {
        if (error) *error = "rk3588 engine not initialized";
        return false;
    }
    try {
        if (!model_->forward_hidden_states(input_f16.data(), seq, pos_base, output_f16)) {
            if (error) *error = "rk3588 stage forward returned false";
            return false;
        }
        return true;
    } catch (const std::exception& e) {
        if (error) *error = e.what();
        model_->reset_kv_cache();
        return false;
    }
}

bool LLMEngine::forward_hidden_to_token(const std::vector<uint16_t>& input_f16,
                                        int seq,
                                        int pos_base,
                                        int& output_token_id,
                                        std::string* error) {
    if (!model_) {
        if (error) *error = "rk3588 engine not initialized";
        return false;
    }
    try {
        if (!model_->forward_hidden_to_token(input_f16.data(), seq, pos_base, output_token_id)) {
            if (error) *error = "rk3588 hidden_to_token returned false";
            return false;
        }
        return true;
    } catch (const std::exception& e) {
        if (error) *error = e.what();
        model_->reset_kv_cache();
        return false;
    }
}
