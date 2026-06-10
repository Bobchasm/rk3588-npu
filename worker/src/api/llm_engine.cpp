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
    // API 层只负责生命周期转发：真正的权重解析、NPU backend 创建、
    // KV cache 分配都在 Qwen2Model::load() 内完成。
    bool ok = model_->load(model_dir, backend);
    if (!ok) model_->destroy();
    return ok;
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
    // 第一次 forward 接收完整 prompt。模型会一次性写入 prompt 的全部 KV，
    // 并返回最后一个 prompt 位置预测出的 next token。
    try {
        int64_t t0 = now_us();
        int next_id = model_->forward_next_token(input_ids);
        r.prefill_ms     = (now_us() - t0) / 1e3f;
        r.prefill_tokens = (int)input_ids.size();

        std::vector<int> one_token(1);

        // ---- Decode ----
        // 后续每步只喂上一步生成的单个 token。KV Cache 已经保存历史，
        // 所以 decode 的主要工作是 seq=1 的增量 forward。
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
