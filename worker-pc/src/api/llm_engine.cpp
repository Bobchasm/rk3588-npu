#include "api/llm_engine.h"
#include "model/qwen2_model.h"

#include <algorithm>
#include <cstdio>
#include <exception>
#include <sys/time.h>
#include <vector>

namespace {

int64_t now_us() {
    struct timeval tv{};
    gettimeofday(&tv, nullptr);
    return static_cast<int64_t>(tv.tv_sec) * 1000000LL + tv.tv_usec;
}

bool detect_repetition(const std::vector<int>& ids, int window) {
    if (window <= 0 || static_cast<int>(ids.size()) < window) {
        return false;
    }
    const int n = static_cast<int>(ids.size());
    for (int p = 1; p <= window / 2; ++p) {
        bool rep = true;
        for (int i = n - 1; i >= n - window; --i) {
            if (i - p < 0) {
                rep = false;
                break;
            }
            if (ids[i] != ids[i - p]) {
                rep = false;
                break;
            }
        }
        if (rep) {
            return true;
        }
    }
    return false;
}

}  // namespace

LLMEngine::LLMEngine() : model_(new Qwen2Model()) {}
LLMEngine::~LLMEngine() { destroy(); }

bool LLMEngine::load(const std::string& model_dir, ComputeDevice device) {
    const int64_t t0 = now_us();
    std::fprintf(stderr, "[worker-pc/LLMEngine] load begin model_dir=%s\n", model_dir.c_str());
    device_cfg_ = resolve_device_config(device);

    if (device_cfg_.requested == ComputeDevice::kAuto) {
        std::fprintf(stderr, "[worker-pc/LLMEngine] auto device resolved to %s\n",
                     compute_device_name(device_cfg_.resolved));
    } else if (device_cfg_.used_fallback) {
        std::fprintf(stderr, "[worker-pc/LLMEngine] requested device=%s unavailable, fallback to %s\n",
                     compute_device_name(device_cfg_.requested),
                     compute_device_name(device_cfg_.resolved));
    }

    const bool ok = model_->load(model_dir, device_cfg_.resolved);
    if (!ok) {
        model_->destroy();
        std::fprintf(stderr, "[worker-pc/LLMEngine] load failed elapsed_ms=%.2f\n",
                     (now_us() - t0) / 1e3f);
    } else {
        std::fprintf(stderr, "[worker-pc/LLMEngine] load done elapsed_ms=%.2f\n",
                     (now_us() - t0) / 1e3f);
    }
    return ok;
}

void LLMEngine::destroy() {
    if (model_) {
        model_->destroy();
    }
}

void LLMEngine::reset() {
    model_->reset_kv_cache();
}

GenerationResult LLMEngine::generate(
    const std::vector<int>& input_ids,
    const GenerationConfig& cfg,
    TokenCallback on_token) {
    std::fprintf(stderr,
                 "[worker-pc/LLMEngine] generate begin input_tokens=%d max_new_tokens=%d\n",
                 static_cast<int>(input_ids.size()), cfg.max_new_tokens);
    GenerationResult result;
    if (input_ids.empty()) {
        return result;
    }

    auto is_stop = [&](int id) {
        return std::find(cfg.stop_tokens.begin(), cfg.stop_tokens.end(), id) != cfg.stop_tokens.end();
    };

    try {
        const int64_t t0 = now_us();
        int next_id = model_->forward_next_token(input_ids);
        result.prefill_ms = (now_us() - t0) / 1e3f;
        result.prefill_tokens = static_cast<int>(input_ids.size());

        std::vector<int> one_token(1);
        const int64_t decode_begin = now_us();
        for (int step = 0; step < cfg.max_new_tokens; ++step) {
            if (is_stop(next_id)) {
                result.hit_stop = true;
                break;
            }

            const int64_t step_begin = now_us();
            const int emit_id = next_id;
            result.output_ids.push_back(emit_id);

            if (cfg.repetition_window > 0 &&
                detect_repetition(result.output_ids, cfg.repetition_window)) {
                std::fprintf(stderr, "[worker-pc/LLMEngine] detect repetition, stop at step=%d\n", step);
                result.hit_repetition = true;
                break;
            }

            if (step + 1 >= cfg.max_new_tokens) {
                break;
            }

            one_token[0] = emit_id;
            next_id = model_->forward_next_token(one_token);
            const float elapsed_ms = (now_us() - step_begin) / 1e3f;
            if (on_token) {
                on_token(step, emit_id, elapsed_ms);
            }
        }

        result.decode_ms = (now_us() - decode_begin) / 1e3f;
        result.decode_tokens = static_cast<int>(result.output_ids.size());
        std::fprintf(stderr,
                     "[worker-pc/LLMEngine] generate done prefill_ms=%.2f decode_ms=%.2f output_tokens=%d\n",
                     result.prefill_ms, result.decode_ms, result.decode_tokens);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[worker-pc/LLMEngine] generate failed: %s\n", e.what());
        model_->reset_kv_cache();
    }

    return result;
}
