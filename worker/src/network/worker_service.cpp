#include "network/worker_service.h"

#include <api/generation_config.h>

#include <chrono>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <utility>
#include <vector>

namespace {

GenerationConfig make_generation_config(const distributed::GenerationParameters& params) {
    GenerationConfig cfg;
    cfg.max_new_tokens = params.max_new_tokens;
    cfg.repetition_window = params.repetition_window;
    return cfg;
}

const char* status_to_cstr(distributed::StatusCode status) {
    switch (status) {
    case distributed::StatusCode::kOk:
        return "OK";
    case distributed::StatusCode::kError:
        return "ERROR";
    case distributed::StatusCode::kUnsupported:
        return "UNSUPPORTED";
    default:
        return "UNKNOWN";
    }
}

std::string make_request_brief(const distributed::GenerateTokensRequest& request) {
    std::ostringstream oss;
    oss << "input_tokens=" << request.input_token_ids.size()
        << " max_new_tokens=" << request.generation.max_new_tokens
        << " repetition_window=" << request.generation.repetition_window;
    return oss.str();
}

std::string make_stage_brief(const distributed::StageForwardRequest& request) {
    std::ostringstream oss;
    oss << "dtype=" << static_cast<int>(request.input_tensor.dtype)
        << " shape=[";
    for (size_t i = 0; i < request.input_tensor.shape.size(); ++i) {
        if (i != 0) {
            oss << ",";
        }
        oss << request.input_tensor.shape[i];
    }
    oss << "] bytes=" << request.input_tensor.bytes.size();
    return oss.str();
}

size_t common_prefix_len(const std::vector<int32_t>& lhs, const std::vector<int>& rhs) {
    const size_t limit = std::min(lhs.size(), rhs.size());
    size_t index = 0;
    while (index < limit && lhs[index] == rhs[index]) {
        ++index;
    }
    return index;
}

}  // namespace

WorkerService::WorkerService()
    : loaded_(false) {}

WorkerService::~WorkerService() {
    rpc_server_.stop();
}

bool WorkerService::register_service(const std::string& address,
                                     int port,
                                     const std::string& model_dir) {
    if (!engine_.load(model_dir)) {
        std::cerr << "[WorkerService] failed to load model: " << model_dir << std::endl;
        return false;
    }
    loaded_ = true;
    return rpc_server_.start(address, port, [this](const std::string& request, const ClientEndpoint& client) {
        return handle_request(request, client);
    });
}

distributed::GenerateTokensResponse WorkerService::handle_generate_tokens(
    const distributed::GenerateTokensRequest& request) {
    distributed::GenerateTokensResponse response;
    response.context = request.context;

    if (request.input_token_ids.empty()) {
        response.status = distributed::StatusCode::kError;
        response.message = "input_token_ids is empty";
        return response;
    }

    std::vector<int> effective_input_ids(request.input_token_ids.begin(), request.input_token_ids.end());
    bool cache_reused = false;
    int32_t reused_tokens = 0;
    if (!request.context.session_id.empty()) {
        const auto cached_it = session_caches_.find(request.context.session_id);
        if (active_session_id_ != request.context.session_id) {
            if (cached_it != session_caches_.end() &&
                engine_.restore_kv_state(cached_it->second.kv_state)) {
                active_session_id_ = request.context.session_id;
                cached_prompt_ids_ = cached_it->second.cached_prompt_ids;
            } else {
                engine_.reset();
                active_session_id_ = request.context.session_id;
                cached_prompt_ids_.clear();
            }
        }

        const size_t prefix_len = common_prefix_len(request.input_token_ids, cached_prompt_ids_);
        if (prefix_len == cached_prompt_ids_.size() &&
            prefix_len < request.input_token_ids.size()) {
            effective_input_ids.assign(request.input_token_ids.begin() + static_cast<std::ptrdiff_t>(prefix_len),
                                       request.input_token_ids.end());
            cache_reused = true;
            reused_tokens = static_cast<int32_t>(prefix_len);
        } else if (prefix_len != cached_prompt_ids_.size()) {
            engine_.reset();
            cached_prompt_ids_.clear();
            effective_input_ids.assign(request.input_token_ids.begin(), request.input_token_ids.end());
        }
    } else {
        engine_.reset();
        active_session_id_.clear();
        cached_prompt_ids_.clear();
    }

    const GenerationConfig cfg = make_generation_config(request.generation);
    const auto generation = engine_.generate(
        effective_input_ids,
        cfg,
        nullptr);

    response.status = distributed::StatusCode::kOk;
    response.output_token_ids.assign(generation.output_ids.begin(), generation.output_ids.end());
    response.prefill_tokens = generation.prefill_tokens;
    response.decode_tokens = generation.decode_tokens;
    response.prefill_ms = generation.prefill_ms;
    response.decode_ms = generation.decode_ms;
    response.hit_stop = generation.hit_stop;
    response.hit_repetition = generation.hit_repetition;
    if (!request.context.session_id.empty()) {
        cached_prompt_ids_.assign(request.input_token_ids.begin(), request.input_token_ids.end());
        cached_prompt_ids_.insert(cached_prompt_ids_.end(),
                                  generation.output_ids.begin(),
                                  generation.output_ids.end());
        SessionCacheEntry entry;
        entry.kv_state = engine_.snapshot_kv_state();
        entry.cached_prompt_ids = cached_prompt_ids_;
        session_caches_[request.context.session_id] = std::move(entry);
        response.message =
            std::string("cache_reused=") + (cache_reused ? "1" : "0") +
            " reused_tokens=" + std::to_string(reused_tokens) +
            " effective_prefill_tokens=" + std::to_string(effective_input_ids.size());
    }
    return response;
}

distributed::StageForwardResponse WorkerService::handle_stage_forward(
    const distributed::StageForwardRequest& request) {
    distributed::StageForwardResponse response;
    response.context = request.context;
    response.status = distributed::StatusCode::kUnsupported;
    response.message =
        "current worker only supports single-node full-model generation; "
        "stage forwarding is reserved for future multi-worker pipeline";
    return response;
}

void WorkerService::log_request_summary(const ClientEndpoint& client,
                                        const char* command_name,
                                        const distributed::RequestContext& context,
                                        distributed::StatusCode status,
                                        double elapsed_ms,
                                        const std::string& extra) const {
    // 日志字段保持扁平，便于后续直接 grep 或接入更正式的日志采集。
    std::fprintf(stderr,
                 "[WorkerService] client=%s:%d command=%s session=%s request=%llu "
                 "trace=%s stage=%d hop=%d status=%s elapsed_ms=%.2f %s\n",
                 client.ip.empty() ? "unknown" : client.ip.c_str(),
                 client.port,
                 command_name,
                 context.session_id.empty() ? "-" : context.session_id.c_str(),
                 static_cast<unsigned long long>(context.request_id),
                 context.trace_id.empty() ? "-" : context.trace_id.c_str(),
                 context.route.stage_id,
                 context.route.hop_index,
                 status_to_cstr(status),
                 elapsed_ms,
                 extra.empty() ? "" : extra.c_str());
}

std::string WorkerService::handle_request(const std::string& request,
                                          const ClientEndpoint& client) {
    using Clock = std::chrono::steady_clock;
    const auto start_time = Clock::now();

    if (!loaded_) {
        distributed::RequestContext context;
        const std::string response = distributed::serialize_error_response(
            distributed::RpcCommand::kGenerateTokens,
            context,
            distributed::StatusCode::kError,
            "model-not-loaded");
        log_request_summary(client,
                            "GENERATE_TOKENS",
                            context,
                            distributed::StatusCode::kError,
                            std::chrono::duration<double, std::milli>(Clock::now() - start_time).count(),
                            "message=model-not-loaded");
        return response;
    }

    if (distributed::is_ping_payload(request)) {
        distributed::RequestContext context;
        log_request_summary(client,
                            "PING",
                            context,
                            distributed::StatusCode::kOk,
                            std::chrono::duration<double, std::milli>(Clock::now() - start_time).count(),
                            "message=pong");
        return "PONG";
    }

    distributed::GenerateTokensRequest generate_request;
    if (distributed::deserialize_generate_request(request, generate_request)) {
        const auto response = handle_generate_tokens(generate_request);
        std::ostringstream extra;
        const float decode_tok_s =
            response.decode_ms > 0.0f
                ? response.decode_tokens / (response.decode_ms / 1000.0f)
                : 0.0f;
        extra << make_request_brief(generate_request)
              << " output_tokens=" << response.output_token_ids.size()
              << " prefill_ms=" << response.prefill_ms
              << " decode_ms=" << response.decode_ms
              << " decode_tok_s=" << decode_tok_s
              << " message=" << (response.message.empty() ? "-" : response.message);
        log_request_summary(client,
                            "GENERATE_TOKENS",
                            response.context,
                            response.status,
                            std::chrono::duration<double, std::milli>(Clock::now() - start_time).count(),
                            extra.str());
        return distributed::serialize_generate_response(response);
    }

    distributed::StageForwardRequest stage_request;
    if (distributed::deserialize_stage_request(request, stage_request)) {
        const auto response = handle_stage_forward(stage_request);
        std::ostringstream extra;
        extra << make_stage_brief(stage_request)
              << " output_bytes=" << response.output_tensor.bytes.size()
              << " message=" << (response.message.empty() ? "-" : response.message);
        log_request_summary(client,
                            "FORWARD_STAGE",
                            response.context,
                            response.status,
                            std::chrono::duration<double, std::milli>(Clock::now() - start_time).count(),
                            extra.str());
        return distributed::serialize_stage_response(response);
    }

    distributed::RequestContext context;
    const std::string response = distributed::serialize_error_response(
        distributed::RpcCommand::kGenerateTokens,
        context,
        distributed::StatusCode::kError,
        "invalid-or-unsupported-request");
    log_request_summary(client,
                        "UNKNOWN",
                        context,
                        distributed::StatusCode::kError,
                        std::chrono::duration<double, std::milli>(Clock::now() - start_time).count(),
                        "message=invalid-or-unsupported-request");
    return response;
}
