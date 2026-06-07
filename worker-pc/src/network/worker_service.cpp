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
    case distributed::StatusCode::kOk: return "OK";
    case distributed::StatusCode::kError: return "ERROR";
    case distributed::StatusCode::kUnsupported: return "UNSUPPORTED";
    default: return "UNKNOWN";
    }
}

}  // namespace

WorkerService::WorkerService() = default;
WorkerService::~WorkerService() {
    rpc_server_.stop();
}

bool WorkerService::register_service(const std::string& address,
                                     int port,
                                     const std::string& model_dir,
                                     ComputeDevice device) {
    device_ = device;
    if (!engine_.load(model_dir, device_)) {
        std::cerr << "[worker-pc/WorkerService] failed to load model: " << model_dir << std::endl;
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

    engine_.reset();
    const GenerationConfig cfg = make_generation_config(request.generation);
    const auto generation = engine_.generate(
        std::vector<int>(request.input_token_ids.begin(), request.input_token_ids.end()),
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
    return response;
}

distributed::StageForwardResponse WorkerService::handle_stage_forward(
    const distributed::StageForwardRequest& request) {
    distributed::StageForwardResponse response;
    response.context = request.context;
    response.status = distributed::StatusCode::kUnsupported;
    response.message =
        "worker-pc currently supports full-model token generation only; stage forwarding is reserved for future block-partition execution";
    return response;
}

void WorkerService::log_request_summary(const ClientEndpoint& client,
                                        const char* command_name,
                                        const distributed::RequestContext& context,
                                        distributed::StatusCode status,
                                        double elapsed_ms,
                                        const std::string& extra) const {
    std::fprintf(stderr,
                 "[worker-pc/WorkerService] client=%s:%d command=%s session=%s request=%llu "
                 "device=%s status=%s elapsed_ms=%.2f %s\n",
                 client.ip.empty() ? "unknown" : client.ip.c_str(),
                 client.port,
                 command_name,
                 context.session_id.empty() ? "-" : context.session_id.c_str(),
                 static_cast<unsigned long long>(context.request_id),
                 compute_device_name(device_),
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
        log_request_summary(client, "GENERATE_TOKENS", context, distributed::StatusCode::kError,
                            std::chrono::duration<double, std::milli>(Clock::now() - start_time).count(),
                            "message=model-not-loaded");
        return response;
    }

    if (distributed::is_ping_payload(request)) {
        distributed::RequestContext context;
        log_request_summary(client, "PING", context, distributed::StatusCode::kOk,
                            std::chrono::duration<double, std::milli>(Clock::now() - start_time).count(),
                            "message=pong");
        return "PONG";
    }

    distributed::GenerateTokensRequest generate_request;
    if (distributed::deserialize_generate_request(request, generate_request)) {
        const auto response = handle_generate_tokens(generate_request);
        std::ostringstream extra;
        extra << "input_tokens=" << generate_request.input_token_ids.size()
              << " output_tokens=" << response.output_token_ids.size()
              << " prefill_ms=" << response.prefill_ms
              << " decode_ms=" << response.decode_ms;
        log_request_summary(client, "GENERATE_TOKENS", response.context, response.status,
                            std::chrono::duration<double, std::milli>(Clock::now() - start_time).count(),
                            extra.str());
        return distributed::serialize_generate_response(response);
    }

    distributed::StageForwardRequest stage_request;
    if (distributed::deserialize_stage_request(request, stage_request)) {
        const auto response = handle_stage_forward(stage_request);
        log_request_summary(client, "FORWARD_STAGE", response.context, response.status,
                            std::chrono::duration<double, std::milli>(Clock::now() - start_time).count(),
                            response.message);
        return distributed::serialize_stage_response(response);
    }

    distributed::RequestContext context;
    const std::string response = distributed::serialize_error_response(
        distributed::RpcCommand::kGenerateTokens,
        context,
        distributed::StatusCode::kError,
        "invalid-or-unsupported-request");
    log_request_summary(client, "UNKNOWN", context, distributed::StatusCode::kError,
                        std::chrono::duration<double, std::milli>(Clock::now() - start_time).count(),
                        "message=invalid-or-unsupported-request");
    return response;
}

