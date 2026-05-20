#include "network/worker_service.h"

#include <api/generation_config.h>

#include <iostream>
#include <utility>
#include <vector>

namespace {

GenerationConfig make_generation_config(const distributed::GenerationParameters& params) {
    GenerationConfig cfg;
    cfg.max_new_tokens = params.max_new_tokens;
    cfg.repetition_window = params.repetition_window;
    return cfg;
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
    return rpc_server_.start(address, port, [this](const std::string& request) {
        return handle_request(request);
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
        "current worker only supports single-node full-model generation; "
        "stage forwarding is reserved for future multi-worker pipeline";
    return response;
}

std::string WorkerService::handle_request(const std::string& request) {
    if (!loaded_) {
        distributed::RequestContext context;
        return distributed::serialize_error_response(
            distributed::RpcCommand::kGenerateTokens,
            context,
            distributed::StatusCode::kError,
            "model-not-loaded");
    }

    if (distributed::is_ping_payload(request)) {
        return "PONG";
    }

    distributed::GenerateTokensRequest generate_request;
    if (distributed::deserialize_generate_request(request, generate_request)) {
        return distributed::serialize_generate_response(handle_generate_tokens(generate_request));
    }

    distributed::StageForwardRequest stage_request;
    if (distributed::deserialize_stage_request(request, stage_request)) {
        return distributed::serialize_stage_response(handle_stage_forward(stage_request));
    }

    distributed::RequestContext context;
    return distributed::serialize_error_response(
        distributed::RpcCommand::kGenerateTokens,
        context,
        distributed::StatusCode::kError,
        "invalid-or-unsupported-request");
}
