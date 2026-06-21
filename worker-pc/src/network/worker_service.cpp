#include "network/worker_service.h"

#include <api/generation_config.h>
#include <ops/op_cast.h>

#include <chrono>
#include <cstring>
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

const char* mode_to_cstr(WorkerService::RuntimeMode mode) {
    switch (mode) {
    case WorkerService::RuntimeMode::kFullModel: return "full";
    case WorkerService::RuntimeMode::kHead: return "head";
    case WorkerService::RuntimeMode::kStage: return "stage";
    case WorkerService::RuntimeMode::kTail: return "tail";
    default: return "unknown";
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
    return register_service(address, port, model_dir, device, ServiceConfig{});
}

bool WorkerService::register_service(const std::string& address,
                                     int port,
                                     const std::string& model_dir,
                                     ComputeDevice device,
                                     const ServiceConfig& service_cfg) {
    device_ = device;
    service_cfg_ = service_cfg;
    LLMEngine::PartitionConfig partition;
    partition.layer_begin = service_cfg_.layer_begin;
    partition.layer_end = service_cfg_.layer_end;
    partition.include_embedding =
        (service_cfg_.mode == RuntimeMode::kFullModel || service_cfg_.mode == RuntimeMode::kHead);
    partition.include_final_norm_and_head =
        (service_cfg_.mode == RuntimeMode::kFullModel || service_cfg_.mode == RuntimeMode::kTail);

    if (!engine_.load(model_dir, device_, partition)) {
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

    if (service_cfg_.mode != RuntimeMode::kFullModel || !engine_.supports_token_generation()) {
        response.status = distributed::StatusCode::kUnsupported;
        response.message = "worker-pc instance is not running in full-model mode";
        return response;
    }

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

distributed::TokensToHiddenResponse WorkerService::handle_tokens_to_hidden(
    const distributed::TokensToHiddenRequest& request) {
    distributed::TokensToHiddenResponse response;
    response.context = request.context;

    if (service_cfg_.mode != RuntimeMode::kHead || !engine_.supports_tokens_to_hidden()) {
        response.status = distributed::StatusCode::kUnsupported;
        response.message = "worker-pc instance is not running in head mode";
        return response;
    }
    if (request.input_token_ids.empty()) {
        response.status = distributed::StatusCode::kError;
        response.message = "input_token_ids is empty";
        return response;
    }

    std::vector<uint16_t> output_f16;
    std::string error;
    if (!engine_.forward_tokens_to_hidden(
            std::vector<int>(request.input_token_ids.begin(), request.input_token_ids.end()),
            output_f16,
            &error)) {
        response.status = distributed::StatusCode::kError;
        response.message = error.empty() ? "tokens_to_hidden failed" : error;
        return response;
    }

    response.status = distributed::StatusCode::kOk;
    response.output_tensor.dtype = distributed::TensorDataType::kFloat16;
    response.output_tensor.shape = {
        static_cast<int32_t>(request.input_token_ids.size()),
        static_cast<int32_t>(engine_.hidden_size())
    };
    response.output_tensor.bytes.resize(output_f16.size() * sizeof(uint16_t));
    std::memcpy(response.output_tensor.bytes.data(), output_f16.data(), response.output_tensor.bytes.size());
    return response;
}

distributed::StageForwardResponse WorkerService::handle_stage_forward(
    const distributed::StageForwardRequest& request) {
    distributed::StageForwardResponse response;
    response.context = request.context;
    if (service_cfg_.mode != RuntimeMode::kStage || !engine_.supports_stage_forward()) {
        response.status = distributed::StatusCode::kUnsupported;
        response.message = "worker-pc instance is not running in stage mode";
        return response;
    }
    if (request.input_tensor.shape.size() != 2) {
        response.status = distributed::StatusCode::kError;
        response.message = "stage input tensor must have shape [seq, hidden]";
        return response;
    }
    if (request.input_tensor.shape[1] <= 0 || request.input_tensor.shape[0] <= 0) {
        response.status = distributed::StatusCode::kError;
        response.message = "invalid stage input tensor shape";
        return response;
    }

    const int seq = request.input_tensor.shape[0];
    const int hidden = request.input_tensor.shape[1];
    std::vector<uint16_t> input_f16;
    if (request.input_tensor.dtype == distributed::TensorDataType::kFloat16) {
        if (request.input_tensor.bytes.size() != static_cast<size_t>(seq) * hidden * sizeof(uint16_t)) {
            response.status = distributed::StatusCode::kError;
            response.message = "stage input bytes size does not match FP16 tensor shape";
            return response;
        }
        input_f16.resize(static_cast<size_t>(seq) * hidden);
        std::memcpy(input_f16.data(), request.input_tensor.bytes.data(), request.input_tensor.bytes.size());
    } else if (request.input_tensor.dtype == distributed::TensorDataType::kFloat32) {
        if (request.input_tensor.bytes.size() != static_cast<size_t>(seq) * hidden * sizeof(float)) {
            response.status = distributed::StatusCode::kError;
            response.message = "stage input bytes size does not match FP32 tensor shape";
            return response;
        }
        const float* src = reinterpret_cast<const float*>(request.input_tensor.bytes.data());
        input_f16.resize(static_cast<size_t>(seq) * hidden);
        op_f32_to_f16(src, input_f16.data(), seq * hidden);
    } else {
        response.status = distributed::StatusCode::kError;
        response.message = "stage input tensor dtype must be FP16 or FP32";
        return response;
    }

    std::vector<uint16_t> output_f16;
    std::string error;
    if (!engine_.forward_hidden_states(input_f16,
                                       seq,
                                       request.context.route.pos_base,
                                       output_f16,
                                       &error)) {
        response.status = distributed::StatusCode::kError;
        response.message = error.empty() ? "stage forward failed" : error;
        return response;
    }

    response.status = distributed::StatusCode::kOk;
    response.output_tensor.dtype = distributed::TensorDataType::kFloat16;
    response.output_tensor.shape = request.input_tensor.shape;
    response.output_tensor.bytes.resize(output_f16.size() * sizeof(uint16_t));
    std::memcpy(response.output_tensor.bytes.data(),
                output_f16.data(),
                response.output_tensor.bytes.size());
    return response;
}

distributed::HiddenToTokenResponse WorkerService::handle_hidden_to_token(
    const distributed::HiddenToTokenRequest& request) {
    distributed::HiddenToTokenResponse response;
    response.context = request.context;
    if (service_cfg_.mode != RuntimeMode::kTail || !engine_.supports_hidden_to_token()) {
        response.status = distributed::StatusCode::kUnsupported;
        response.message = "worker-pc instance is not running in tail mode";
        return response;
    }
    if (request.input_tensor.shape.size() != 2) {
        response.status = distributed::StatusCode::kError;
        response.message = "tail input tensor must have shape [seq, hidden]";
        return response;
    }
    const int seq = request.input_tensor.shape[0];
    const int hidden = request.input_tensor.shape[1];
    if (seq <= 0 || hidden <= 0) {
        response.status = distributed::StatusCode::kError;
        response.message = "invalid tail input tensor shape";
        return response;
    }

    std::vector<uint16_t> input_f16;
    if (request.input_tensor.dtype == distributed::TensorDataType::kFloat16) {
        if (request.input_tensor.bytes.size() != static_cast<size_t>(seq) * hidden * sizeof(uint16_t)) {
            response.status = distributed::StatusCode::kError;
            response.message = "tail input bytes size does not match FP16 tensor shape";
            return response;
        }
        input_f16.resize(static_cast<size_t>(seq) * hidden);
        std::memcpy(input_f16.data(), request.input_tensor.bytes.data(), request.input_tensor.bytes.size());
    } else if (request.input_tensor.dtype == distributed::TensorDataType::kFloat32) {
        if (request.input_tensor.bytes.size() != static_cast<size_t>(seq) * hidden * sizeof(float)) {
            response.status = distributed::StatusCode::kError;
            response.message = "tail input bytes size does not match FP32 tensor shape";
            return response;
        }
        input_f16.resize(static_cast<size_t>(seq) * hidden);
        op_f32_to_f16(reinterpret_cast<const float*>(request.input_tensor.bytes.data()),
                      input_f16.data(),
                      seq * hidden);
    } else {
        response.status = distributed::StatusCode::kError;
        response.message = "tail input tensor dtype must be FP16 or FP32";
        return response;
    }

    int output_token_id = 0;
    std::string error;
    if (!engine_.forward_hidden_to_token(input_f16,
                                         seq,
                                         request.context.route.pos_base,
                                         output_token_id,
                                         &error)) {
        response.status = distributed::StatusCode::kError;
        response.message = error.empty() ? "hidden_to_token failed" : error;
        return response;
    }

    response.status = distributed::StatusCode::kOk;
    response.output_token_id = output_token_id;
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
                 "device=%s mode=%s layers=[%d,%d) status=%s elapsed_ms=%.2f %s\n",
                 client.ip.empty() ? "unknown" : client.ip.c_str(),
                 client.port,
                 command_name,
                 context.session_id.empty() ? "-" : context.session_id.c_str(),
                 static_cast<unsigned long long>(context.request_id),
                 compute_device_name(device_),
                 mode_to_cstr(service_cfg_.mode),
                 service_cfg_.layer_begin,
                 service_cfg_.layer_end,
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

    if (distributed::is_reset_cache_payload(request)) {
        distributed::RequestContext context;
        engine_.reset();
        log_request_summary(client, "RESET_CACHE", context, distributed::StatusCode::kOk,
                            std::chrono::duration<double, std::milli>(Clock::now() - start_time).count(),
                            "message=cache-reset");
        return "OK";
    }

    distributed::GenerateTokensRequest generate_request;
    if (distributed::deserialize_generate_request(request, generate_request)) {
        const auto response = handle_generate_tokens(generate_request);
        std::ostringstream extra;
        const float decode_tok_s =
            response.decode_ms > 0.0f
                ? response.decode_tokens / (response.decode_ms / 1000.0f)
                : 0.0f;
        extra << "input_tokens=" << generate_request.input_token_ids.size()
              << " output_tokens=" << response.output_token_ids.size()
              << " prefill_ms=" << response.prefill_ms
              << " decode_ms=" << response.decode_ms
              << " decode_tok_s=" << decode_tok_s;
        log_request_summary(client, "GENERATE_TOKENS", response.context, response.status,
                            std::chrono::duration<double, std::milli>(Clock::now() - start_time).count(),
                            extra.str());
        return distributed::serialize_generate_response(response);
    }

    distributed::TokensToHiddenRequest head_request;
    if (distributed::deserialize_tokens_to_hidden_request(request, head_request)) {
        const auto response = handle_tokens_to_hidden(head_request);
        std::ostringstream extra;
        extra << "input_tokens=" << head_request.input_token_ids.size()
              << " output_bytes=" << response.output_tensor.bytes.size();
        log_request_summary(client, "TOKENS_TO_HIDDEN", response.context, response.status,
                            std::chrono::duration<double, std::milli>(Clock::now() - start_time).count(),
                            response.status == distributed::StatusCode::kOk
                                ? extra.str()
                                : response.message);
        return distributed::serialize_tokens_to_hidden_response(response);
    }

    distributed::StageForwardRequest stage_request;
    if (distributed::deserialize_stage_request(request, stage_request)) {
        const auto response = handle_stage_forward(stage_request);
        std::ostringstream extra;
        extra << "input_shape=[";
        for (size_t i = 0; i < stage_request.input_tensor.shape.size(); ++i) {
            if (i != 0) extra << ",";
            extra << stage_request.input_tensor.shape[i];
        }
        extra << "] output_bytes=" << response.output_tensor.bytes.size();
        log_request_summary(client, "FORWARD_STAGE", response.context, response.status,
                            std::chrono::duration<double, std::milli>(Clock::now() - start_time).count(),
                            response.status == distributed::StatusCode::kOk
                                ? extra.str()
                                : response.message);
        return distributed::serialize_stage_response(response);
    }

    distributed::HiddenToTokenRequest tail_request;
    if (distributed::deserialize_hidden_to_token_request(request, tail_request)) {
        const auto response = handle_hidden_to_token(tail_request);
        std::ostringstream extra;
        extra << "output_token_id=" << response.output_token_id;
        log_request_summary(client, "HIDDEN_TO_TOKEN", response.context, response.status,
                            std::chrono::duration<double, std::milli>(Clock::now() - start_time).count(),
                            response.status == distributed::StatusCode::kOk
                                ? extra.str()
                                : response.message);
        return distributed::serialize_hidden_to_token_response(response);
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
