#pragma once

#include "api/llm_engine.h"
#include "network/rpc_server.h"

#include <distributed/protocol.h>

#include <string>

class WorkerService {
public:
    enum class RuntimeMode {
        kFullModel,
        kHead,
        kStage,
        kTail,
    };

    struct ServiceConfig {
        RuntimeMode mode = RuntimeMode::kFullModel;
        int layer_begin = 0;
        int layer_end = -1;
    };

    WorkerService();
    ~WorkerService();

    bool register_service(const std::string& address,
                          int port,
                          const std::string& model_dir,
                          ComputeDevice device);
    bool register_service(const std::string& address,
                          int port,
                          const std::string& model_dir,
                          ComputeDevice device,
                          const ServiceConfig& service_cfg);

    std::string handle_request(const std::string& request,
                               const ClientEndpoint& client);

private:
    distributed::GenerateTokensResponse handle_generate_tokens(
        const distributed::GenerateTokensRequest& request);
    distributed::TokensToHiddenResponse handle_tokens_to_hidden(
        const distributed::TokensToHiddenRequest& request);
    distributed::StageForwardResponse handle_stage_forward(
        const distributed::StageForwardRequest& request);
    distributed::HiddenToTokenResponse handle_hidden_to_token(
        const distributed::HiddenToTokenRequest& request);

    void log_request_summary(const ClientEndpoint& client,
                             const char* command_name,
                             const distributed::RequestContext& context,
                             distributed::StatusCode status,
                             double elapsed_ms,
                             const std::string& extra) const;

    WorkerRpcServer rpc_server_;
    LLMEngine engine_;
    ComputeDevice device_ = ComputeDevice::kCpu;
    bool loaded_ = false;
    ServiceConfig service_cfg_;
};
