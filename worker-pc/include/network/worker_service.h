#pragma once

#include "api/llm_engine.h"
#include "network/rpc_server.h"

#include <distributed/protocol.h>

#include <string>

class WorkerService {
public:
    WorkerService();
    ~WorkerService();

    bool register_service(const std::string& address,
                          int port,
                          const std::string& model_dir,
                          ComputeDevice device);

    std::string handle_request(const std::string& request,
                               const ClientEndpoint& client);

private:
    distributed::GenerateTokensResponse handle_generate_tokens(
        const distributed::GenerateTokensRequest& request);
    distributed::StageForwardResponse handle_stage_forward(
        const distributed::StageForwardRequest& request);

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
};

