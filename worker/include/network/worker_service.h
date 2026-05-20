#pragma once

#include <string>

#include <api/llm_engine.h>
#include <distributed/protocol.h>
#include "rpc_server.h"

class WorkerService {
public:
    WorkerService();
    ~WorkerService();

    bool register_service(const std::string& address,
                          int port,
                          const std::string& model_dir);
    std::string handle_request(const std::string& request);

private:
    distributed::GenerateTokensResponse handle_generate_tokens(
        const distributed::GenerateTokensRequest& request);
    distributed::StageForwardResponse handle_stage_forward(
        const distributed::StageForwardRequest& request);

    WorkerRpcServer rpc_server_;
    LLMEngine engine_;
    bool loaded_;
};
