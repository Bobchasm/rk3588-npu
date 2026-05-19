#pragma once

#include <string>

#include <api/llm_engine.h>
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
    WorkerRpcServer rpc_server_;
    LLMEngine engine_;
    bool loaded_;
};
