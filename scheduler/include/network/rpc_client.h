#pragma once

#include "scheduler_types.h"
#include <string>

class RpcClient {
public:
    RpcClient();
    ~RpcClient();

    bool connect(const std::string& endpoint);
    bool send_ping();

    bool send_generate_tokens(const GenerateTokensRequest& req, GenerateTokensResponse& resp);
    bool send_forward_stage(const StageForwardRequest& req, StageForwardResponse& resp);

private:
    std::string endpoint_;
};
