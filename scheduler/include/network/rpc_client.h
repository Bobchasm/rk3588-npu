#pragma once

#include "scheduler_types.h"
#include <string>

class RpcClient {
public:
    RpcClient();
    ~RpcClient();

    bool connect(const std::string& endpoint);
    bool send_ping();
    bool send_reset_cache();

    bool send_generate_tokens(const GenerateTokensRequest& req, GenerateTokensResponse& resp);
    bool send_tokens_to_hidden(const distributed::TokensToHiddenRequest& req,
                               distributed::TokensToHiddenResponse& resp);
    bool send_forward_stage(const StageForwardRequest& req, StageForwardResponse& resp);
    bool send_hidden_to_token(const distributed::HiddenToTokenRequest& req,
                              distributed::HiddenToTokenResponse& resp);

private:
    std::string endpoint_;
};
