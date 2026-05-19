#pragma once

#include "scheduler_types.h"
#include <string>

class RpcClient {
public:
    RpcClient();
    ~RpcClient();

    bool connect(const std::string& endpoint);
    bool send_ping();

    bool send_prefill(const PrefillRequest& req, PrefillResponse& resp);
    bool send_generate(const SessionId& session_id,
                       RequestId request_id,
                       const std::vector<int>& input_ids,
                       int max_new_tokens,
                       GenerationResult& result);

private:
    std::string endpoint_;
};
