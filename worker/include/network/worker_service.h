#pragma once

#include <map>
#include <string>
#include <vector>

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
    std::string handle_request(const std::string& request,
                               const ClientEndpoint& client);

private:
    struct SessionCacheEntry {
        LLMEngine::KvState kv_state;
        std::vector<int> cached_prompt_ids;
    };

    distributed::GenerateTokensResponse handle_generate_tokens(
        const distributed::GenerateTokensRequest& request);
    distributed::StageForwardResponse handle_stage_forward(
        const distributed::StageForwardRequest& request);

    // 将请求上下文、客户端来源、耗时和状态统一封装成一条日志，避免散落在多个分支里。
    void log_request_summary(const ClientEndpoint& client,
                             const char* command_name,
                             const distributed::RequestContext& context,
                             distributed::StatusCode status,
                             double elapsed_ms,
                             const std::string& extra) const;

    WorkerRpcServer rpc_server_;
    LLMEngine engine_;
    bool loaded_;
    std::string active_session_id_;
    std::vector<int> cached_prompt_ids_;
    std::map<std::string, SessionCacheEntry> session_caches_;
};
