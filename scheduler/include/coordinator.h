#pragma once

#include "scheduler_types.h"
#include "session_manager.h"
#include "tokenizer/tokenizer_adapter.h"
#include "worker_interface.h"

#include <memory>
#include <string>
#include <vector>

class Coordinator {
public:
    Coordinator(std::unique_ptr<WorkerInterface> local_worker,
                std::unique_ptr<TokenizerAdapter> tokenizer);
    ~Coordinator();

    SessionId create_session(const std::string& system_prompt = "");
    bool session_exists(const SessionId& session_id) const;

    bool register_worker(std::unique_ptr<WorkerInterface> worker);
    GenerationResult submit_text_request(const SessionId& session_id,
                                         const std::string& user_text,
                                         int max_new_tokens,
                                         TokenCallback on_token = nullptr);
    GenerationResult submit_token_request(const SessionId& session_id,
                                          const std::vector<int>& prompt_ids,
                                          int max_new_tokens,
                                          TokenCallback on_token = nullptr);

private:
    GenerateTokensRequest build_generate_request(const SessionId& session_id,
                                                 const std::vector<int>& prompt_ids,
                                                 int max_new_tokens) const;

    std::unique_ptr<WorkerInterface> primary_worker_;
    std::vector<std::unique_ptr<WorkerInterface>> remote_workers_;
    std::unique_ptr<TokenizerAdapter> tokenizer_;
    SessionManager session_manager_;
    RequestId next_request_id_ = 1;
};
