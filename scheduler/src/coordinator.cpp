#include "coordinator.h"

#include <iostream>

namespace {

GenerationResult make_error_result(const std::string& message) {
    GenerationResult result;
    result.error_message = message;
    return result;
}

}  // namespace

Coordinator::Coordinator(std::unique_ptr<WorkerInterface> local_worker,
                         std::unique_ptr<TokenizerAdapter> tokenizer)
    : primary_worker_(std::move(local_worker)),
      tokenizer_(std::move(tokenizer))
{
}

Coordinator::~Coordinator() = default;

SessionId Coordinator::create_session(const std::string& system_prompt) {
    return session_manager_.create_session(system_prompt);
}

bool Coordinator::session_exists(const SessionId& session_id) const {
    return session_manager_.session_exists(session_id);
}

bool Coordinator::register_worker(std::unique_ptr<WorkerInterface> worker) {
    if (!worker) return false;
    if (worker->worker_id().empty()) {
        std::cerr << "[Coordinator] worker has no id" << std::endl;
        return false;
    }
    remote_workers_.push_back(std::move(worker));
    return true;
}

GenerateTokensRequest Coordinator::build_generate_request(const SessionId& session_id,
                                                          const std::vector<int>& prompt_ids,
                                                          int max_new_tokens) const
{
    GenerateTokensRequest request;
    request.context.session_id = session_id;
    request.context.request_id = next_request_id_;
    request.context.trace_id = session_id + "-" + std::to_string(next_request_id_);
    request.context.priority = 0;
    request.context.timeout_ms = 0;
    request.context.route.stage_id = 0;
    request.context.route.total_stages = 1;
    request.context.route.hop_index = 0;
    request.context.route.pos_base = 0;
    request.context.route.target_worker_id = primary_worker_ ? primary_worker_->worker_id() : "";
    request.generation.max_new_tokens = max_new_tokens;
    request.input_token_ids.assign(prompt_ids.begin(), prompt_ids.end());
    return request;
}

GenerationResult Coordinator::submit_text_request(const SessionId& session_id,
                                                  const std::string& user_text,
                                                  int max_new_tokens,
                                                  TokenCallback on_token)
{
    if (!session_exists(session_id)) {
        return make_error_result("session does not exist");
    }
    if (!primary_worker_) {
        return make_error_result("no worker available");
    }
    if (!tokenizer_) {
        return make_error_result("tokenizer not configured");
    }
    if (!session_manager_.append_user_message(session_id, user_text)) {
        return make_error_result("failed to append user message");
    }

    const std::string prompt = session_manager_.get_prompt(session_id);
    std::vector<int> prompt_ids;
    if (!tokenizer_->encode(prompt, prompt_ids)) {
        return make_error_result("failed to tokenize prompt");
    }

    GenerationResult result = submit_token_request(session_id, prompt_ids, max_new_tokens, on_token);
    if (!result.error_message.empty()) {
        return result;
    }

    if (!session_manager_.append_assistant_message(session_id, result.output_text)) {
        result.error_message = "failed to append assistant message";
    }
    return result;
}

GenerationResult Coordinator::submit_token_request(const SessionId& session_id,
                                                   const std::vector<int>& prompt_ids,
                                                   int max_new_tokens,
                                                   TokenCallback on_token)
{
    if (!session_exists(session_id)) {
        return make_error_result("session does not exist");
    }
    if (!primary_worker_) {
        return make_error_result("no worker available");
    }

    GenerateTokensRequest request = build_generate_request(session_id, prompt_ids, max_new_tokens);
    ++next_request_id_;

    GenerationResult result = primary_worker_->generate_tokens(request, on_token);
    result.prompt_ids = prompt_ids;

    if (result.error_message.empty() && tokenizer_) {
        if (!tokenizer_->decode(result.output_ids, result.output_text)) {
            result.error_message = "failed to decode output tokens";
        }
    }
    return result;
}
