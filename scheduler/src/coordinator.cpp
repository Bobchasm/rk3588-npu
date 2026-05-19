#include "coordinator.h"
#include <iostream>

Coordinator::Coordinator(std::unique_ptr<WorkerInterface> local_worker)
    : primary_worker_(std::move(local_worker))
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

GenerationResult Coordinator::submit_request(
    const SessionId& session_id,
    const std::vector<int>& input_ids,
    int max_new_tokens,
    TokenCallback on_token)
{
    GenerationResult result;
    if (!session_exists(session_id)) {
        std::cerr << "[Coordinator] session does not exist: " << session_id << std::endl;
        return result;
    }
    if (!primary_worker_) {
        std::cerr << "[Coordinator] no worker available" << std::endl;
        return result;
    }
    RequestId request_id = next_request_id_++;
    (void)request_id;
    return primary_worker_->generate_local(session_id, input_ids, max_new_tokens, on_token);
}
