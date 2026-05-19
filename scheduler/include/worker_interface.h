#pragma once

#include "scheduler_types.h"
#ifdef SCHEDULER_USE_WORKER_CORE
#include <api/llm_engine.h>
#endif
#include <memory>
#include <string>
#include <vector>

class WorkerInterface {
public:
    virtual ~WorkerInterface() = default;

    virtual bool register_node(const WorkerId& id) = 0;
    virtual void set_worker_id(const WorkerId& id) = 0;
    virtual WorkerId worker_id() const = 0;

    virtual bool supports_full_model() const = 0;
    virtual bool supports_stage() const = 0;

    virtual GenerationResult generate_local(
        const SessionId& session_id,
        const std::vector<int>& input_ids,
        int max_new_tokens,
        TokenCallback on_token = nullptr) = 0;

    virtual PrefillResponse prefill(const PrefillRequest& req) = 0;
    virtual StageRunResponse run_stage(const StageRunRequest& req) = 0;
    virtual DecodeStepResponse decode_step(const DecodeStepRequest& req) = 0;
};

std::unique_ptr<WorkerInterface> make_local_worker(const std::string& model_dir);
std::unique_ptr<WorkerInterface> make_remote_worker(const std::string& endpoint,
                                                    const WorkerId& worker_id);
