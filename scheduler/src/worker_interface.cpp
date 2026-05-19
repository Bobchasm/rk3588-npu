#include "worker_interface.h"
#include "network/rpc_client.h"
#include <iostream>

#ifdef SCHEDULER_USE_WORKER_CORE
#include <api/llm_engine.h>
#endif

class LocalWorker : public WorkerInterface {
public:
    explicit LocalWorker(const std::string& model_dir)
        : worker_id_("local_worker"), model_dir_(model_dir)
    {
#ifdef SCHEDULER_USE_WORKER_CORE
        if (!engine_.load(model_dir_)) {
            std::cerr << "[LocalWorker] failed to load model: " << model_dir_ << "\n";
        }
#else
        std::cerr << "[LocalWorker] host build stub mode; model load skipped." << std::endl;
#endif
    }

#ifdef SCHEDULER_USE_WORKER_CORE
    ~LocalWorker() override {
        engine_.destroy();
    }
#else
    ~LocalWorker() override = default;
#endif

    bool register_node(const WorkerId& id) override {
        worker_id_ = id;
        return true;
    }

    void set_worker_id(const WorkerId& id) override {
        worker_id_ = id;
    }

    WorkerId worker_id() const override {
        return worker_id_;
    }

    bool supports_full_model() const override {
        return true;
    }

    bool supports_stage() const override {
        return false;
    }

    GenerationResult generate_local(
        const SessionId& session_id,
        const std::vector<int>& input_ids,
        int max_new_tokens,
        TokenCallback on_token = nullptr) override
    {
#ifdef SCHEDULER_USE_WORKER_CORE
        GenerationConfig cfg;
        cfg.max_new_tokens = max_new_tokens;
        return engine_.generate(input_ids, cfg, on_token);
#else
        GenerationResult result;
        result.prefill_ms = 0;
        result.decode_ms = 0;
        result.output_ids.clear();
        result.error_message = "Host stub build: no RKNN backend available.";
        return result;
#endif
    }

    PrefillResponse prefill(const PrefillRequest& req) override {
        PrefillResponse resp;
        resp.request_id = req.request_id;
#ifdef SCHEDULER_USE_WORKER_CORE
        engine_.reset();
        GenerationConfig cfg;
        cfg.max_new_tokens = req.max_new_tokens;
        engine_.generate(req.input_ids, cfg, nullptr);
        resp.status = RequestStatus::OK;
#else
        resp.status = RequestStatus::ERROR;
        resp.message = "Host stub build: prefill unavailable.";
#endif
        return resp;
    }

    StageRunResponse run_stage(const StageRunRequest& req) override {
        StageRunResponse resp;
        resp.request_id = req.request_id;
        resp.status = RequestStatus::ERROR;
        resp.message = "Stage execution not supported by LocalWorker";
        return resp;
    }

    DecodeStepResponse decode_step(const DecodeStepRequest& req) override {
        DecodeStepResponse resp;
        resp.request_id = req.request_id;
        resp.status = RequestStatus::ERROR;
        resp.message = "Decode step not supported by LocalWorker";
        return resp;
    }

private:
    WorkerId worker_id_;
    std::string model_dir_;
#ifdef SCHEDULER_USE_WORKER_CORE
    LLMEngine engine_;
#endif
};

class RemoteWorker : public WorkerInterface {
public:
    RemoteWorker(const std::string& endpoint, const WorkerId& worker_id)
        : worker_id_(worker_id), endpoint_(endpoint), connected_(false), next_request_id_(1)
    {
        connected_ = client_.connect(endpoint_);
        if (!connected_) {
            std::cerr << "[RemoteWorker] failed to connect to " << endpoint_ << std::endl;
        }
    }

    ~RemoteWorker() override = default;

    bool register_node(const WorkerId& id) override {
        worker_id_ = id;
        return true;
    }

    void set_worker_id(const WorkerId& id) override {
        worker_id_ = id;
    }

    WorkerId worker_id() const override {
        return worker_id_;
    }

    bool supports_full_model() const override {
        return true;
    }

    bool supports_stage() const override {
        return false;
    }

    bool connected() const {
        return connected_;
    }

    GenerationResult generate_local(
        const SessionId& session_id,
        const std::vector<int>& input_ids,
        int max_new_tokens,
        TokenCallback on_token = nullptr) override
    {
        GenerationResult result;
        if (!connected_) {
            result.error_message = "RemoteWorker not connected";
            return result;
        }

        RequestId request_id = next_request_id_++;
        if (!client_.send_generate(session_id, request_id, input_ids, max_new_tokens, result)) {
            result.error_message = "Remote generation failed";
            return result;
        }

        if (on_token) {
            for (int step = 0; step < (int)result.output_ids.size(); ++step) {
                on_token(step, result.output_ids[step], result.decode_ms);
            }
        }

        return result;
    }

    PrefillResponse prefill(const PrefillRequest& req) override {
        PrefillResponse resp;
        resp.request_id = req.request_id;
        if (!connected_) {
            resp.status = RequestStatus::ERROR;
            resp.message = "RemoteWorker not connected";
            return resp;
        }
        if (!client_.send_prefill(req, resp)) {
            resp.status = RequestStatus::ERROR;
            resp.message = "Remote prefill failed";
        }
        return resp;
    }

    StageRunResponse run_stage(const StageRunRequest& req) override {
        StageRunResponse resp;
        resp.request_id = req.request_id;
        resp.status = RequestStatus::ERROR;
        resp.message = "Stage execution not supported by RemoteWorker";
        return resp;
    }

    DecodeStepResponse decode_step(const DecodeStepRequest& req) override {
        DecodeStepResponse resp;
        resp.request_id = req.request_id;
        resp.status = RequestStatus::ERROR;
        resp.message = "Decode step not supported by RemoteWorker";
        return resp;
    }

private:
    WorkerId worker_id_;
    std::string endpoint_;
    bool connected_;
    RpcClient client_;
    RequestId next_request_id_;
};

std::unique_ptr<WorkerInterface> make_local_worker(const std::string& model_dir) {
    return std::make_unique<LocalWorker>(model_dir);
}

std::unique_ptr<WorkerInterface> make_remote_worker(const std::string& endpoint,
                                                    const WorkerId& worker_id) {
    auto worker = std::make_unique<RemoteWorker>(endpoint, worker_id);
    if (!worker->connected()) {
        return nullptr;
    }
    return worker;
}
