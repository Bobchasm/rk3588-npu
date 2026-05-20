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

    GenerationResult generate_tokens(const GenerateTokensRequest& req,
                                     TokenCallback on_token = nullptr) override
    {
        (void)req.context;
#ifdef SCHEDULER_USE_WORKER_CORE
        GenerationConfig cfg;
        cfg.max_new_tokens = req.generation.max_new_tokens;
        cfg.repetition_window = req.generation.repetition_window;

        const auto local_result = engine_.generate(
            std::vector<int>(req.input_token_ids.begin(), req.input_token_ids.end()),
            cfg,
            on_token);

        GenerationResult result;
        result.output_ids.assign(local_result.output_ids.begin(), local_result.output_ids.end());
        result.prefill_tokens = local_result.prefill_tokens;
        result.decode_tokens = local_result.decode_tokens;
        result.prefill_ms = local_result.prefill_ms;
        result.decode_ms = local_result.decode_ms;
        result.hit_stop = local_result.hit_stop;
        result.hit_repetition = local_result.hit_repetition;
        return result;
#else
        GenerationResult result;
        result.error_message = "Host stub build: no RKNN backend available.";
        return result;
#endif
    }

    StageForwardResponse forward_stage(const StageForwardRequest& req) override {
        StageForwardResponse resp;
        resp.context = req.context;
        resp.status = RequestStatus::kUnsupported;
        resp.message = "LocalWorker does not support stage forwarding yet";
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

    GenerationResult generate_tokens(const GenerateTokensRequest& req,
                                     TokenCallback on_token = nullptr) override
    {
        GenerationResult result;
        if (!connected_) {
            result.error_message = "RemoteWorker not connected";
            return result;
        }

        GenerateTokensRequest request = req;
        if (request.context.request_id == 0) {
            request.context.request_id = next_request_id_++;
        }
        request.context.route.target_worker_id = worker_id_;

        GenerateTokensResponse response;
        if (!client_.send_generate_tokens(request, response)) {
            result.error_message = "Remote generate RPC failed";
            return result;
        }
        if (!distributed::is_success(response.status)) {
            result.error_message = response.message;
            return result;
        }

        result.output_ids.assign(response.output_token_ids.begin(), response.output_token_ids.end());
        result.prefill_tokens = response.prefill_tokens;
        result.decode_tokens = response.decode_tokens;
        result.prefill_ms = response.prefill_ms;
        result.decode_ms = response.decode_ms;
        result.hit_stop = response.hit_stop;
        result.hit_repetition = response.hit_repetition;

        if (on_token) {
            for (int step = 0; step < (int)result.output_ids.size(); ++step) {
                on_token(step, result.output_ids[step], result.decode_ms);
            }
        }

        return result;
    }

    StageForwardResponse forward_stage(const StageForwardRequest& req) override {
        StageForwardResponse resp;
        resp.context = req.context;
        if (!connected_) {
            resp.status = RequestStatus::kError;
            resp.message = "RemoteWorker not connected";
            return resp;
        }
        if (!client_.send_forward_stage(req, resp)) {
            resp.status = RequestStatus::kError;
            resp.message = "Remote stage RPC failed";
        }
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
