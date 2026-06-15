#include "worker_interface.h"

#include "network/rpc_client.h"
#include <cstdio>
#include <iostream>
#include <sstream>

#ifdef SCHEDULER_USE_WORKER_CORE
#include <api/llm_engine.h>
#endif

namespace {

#ifndef SCHEDULER_REPO_ROOT
#define SCHEDULER_REPO_ROOT "."
#endif

std::string escape_shell_arg(const std::string& arg) {
    std::string escaped;
    escaped.reserve(arg.size() + 2);
    for (char ch : arg) {
        if (ch == '\\' || ch == '"') {
            escaped.push_back('\\');
        }
        escaped.push_back(ch);
    }
    return "\"" + escaped + "\"";
}

bool run_command(const std::string& command, std::string& output) {
    FILE* pipe = popen(command.c_str(), "r");
    if (!pipe) {
        return false;
    }

    output.clear();
    char buffer[256];
    while (fgets(buffer, sizeof(buffer), pipe)) {
        output += buffer;
    }

    const int status = pclose(pipe);
    return status == 0;
}

std::string ray_generate_script_path() {
    return std::string(SCHEDULER_REPO_ROOT) + "/scheduler/tools/ray_generate.py";
}

std::string scheduler_python_executable() {
    const char* env_python = std::getenv("SCHEDULER_RAY_PYTHON");
    if (env_python && *env_python) {
        return std::string(env_python);
    }
    return "python3";
}

}  // namespace

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

    bool supports_tokens_to_hidden() const override {
        return false;
    }

    bool supports_stage() const override {
        return false;
    }

    bool supports_hidden_to_token() const override {
        return false;
    }

    bool reset_cache() override {
#ifdef SCHEDULER_USE_WORKER_CORE
        engine_.reset();
        return true;
#else
        return false;
#endif
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

    TokensToHiddenResponse tokens_to_hidden(const TokensToHiddenRequest& req) override {
        TokensToHiddenResponse resp;
        resp.context = req.context;
        resp.status = RequestStatus::kUnsupported;
        resp.message = "LocalWorker does not support tokens_to_hidden yet";
        return resp;
    }

    HiddenToTokenResponse hidden_to_token(const HiddenToTokenRequest& req) override {
        HiddenToTokenResponse resp;
        resp.context = req.context;
        resp.status = RequestStatus::kUnsupported;
        resp.message = "LocalWorker does not support hidden_to_token yet";
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

    bool supports_tokens_to_hidden() const override {
        return true;
    }

    bool supports_stage() const override {
        return true;
    }

    bool supports_hidden_to_token() const override {
        return true;
    }

    bool reset_cache() override {
        return connected_ && client_.send_reset_cache();
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

    TokensToHiddenResponse tokens_to_hidden(const TokensToHiddenRequest& req) override {
        TokensToHiddenResponse resp;
        resp.context = req.context;
        if (!connected_) {
            resp.status = RequestStatus::kError;
            resp.message = "RemoteWorker not connected";
            return resp;
        }
        if (!client_.send_tokens_to_hidden(req, resp)) {
            resp.status = RequestStatus::kError;
            resp.message = "Remote tokens_to_hidden RPC failed";
        }
        return resp;
    }

    HiddenToTokenResponse hidden_to_token(const HiddenToTokenRequest& req) override {
        HiddenToTokenResponse resp;
        resp.context = req.context;
        if (!connected_) {
            resp.status = RequestStatus::kError;
            resp.message = "RemoteWorker not connected";
            return resp;
        }
        if (!client_.send_hidden_to_token(req, resp)) {
            resp.status = RequestStatus::kError;
            resp.message = "Remote hidden_to_token RPC failed";
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

class RayWorker : public WorkerInterface {
public:
    RayWorker(std::string actor_name, WorkerId worker_id)
        : worker_id_(std::move(worker_id)), actor_name_(std::move(actor_name)) {}

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

    bool supports_tokens_to_hidden() const override {
        return true;
    }

    bool supports_stage() const override {
        return true;
    }

    bool supports_hidden_to_token() const override {
        return true;
    }

    bool reset_cache() override {
        return false;
    }

    GenerationResult generate_tokens(const GenerateTokensRequest& req,
                                     TokenCallback on_token = nullptr) override
    {
        (void)req.context;
        std::ostringstream cmd;
        cmd << "PYTHONPATH=" << escape_shell_arg(std::string(SCHEDULER_REPO_ROOT) + "/bindings/python")
            << " " << scheduler_python_executable()
            << " "
            << escape_shell_arg(ray_generate_script_path())
            << " --actor-name " << escape_shell_arg(actor_name_)
            << " --max-new-tokens " << req.generation.max_new_tokens
            << " --repetition-window " << req.generation.repetition_window;
        for (int token_id : req.input_token_ids) {
            cmd << " " << token_id;
        }

        std::string raw;
        GenerationResult result;
        if (!run_command(cmd.str(), raw)) {
            result.error_message = "Ray worker command failed";
            return result;
        }

        std::istringstream lines(raw);
        std::string line;
        while (std::getline(lines, line)) {
            if (line.rfind("STATUS ", 0) == 0) {
                if (line.substr(7) != "OK") {
                    if (result.error_message.empty()) {
                        result.error_message = "Ray worker returned error";
                    }
                }
            } else if (line.rfind("ERROR ", 0) == 0) {
                result.error_message = line.substr(6);
            } else if (line.rfind("OUTPUT_IDS ", 0) == 0) {
                std::istringstream iss(line.substr(11));
                int token_id = 0;
                while (iss >> token_id) {
                    result.output_ids.push_back(token_id);
                }
            } else if (line.rfind("PREFILL_TOKENS ", 0) == 0) {
                result.prefill_tokens = std::stoi(line.substr(15));
            } else if (line.rfind("DECODE_TOKENS ", 0) == 0) {
                result.decode_tokens = std::stoi(line.substr(14));
            } else if (line.rfind("PREFILL_MS ", 0) == 0) {
                result.prefill_ms = std::stof(line.substr(11));
            } else if (line.rfind("DECODE_MS ", 0) == 0) {
                result.decode_ms = std::stof(line.substr(10));
            } else if (line.rfind("HIT_STOP ", 0) == 0) {
                result.hit_stop = line.substr(9) == "1";
            } else if (line.rfind("HIT_REPETITION ", 0) == 0) {
                result.hit_repetition = line.substr(15) == "1";
            } else if (line.rfind("REQUEST_COUNT ", 0) == 0) {
                result.request_count = std::stoi(line.substr(14));
            }
        }

        if (result.error_message.empty() && on_token) {
            for (int step = 0; step < static_cast<int>(result.output_ids.size()); ++step) {
                on_token(step, result.output_ids[step], result.decode_ms);
            }
        }
        return result;
    }

    StageForwardResponse forward_stage(const StageForwardRequest& req) override {
        StageForwardResponse resp;
        resp.context = req.context;
        resp.status = RequestStatus::kUnsupported;
        resp.message = "RayWorker does not support stage forwarding yet";
        return resp;
    }

    TokensToHiddenResponse tokens_to_hidden(const TokensToHiddenRequest& req) override {
        TokensToHiddenResponse resp;
        resp.context = req.context;
        resp.status = RequestStatus::kUnsupported;
        resp.message = "RayWorker does not support tokens_to_hidden yet";
        return resp;
    }

    HiddenToTokenResponse hidden_to_token(const HiddenToTokenRequest& req) override {
        HiddenToTokenResponse resp;
        resp.context = req.context;
        resp.status = RequestStatus::kUnsupported;
        resp.message = "RayWorker does not support hidden_to_token yet";
        return resp;
    }

private:
    WorkerId worker_id_;
    std::string actor_name_;
};

class DistributedWorker : public WorkerInterface {
public:
    DistributedWorker(std::unique_ptr<WorkerInterface> head_worker,
                      std::vector<std::unique_ptr<WorkerInterface>> stage_workers,
                      std::unique_ptr<WorkerInterface> tail_worker,
                      WorkerId worker_id)
        : worker_id_(std::move(worker_id)),
          head_worker_(std::move(head_worker)),
          stage_workers_(std::move(stage_workers)),
          tail_worker_(std::move(tail_worker)) {}

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
        return false;
    }

    bool supports_tokens_to_hidden() const override {
        return head_worker_ && head_worker_->supports_tokens_to_hidden();
    }

    bool supports_stage() const override {
        return !stage_workers_.empty();
    }

    bool supports_hidden_to_token() const override {
        return tail_worker_ && tail_worker_->supports_hidden_to_token();
    }

    bool reset_cache() override {
        bool ok = true;
        if (head_worker_) {
            ok = head_worker_->reset_cache() && ok;
        }
        for (auto& stage_worker : stage_workers_) {
            ok = stage_worker->reset_cache() && ok;
        }
        if (tail_worker_) {
            ok = tail_worker_->reset_cache() && ok;
        }
        return ok;
    }

    GenerationResult generate_tokens(const GenerateTokensRequest& req,
                                     TokenCallback on_token = nullptr) override {
        GenerationResult result;
        if (!head_worker_ || !tail_worker_) {
            result.error_message = "distributed worker missing head/tail";
            return result;
        }
        if (!reset_cache()) {
            result.error_message = "failed to reset distributed worker caches";
            return result;
        }

        const std::vector<int> prompt_ids(req.input_token_ids.begin(), req.input_token_ids.end());
        std::vector<int> current_input = prompt_ids;

        auto is_stop = [](int token_id) {
            return token_id == 151645 || token_id == 151643;
        };

        for (int step = 0; step < req.generation.max_new_tokens; ++step) {
            TokensToHiddenRequest head_req;
            head_req.context = req.context;
            head_req.context.request_id = req.context.request_id != 0 ? req.context.request_id : static_cast<uint64_t>(step + 1);
            head_req.context.route.pos_base =
                static_cast<int32_t>(prompt_ids.size() + result.output_ids.size() - current_input.size());
            head_req.input_token_ids.assign(current_input.begin(), current_input.end());

            TokensToHiddenResponse head_resp = head_worker_->tokens_to_hidden(head_req);
            if (!distributed::is_success(head_resp.status)) {
                result.error_message = head_resp.message.empty() ? "head tokens_to_hidden failed" : head_resp.message;
                return result;
            }

            TensorBuffer hidden = head_resp.output_tensor;
            const int32_t pos_base = head_req.context.route.pos_base;

            for (size_t i = 0; i < stage_workers_.size(); ++i) {
                StageForwardRequest stage_req;
                stage_req.context = head_req.context;
                stage_req.context.route.hop_index = static_cast<int32_t>(i + 1);
                stage_req.context.route.pos_base = pos_base;
                stage_req.input_tensor = hidden;

                StageForwardResponse stage_resp = stage_workers_[i]->forward_stage(stage_req);
                if (!distributed::is_success(stage_resp.status)) {
                    result.error_message = stage_resp.message.empty() ? "stage forward failed" : stage_resp.message;
                    return result;
                }
                hidden = stage_resp.output_tensor;
            }

            HiddenToTokenRequest tail_req;
            tail_req.context = head_req.context;
            tail_req.context.route.hop_index = static_cast<int32_t>(stage_workers_.size() + 1);
            tail_req.context.route.pos_base = pos_base;
            tail_req.input_tensor = hidden;

            HiddenToTokenResponse tail_resp = tail_worker_->hidden_to_token(tail_req);
            if (!distributed::is_success(tail_resp.status)) {
                result.error_message = tail_resp.message.empty() ? "tail hidden_to_token failed" : tail_resp.message;
                return result;
            }

            const int token_id = tail_resp.output_token_id;
            result.output_ids.push_back(token_id);
            if (on_token) {
                on_token(step, token_id, 0.0f);
            }
            if (is_stop(token_id)) {
                result.hit_stop = true;
                break;
            }
            current_input.assign(1, token_id);
        }

        result.prefill_tokens = static_cast<int>(prompt_ids.size());
        result.decode_tokens = static_cast<int>(result.output_ids.size());
        return result;
    }

    StageForwardResponse forward_stage(const StageForwardRequest& req) override {
        StageForwardResponse resp;
        resp.context = req.context;
        resp.status = RequestStatus::kUnsupported;
        resp.message = "DistributedWorker does not expose raw stage forwarding";
        return resp;
    }

    TokensToHiddenResponse tokens_to_hidden(const TokensToHiddenRequest& req) override {
        TokensToHiddenResponse resp;
        resp.context = req.context;
        resp.status = RequestStatus::kUnsupported;
        resp.message = "DistributedWorker does not expose raw tokens_to_hidden";
        return resp;
    }

    HiddenToTokenResponse hidden_to_token(const HiddenToTokenRequest& req) override {
        HiddenToTokenResponse resp;
        resp.context = req.context;
        resp.status = RequestStatus::kUnsupported;
        resp.message = "DistributedWorker does not expose raw hidden_to_token";
        return resp;
    }

private:
    WorkerId worker_id_;
    std::unique_ptr<WorkerInterface> head_worker_;
    std::vector<std::unique_ptr<WorkerInterface>> stage_workers_;
    std::unique_ptr<WorkerInterface> tail_worker_;
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

std::unique_ptr<WorkerInterface> make_distributed_worker(
    const std::string& head_endpoint,
    const std::vector<std::string>& stage_endpoints,
    const std::string& tail_endpoint,
    const WorkerId& worker_id) {
    auto head_worker = make_remote_worker(head_endpoint, worker_id + "-head");
    auto tail_worker = make_remote_worker(tail_endpoint, worker_id + "-tail");
    if (!head_worker || !tail_worker) {
        return nullptr;
    }

    std::vector<std::unique_ptr<WorkerInterface>> stage_workers;
    stage_workers.reserve(stage_endpoints.size());
    for (size_t i = 0; i < stage_endpoints.size(); ++i) {
        auto stage_worker = make_remote_worker(stage_endpoints[i], worker_id + "-stage-" + std::to_string(i));
        if (!stage_worker) {
            return nullptr;
        }
        stage_workers.push_back(std::move(stage_worker));
    }

    return std::make_unique<DistributedWorker>(
        std::move(head_worker),
        std::move(stage_workers),
        std::move(tail_worker),
        worker_id);
}

std::unique_ptr<WorkerInterface> make_ray_worker(const std::string& actor_name,
                                                 const WorkerId& worker_id) {
    return std::make_unique<RayWorker>(actor_name, worker_id);
}
