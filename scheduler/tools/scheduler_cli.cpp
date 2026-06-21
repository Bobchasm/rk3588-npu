#include "coordinator.h"

#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

void print_help() {
    std::cout
        << "Commands:\n"
        << "  /new                 create a new session and switch to it\n"
        << "  /new <system_prompt> create a new session with custom system prompt\n"
        << "  /switch <session_id> switch current session\n"
        << "  /sessions            list all sessions\n"
        << "  /history             show current session history\n"
        << "  /reset               reset current session to default system prompt\n"
        << "  /reset <system>      reset current session with custom system prompt\n"
        << "  /max_tokens <n>      set max_new_tokens for future requests\n"
        << "  /help                show this help\n"
        << "  /exit                quit\n";
}

std::string trim_left(const std::string& text) {
    std::size_t pos = text.find_first_not_of(" \t");
    return pos == std::string::npos ? "" : text.substr(pos);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model_dir> [remote_endpoint|ray:<actor_name>|head:<endpoint> tail:<endpoint> [stage:<endpoint> ...]]" << std::endl;
        std::cerr << "Example: " << argv[0] << " models/qwen1.5b-instruct 127.0.0.1:5001" << std::endl;
        std::cerr << "Example: " << argv[0] << " models/qwen1.5b-instruct ray:pc-full-model" << std::endl;
        std::cerr << "Example Ray distributed: " << argv[0] << " models/qwen1.5b-instruct ray:pc-distributed" << std::endl;
        std::cerr << "Example distributed: " << argv[0]
                  << " models/qwen1.5b-instruct head:127.0.0.1:5001 tail:127.0.0.1:5004 stage:127.0.0.1:5002 stage:127.0.0.1:5003"
                  << std::endl;
        return 1;
    }

    const std::string model_dir = argv[1];

    std::unique_ptr<WorkerInterface> worker;
    if (argc >= 3) {
        const std::string endpoint = argv[2];
        if (endpoint.rfind("head:", 0) == 0) {
            const std::string head_endpoint = endpoint.substr(5);
            if (argc < 4) {
                std::cerr << "Distributed mode requires tail endpoint" << std::endl;
                return 1;
            }
            const std::string tail_arg = argv[3];
            if (tail_arg.rfind("tail:", 0) != 0) {
                std::cerr << "Expected tail:<endpoint>" << std::endl;
                return 1;
            }
            const std::string tail_endpoint = tail_arg.substr(5);
            std::vector<std::string> stage_endpoints;
            for (int i = 4; i < argc; ++i) {
                const std::string arg = argv[i];
                if (arg.rfind("stage:", 0) != 0) {
                    std::cerr << "Expected stage:<endpoint>, got: " << arg << std::endl;
                    return 1;
                }
                stage_endpoints.push_back(arg.substr(6));
            }
            worker = make_distributed_worker(
                head_endpoint,
                stage_endpoints,
                tail_endpoint,
                "distributed-worker");
            if (!worker) {
                std::cerr << "Failed to create distributed worker chain" << std::endl;
                return 1;
            }
            std::cerr << "Using distributed workers: head=" << head_endpoint
                      << " tail=" << tail_endpoint
                      << " stages=" << stage_endpoints.size() << std::endl;
        } else if (endpoint.rfind("ray:", 0) == 0) {
            const std::string actor_name = endpoint.substr(4);
            if (actor_name.empty()) {
                std::cerr << "Invalid Ray actor name" << std::endl;
                return 1;
            }
            worker = make_ray_worker(actor_name, "ray-full-model");
            std::cerr << "Using Ray worker actor: " << actor_name << std::endl;
        } else {
            worker = make_remote_worker(endpoint, "worker-stage-0");
            if (!worker) {
                std::cerr << "Failed to create remote worker at " << endpoint << std::endl;
                return 1;
            }
            std::cerr << "Using remote worker: " << endpoint << std::endl;
        }
    } else {
        worker = make_local_worker(model_dir);
    }

    Coordinator coordinator(
        std::move(worker),
        std::unique_ptr<TokenizerAdapter>(new TokenizerAdapter(model_dir)));

    SessionId current_session_id = coordinator.create_session(
        "你是一个部署在 RK3588 NPU 集群上的中文助手。");
    int max_new_tokens = 64;

    std::cout << "Scheduler CLI started. Session=" << current_session_id
              << ". Enter text to generate, or type /help." << std::endl;
    while (true) {
        std::cout << "[" << current_session_id << "] > ";
        std::string line;
        if (!std::getline(std::cin, line)) {
            break;
        }
        if (line == "/exit" || line == "/quit") {
            break;
        }
        if (line == "/help") {
            print_help();
            continue;
        }
        if (line == "/sessions") {
            const std::vector<SessionId> session_ids = coordinator.list_sessions();
            std::cout << "Sessions:" << std::endl;
            for (const auto& session_id : session_ids) {
                std::cout << "  " << session_id;
                if (session_id == current_session_id) {
                    std::cout << "  (current)";
                }
                std::cout << std::endl;
            }
            continue;
        }
        if (line == "/history") {
            const std::vector<ChatMessage> history = coordinator.get_session_history(current_session_id);
            std::cout << "History for " << current_session_id << ":" << std::endl;
            for (const auto& message : history) {
                std::cout << "[" << message.role << "] " << message.text << std::endl;
            }
            continue;
        }
        if (line.rfind("/switch", 0) == 0) {
            const std::string target = trim_left(line.substr(7));
            if (target.empty()) {
                std::cerr << "usage: /switch <session_id>" << std::endl;
                continue;
            }
            if (!coordinator.session_exists(target)) {
                std::cerr << "session not found: " << target << std::endl;
                continue;
            }
            current_session_id = target;
            std::cout << "Switched to session " << current_session_id << std::endl;
            continue;
        }
        if (line.rfind("/new", 0) == 0) {
            std::string system_prompt = "你是一个部署在 RK3588 NPU 集群上的中文助手。";
            const std::string custom_prompt = trim_left(line.substr(4));
            if (!custom_prompt.empty()) {
                system_prompt = custom_prompt;
            }
            current_session_id = coordinator.create_session(system_prompt);
            std::cout << "Created session " << current_session_id << std::endl;
            continue;
        }
        if (line.rfind("/reset", 0) == 0) {
            std::string system_prompt = "你是一个部署在 RK3588 NPU 集群上的中文助手。";
            const std::string custom_prompt = trim_left(line.substr(6));
            if (!custom_prompt.empty()) {
                system_prompt = custom_prompt;
            }
            if (!coordinator.reset_session(current_session_id, system_prompt)) {
                std::cerr << "failed to reset session: " << current_session_id << std::endl;
                continue;
            }
            std::cout << "Reset session " << current_session_id << std::endl;
            continue;
        }
        if (line.rfind("/max_tokens", 0) == 0) {
            const std::string value = trim_left(line.substr(11));
            if (value.empty()) {
                std::cerr << "usage: /max_tokens <n>" << std::endl;
                continue;
            }
            try {
                max_new_tokens = std::stoi(value);
            } catch (const std::exception&) {
                std::cerr << "invalid max_new_tokens: " << value << std::endl;
                continue;
            }
            if (max_new_tokens <= 0) {
                std::cerr << "max_new_tokens must be > 0" << std::endl;
                max_new_tokens = 64;
                continue;
            }
            std::cout << "max_new_tokens set to " << max_new_tokens << std::endl;
            continue;
        }
        if (line.empty()) {
            continue;
        }

        GenerationResult result = coordinator.submit_text_request(
            current_session_id,
            line,
            max_new_tokens,
            nullptr);

        if (!result.error_message.empty()) {
            std::cerr << "request failed: " << result.error_message << std::endl;
            continue;
        }

        if (result.request_count > 0) {
            std::cout << "[request #" << result.request_count << "] ";
        }
        std::cout << "output_tokens=" << result.output_ids.size()
                  << " prefill_ms=" << result.prefill_ms
                  << " decode_ms=" << result.decode_ms << std::endl;
        std::cout << result.output_text << std::endl;
    }

    return 0;
}
