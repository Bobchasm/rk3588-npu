#include "coordinator.h"

#include <iostream>
#include <memory>
#include <string>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model_dir> [remote_endpoint|ray:<actor_name>]" << std::endl;
        std::cerr << "Example: " << argv[0] << " models/qwen1.5b-instruct 127.0.0.1:5001" << std::endl;
        std::cerr << "Example: " << argv[0] << " models/qwen1.5b-instruct ray:pc-full-model" << std::endl;
        return 1;
    }

    const std::string model_dir = argv[1];

    std::unique_ptr<WorkerInterface> worker;
    if (argc >= 3) {
        const std::string endpoint = argv[2];
        if (endpoint.rfind("ray:", 0) == 0) {
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

    const SessionId session_id = coordinator.create_session(
        "你是一个部署在 RK3588 NPU 集群上的中文助手。");

    std::cout << "Scheduler CLI started. Enter text to generate, or type /exit." << std::endl;
    while (true) {
        std::cout << "> ";
        std::string line;
        if (!std::getline(std::cin, line)) {
            break;
        }
        if (line == "/exit" || line == "/quit") {
            break;
        }
        if (line.empty()) {
            continue;
        }

        GenerationResult result = coordinator.submit_text_request(
            session_id,
            line,
            10,
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
