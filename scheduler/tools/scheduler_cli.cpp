#include "coordinator.h"

#include <iostream>
#include <memory>
#include <string>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model_dir> [remote_endpoint]" << std::endl;
        std::cerr << "Example: " << argv[0] << " models/qwen1.5b-instruct 127.0.0.1:5001" << std::endl;
        return 1;
    }

    const std::string model_dir = argv[1];

    std::unique_ptr<WorkerInterface> worker;
    if (argc >= 3) {
        const std::string endpoint = argv[2];
        worker = make_remote_worker(endpoint, "worker-stage-0");
        if (!worker) {
            std::cerr << "Failed to create remote worker at " << endpoint << std::endl;
            return 1;
        }
        std::cerr << "Using remote worker: " << endpoint << std::endl;
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
            [](int step, int id, float elapsed_ms) {
                std::cout << "[token] step=" << step
                          << " id=" << id
                          << " elapsed_ms=" << elapsed_ms << std::endl;
            });

        if (!result.error_message.empty()) {
            std::cerr << "request failed: " << result.error_message << std::endl;
            continue;
        }

        std::cout << "prompt tokens: " << result.prompt_ids.size()
                  << ", output tokens: " << result.output_ids.size() << std::endl;
        std::cout << "decoded: " << result.output_text << std::endl;
        std::cout << "prefill_ms=" << result.prefill_ms
                  << " decode_ms=" << result.decode_ms << std::endl;
    }

    return 0;
}
