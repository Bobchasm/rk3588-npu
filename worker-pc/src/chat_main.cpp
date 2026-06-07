#include "api/llm_engine.h"

#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

LLMEngine* g_engine_ptr = nullptr;

void sig_cleanup(int sig) {
    std::fprintf(stderr, "\n[signal %d] cleanup worker-pc resources...\n", sig);
    if (g_engine_ptr) {
        g_engine_ptr->destroy();
        g_engine_ptr = nullptr;
    }
    std::_Exit(0);
}

std::vector<int> parse_ids(const std::string& line) {
    std::vector<int> ids;
    std::istringstream iss(line);
    int value = 0;
    while (iss >> value) {
        ids.push_back(value);
    }
    return ids;
}

int parse_device_and_shift(int argc, char** argv, ComputeDevice& device) {
    int start = 1;
    device = ComputeDevice::kCpu;
    if (argc >= 3 && std::string(argv[1]) == "--device") {
        device = parse_compute_device(argv[2]);
        start = 3;
    }
    return start;
}

}  // namespace

int main(int argc, char* argv[]) {
    ComputeDevice device = ComputeDevice::kCpu;
    const int arg_begin = parse_device_and_shift(argc, argv, device);
    if (argc - arg_begin < 1) {
        std::fprintf(stderr,
                     "Usage: %s [--device cpu|gpu|auto] <model_dir> [max_new_tokens]\n",
                     argv[0]);
        return 1;
    }

    std::signal(SIGINT, sig_cleanup);
    std::signal(SIGTERM, sig_cleanup);

    const std::string model_dir = argv[arg_begin];
    const int max_new_tokens = (argc > arg_begin + 1) ? std::atoi(argv[arg_begin + 1]) : 128;

    LLMEngine engine;
    g_engine_ptr = &engine;
    if (!engine.load(model_dir, device)) {
        std::fprintf(stderr, "[worker-pc/chat] failed to load model\n");
        return 1;
    }

    std::fprintf(stderr, "[worker-pc/chat] ready, device=%s max_new_tokens=%d\n",
                 compute_device_name(device), max_new_tokens);
    std::printf("READY\n");
    std::fflush(stdout);

    GenerationConfig cfg;
    cfg.max_new_tokens = max_new_tokens;

    std::string line;
    while (std::getline(std::cin, line)) {
        const size_t a = line.find_first_not_of(" \t\r\n");
        const size_t b = line.find_last_not_of(" \t\r\n");
        if (a == std::string::npos) {
            continue;
        }
        line = line.substr(a, b - a + 1);
        if (line == "EXIT") break;
        if (line == "RESET") {
            engine.reset();
            std::printf("OK\n");
            std::fflush(stdout);
            continue;
        }

        std::vector<int> input_ids = parse_ids(line);
        if (input_ids.empty()) {
            std::printf("ERR empty_input\n");
            std::fflush(stdout);
            continue;
        }

        engine.reset();
        const auto result = engine.generate(input_ids, cfg, nullptr);
        std::printf("OK");
        for (int id : result.output_ids) {
            std::printf(" %d", id);
        }
        std::printf("\n");
        std::fflush(stdout);
    }

    engine.destroy();
    return 0;
}

