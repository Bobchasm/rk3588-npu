#include "api/llm_engine.h"

#include <algorithm>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <sys/time.h>
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

int64_t now_us() {
    struct timeval tv{};
    gettimeofday(&tv, nullptr);
    return static_cast<int64_t>(tv.tv_sec) * 1000000LL + tv.tv_usec;
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
    if (argc - arg_begin < 2) {
        std::fprintf(stderr,
                     "Usage: %s [--device cpu|gpu|auto] <model_dir> <token_id> [token_id ...]\n",
                     argv[0]);
        return 1;
    }

    std::signal(SIGINT, sig_cleanup);
    std::signal(SIGTERM, sig_cleanup);

    const std::string model_dir = argv[arg_begin];
    std::vector<int> input_ids;
    for (int i = arg_begin + 1; i < argc; ++i) {
        input_ids.push_back(std::atoi(argv[i]));
    }

    std::printf("worker-pc device: %s\n", compute_device_name(device));
    std::printf("model_dir: %s\n", model_dir.c_str());
    std::printf("input tokens: %d\n", static_cast<int>(input_ids.size()));

    LLMEngine engine;
    g_engine_ptr = &engine;

    const int64_t t0 = now_us();
    if (!engine.load(model_dir, device)) {
        std::fprintf(stderr, "failed to load model\n");
        return 1;
    }
    std::printf("model loaded in %.2f s\n", (now_us() - t0) / 1e6);

    engine.reset();
    GenerationConfig cfg;
    cfg.max_new_tokens = 10;
    const auto result = engine.generate(input_ids, cfg,
        [](int step, int id, float elapsed_ms) {
            std::printf("step %2d (%.0f ms): emit=%d\n", step, elapsed_ms, id);
        });

    std::printf("prefill_ms=%.2f decode_ms=%.2f decode_tokens=%d\n",
                result.prefill_ms, result.decode_ms, result.decode_tokens);
    std::printf("output token ids:");
    for (int id : result.output_ids) {
        std::printf(" %d", id);
    }
    std::printf("\n");
    return 0;
}

