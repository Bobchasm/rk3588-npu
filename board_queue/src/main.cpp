#include "rkq/task_queue.h"

#include <exception>
#include <iostream>
#include <string>
#include <vector>

namespace {

void print_usage(const char* argv0) {
    std::cerr
        << "Usage:\n"
        << "  " << argv0 << " run -- <command> [args...]\n"
        << "  " << argv0 << " status\n"
        << "  " << argv0 << " current\n"
        << "  " << argv0 << " cancel <task_id>\n"
        << "\n"
        << "Examples:\n"
        << "  " << argv0 << " run -- ./worker_rpc_server Qwen1.5B 0.0.0.0:5001\n"
        << "  " << argv0 << " run -- ./main -m ./models/model.gguf -p hello -n 32\n";
}

std::vector<std::string> collect_run_args(int argc, char** argv) {
    std::vector<std::string> args;
    bool seen_sep = false;
    for (int i = 2; i < argc; ++i) {
        if (!seen_sep) {
            if (std::string(argv[i]) == "--") {
                seen_sep = true;
            }
            continue;
        }
        args.emplace_back(argv[i]);
    }
    return args;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 2;
    }

    try {
        rkq::TaskQueueManager manager;
        const std::string command = argv[1];

        if (command == "run") {
            const auto args = collect_run_args(argc, argv);
            if (args.empty()) {
                print_usage(argv[0]);
                return 2;
            }
            return manager.run_command(args);
        }
        if (command == "status") {
            return manager.print_status();
        }
        if (command == "current") {
            return manager.print_current();
        }
        if (command == "cancel") {
            if (argc < 3) {
                print_usage(argv[0]);
                return 2;
            }
            return manager.cancel_task(argv[2]);
        }

        print_usage(argv[0]);
        return 2;
    } catch (const std::exception& e) {
        std::cerr << "[rkq] fatal error: " << e.what() << "\n";
        return 1;
    }
}

