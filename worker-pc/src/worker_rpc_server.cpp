#include "backend/device_type.h"
#include "network/worker_service.h"

#include <iostream>
#include <string>

namespace {

using RuntimeMode = WorkerService::RuntimeMode;
using ServiceConfig = WorkerService::ServiceConfig;

bool parse_address_port(const std::string& endpoint, std::string& address, int& port) {
    const auto pos = endpoint.find(':');
    if (pos == std::string::npos) {
        return false;
    }
    address = endpoint.substr(0, pos);
    port = std::stoi(endpoint.substr(pos + 1));
    return true;
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

bool parse_runtime_mode(const std::string& text, RuntimeMode& mode) {
    if (text == "full") {
        mode = RuntimeMode::kFullModel;
        return true;
    }
    if (text == "head") {
        mode = RuntimeMode::kHead;
        return true;
    }
    if (text == "stage") {
        mode = RuntimeMode::kStage;
        return true;
    }
    if (text == "tail") {
        mode = RuntimeMode::kTail;
        return true;
    }
    return false;
}

}  // namespace

int main(int argc, char** argv) {
    ComputeDevice device = ComputeDevice::kCpu;
    const int arg_begin = parse_device_and_shift(argc, argv, device);
    if (argc - arg_begin < 1) {
        std::cerr << "Usage: " << argv[0]
                  << " [--device cpu|gpu|auto] [--mode full|head|stage|tail]"
                  << " [--layer-begin N --layer-end M] <model_dir> [address:port]" << std::endl;
        return 1;
    }

    ServiceConfig service_cfg;
    int cursor = arg_begin;
    while (cursor < argc && std::string(argv[cursor]).rfind("--", 0) == 0) {
        const std::string flag = argv[cursor];
        if (flag == "--mode") {
            if (cursor + 1 >= argc || !parse_runtime_mode(argv[cursor + 1], service_cfg.mode)) {
                std::cerr << "Invalid --mode value" << std::endl;
                return 1;
            }
            cursor += 2;
            continue;
        }
        if (flag == "--layer-begin") {
            if (cursor + 1 >= argc) {
                std::cerr << "Missing value for --layer-begin" << std::endl;
                return 1;
            }
            service_cfg.layer_begin = std::stoi(argv[cursor + 1]);
            cursor += 2;
            continue;
        }
        if (flag == "--layer-end") {
            if (cursor + 1 >= argc) {
                std::cerr << "Missing value for --layer-end" << std::endl;
                return 1;
            }
            service_cfg.layer_end = std::stoi(argv[cursor + 1]);
            cursor += 2;
            continue;
        }
        std::cerr << "Unknown option: " << flag << std::endl;
        return 1;
    }

    if (argc - cursor < 1) {
        std::cerr << "Missing model_dir" << std::endl;
        return 1;
    }

    if (service_cfg.mode == RuntimeMode::kFullModel) {
        service_cfg.layer_begin = 0;
        service_cfg.layer_end = -1;
    } else if (service_cfg.layer_end >= 0 && service_cfg.layer_begin >= service_cfg.layer_end) {
        std::cerr << "Invalid stage layer range" << std::endl;
        return 1;
    }

    const std::string model_dir = argv[cursor];
    std::string endpoint = "0.0.0.0:5001";
    if (argc > cursor + 1) {
        endpoint = argv[cursor + 1];
    }

    std::string address;
    int port = 0;
    if (!parse_address_port(endpoint, address, port)) {
        std::cerr << "Invalid endpoint: " << endpoint << std::endl;
        return 1;
    }

    WorkerService service;
    if (!service.register_service(address, port, model_dir, device, service_cfg)) {
        std::cerr << "Failed to start worker-pc RPC service." << std::endl;
        return 1;
    }

    std::cout << "worker-pc RPC server listening on " << address << ":" << port
              << " device=" << compute_device_name(device)
              << " mode="
              << (service_cfg.mode == RuntimeMode::kFullModel ? "full" :
                  service_cfg.mode == RuntimeMode::kHead ? "head" :
                  service_cfg.mode == RuntimeMode::kTail ? "tail" : "stage")
              << " layers=[" << service_cfg.layer_begin << ","
              << service_cfg.layer_end << ")" << std::endl;
    std::cout << "Type ENTER to stop." << std::endl;
    std::string line;
    std::getline(std::cin, line);
    return 0;
}
