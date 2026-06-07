#include "backend/device_type.h"
#include "network/worker_service.h"

#include <iostream>
#include <string>

namespace {

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

}  // namespace

int main(int argc, char** argv) {
    ComputeDevice device = ComputeDevice::kCpu;
    const int arg_begin = parse_device_and_shift(argc, argv, device);
    if (argc - arg_begin < 1) {
        std::cerr << "Usage: " << argv[0]
                  << " [--device cpu|gpu|auto] <model_dir> [address:port]" << std::endl;
        return 1;
    }

    const std::string model_dir = argv[arg_begin];
    std::string endpoint = "0.0.0.0:5001";
    if (argc > arg_begin + 1) {
        endpoint = argv[arg_begin + 1];
    }

    std::string address;
    int port = 0;
    if (!parse_address_port(endpoint, address, port)) {
        std::cerr << "Invalid endpoint: " << endpoint << std::endl;
        return 1;
    }

    WorkerService service;
    if (!service.register_service(address, port, model_dir, device)) {
        std::cerr << "Failed to start worker-pc RPC service." << std::endl;
        return 1;
    }

    std::cout << "worker-pc RPC server listening on " << address << ":" << port
              << " device=" << compute_device_name(device) << std::endl;
    std::cout << "Type ENTER to stop." << std::endl;
    std::string line;
    std::getline(std::cin, line);
    return 0;
}
