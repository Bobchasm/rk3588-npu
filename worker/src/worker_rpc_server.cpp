#include "network/worker_service.h"
#include <iostream>
#include <string>

static bool parse_address_port(const std::string& endpoint,
                               std::string& address,
                               int& port) {
    auto pos = endpoint.find(':');
    if (pos == std::string::npos) return false;
    address = endpoint.substr(0, pos);
    port = std::stoi(endpoint.substr(pos + 1));
    return true;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model_dir> [address:port]" << std::endl;
        return 1;
    }

    std::string model_dir = argv[1];
    std::string endpoint = "0.0.0.0:5001";
    if (argc >= 3) {
        endpoint = argv[2];
    }

    std::string address;
    int port = 0;
    if (!parse_address_port(endpoint, address, port)) {
        std::cerr << "Invalid endpoint: " << endpoint << std::endl;
        return 1;
    }

    WorkerService service;
    if (!service.register_service(address, port, model_dir)) {
        std::cerr << "Failed to start worker RPC service." << std::endl;
        return 1;
    }

    std::cout << "Worker RPC server listening on " << address << ":" << port << std::endl;
    std::cout << "Type ENTER to stop." << std::endl;
    std::string line;
    std::getline(std::cin, line);

    return 0;
}
