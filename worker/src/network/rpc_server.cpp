#include "network/rpc_server.h"
#include <arpa/inet.h>
#include <atomic>
#include <cerrno>
#include <cstring>
#include <functional>
#include <iostream>
#include <netdb.h>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>

WorkerRpcServer::WorkerRpcServer()
    : running_(false), listen_fd_(-1) {
}

WorkerRpcServer::~WorkerRpcServer() {
    stop();
}

bool WorkerRpcServer::start(const std::string& address,
                            int port,
                            std::function<std::string(const std::string&)> handler) {
    if (running_) {
        std::cerr << "[WorkerRpcServer] already running" << std::endl;
        return false;
    }

    struct addrinfo hints{};
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE;

    std::string port_str = std::to_string(port);
    struct addrinfo* addr = nullptr;
    int err = getaddrinfo(address.c_str(), port_str.c_str(), &hints, &addr);
    if (err != 0) {
        std::cerr << "[WorkerRpcServer] getaddrinfo failed: " << gai_strerror(err) << std::endl;
        return false;
    }

    listen_fd_ = socket(addr->ai_family, addr->ai_socktype, addr->ai_protocol);
    if (listen_fd_ < 0) {
        std::cerr << "[WorkerRpcServer] socket failed: " << strerror(errno) << std::endl;
        freeaddrinfo(addr);
        return false;
    }

    int enable = 1;
    if (setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(enable)) < 0) {
        std::cerr << "[WorkerRpcServer] setsockopt failed: " << strerror(errno) << std::endl;
        close(listen_fd_);
        freeaddrinfo(addr);
        return false;
    }

    if (bind(listen_fd_, addr->ai_addr, addr->ai_addrlen) < 0) {
        std::cerr << "[WorkerRpcServer] bind failed: " << strerror(errno) << std::endl;
        close(listen_fd_);
        freeaddrinfo(addr);
        return false;
    }

    freeaddrinfo(addr);

    if (listen(listen_fd_, 4) < 0) {
        std::cerr << "[WorkerRpcServer] listen failed: " << strerror(errno) << std::endl;
        close(listen_fd_);
        return false;
    }

    running_ = true;
    server_thread_ = std::thread([this, handler]() {
        while (running_) {
            struct sockaddr_in client_addr{};
            socklen_t client_len = sizeof(client_addr);
            int client_fd = accept(listen_fd_, reinterpret_cast<struct sockaddr*>(&client_addr), &client_len);
            if (client_fd < 0) {
                if (running_) {
                    std::cerr << "[WorkerRpcServer] accept failed: " << strerror(errno) << std::endl;
                }
                break;
            }

            std::string request;
            char buffer[1024];
            while (true) {
                ssize_t bytes = recv(client_fd, buffer, sizeof(buffer), 0);
                if (bytes <= 0) break;
                request.append(buffer, bytes);
                auto pos = request.find('\n');
                if (pos != std::string::npos) {
                    request = request.substr(0, pos);
                    break;
                }
            }

            std::string response = handler(request);
            response.push_back('\n');
            send(client_fd, response.data(), response.size(), 0);
            close(client_fd);
        }
    });

    return true;
}

void WorkerRpcServer::stop() {
    if (!running_) return;
    running_ = false;
    if (listen_fd_ >= 0) {
        close(listen_fd_);
        listen_fd_ = -1;
    }
    if (server_thread_.joinable()) {
        server_thread_.join();
    }
}
