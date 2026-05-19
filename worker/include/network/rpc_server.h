#pragma once

#include <atomic>
#include <functional>
#include <string>
#include <thread>

class WorkerRpcServer {
public:
    WorkerRpcServer();
    ~WorkerRpcServer();

    bool start(const std::string& address,
               int port,
               std::function<std::string(const std::string&)> handler);
    void stop();

private:
    std::atomic<bool> running_;
    int listen_fd_;
    std::thread server_thread_;
};
