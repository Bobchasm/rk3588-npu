#pragma once

#include <atomic>
#include <functional>
#include <string>
#include <thread>

// 连接端点信息单独抽象出来，便于在 RPC 层和业务层之间传递。
// 当前先记录最关键的客户端 IP 与端口，后续若要补充 fd / 协议族等信息也方便扩展。
struct ClientEndpoint {
    std::string ip;
    int port = 0;
};

class WorkerRpcServer {
public:
    WorkerRpcServer();
    ~WorkerRpcServer();

    bool start(const std::string& address,
               int port,
               std::function<std::string(const std::string&, const ClientEndpoint&)> handler);
    void stop();

private:
    std::atomic<bool> running_;
    int listen_fd_;
    std::thread server_thread_;
};
