#pragma once

#include <string>

class RpcServer {
public:
    RpcServer() = default;
    ~RpcServer() = default;

    bool start(const std::string& address, int port);
    void stop();
};
