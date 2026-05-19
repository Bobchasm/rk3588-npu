#pragma once

#include <string>

class WorkerRpcClient {
public:
    WorkerRpcClient() = default;
    ~WorkerRpcClient() = default;

    bool connect(const std::string& endpoint);
    bool ping();
};
