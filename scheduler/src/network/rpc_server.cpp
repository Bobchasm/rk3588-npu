#include "network/rpc_server.h"
#include <iostream>

bool RpcServer::start(const std::string& address, int port) {
    std::cerr << "[RpcServer] start(" << address << ", " << port << ") stub" << std::endl;
    return true;
}

void RpcServer::stop() {
    std::cerr << "[RpcServer] stop() stub" << std::endl;
}
