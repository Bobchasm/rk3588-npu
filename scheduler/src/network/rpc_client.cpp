#include "network/rpc_client.h"

#include <arpa/inet.h>
#include <cerrno>
#include <cstring>
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>

#include <iostream>
#include <sstream>

static bool parse_endpoint(const std::string& endpoint,
                           std::string& host,
                           std::string& port) {
    auto pos = endpoint.find(':');
    if (pos == std::string::npos) return false;
    host = endpoint.substr(0, pos);
    port = endpoint.substr(pos + 1);
    return !host.empty() && !port.empty();
}

static bool open_connection(const std::string& endpoint, int& out_fd) {
    std::string host, port;
    if (!parse_endpoint(endpoint, host, port)) {
        std::cerr << "[RpcClient] invalid endpoint: " << endpoint << std::endl;
        return false;
    }

    struct addrinfo hints{};
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;

    struct addrinfo* addr = nullptr;
    int err = getaddrinfo(host.c_str(), port.c_str(), &hints, &addr);
    if (err != 0) {
        std::cerr << "[RpcClient] getaddrinfo failed: " << gai_strerror(err) << std::endl;
        return false;
    }

    int fd = socket(addr->ai_family, addr->ai_socktype, addr->ai_protocol);
    if (fd < 0) {
        std::cerr << "[RpcClient] socket failed: " << strerror(errno) << std::endl;
        freeaddrinfo(addr);
        return false;
    }

    if (::connect(fd, addr->ai_addr, addr->ai_addrlen) < 0) {
        std::cerr << "[RpcClient] connect failed: " << strerror(errno) << std::endl;
        close(fd);
        freeaddrinfo(addr);
        return false;
    }

    freeaddrinfo(addr);
    out_fd = fd;
    return true;
}

static bool send_line(int sock, const std::string& line) {
    std::string payload = line + "\n";
    const char* data = payload.data();
    size_t remaining = payload.size();
    while (remaining > 0) {
        ssize_t sent = ::send(sock, data, remaining, 0);
        if (sent <= 0) return false;
        data += sent;
        remaining -= sent;
    }
    return true;
}

static bool recv_line(int sock, std::string& out) {
    out.clear();
    char buf[1024];
    while (true) {
        ssize_t n = recv(sock, buf, sizeof(buf), 0);
        if (n <= 0) return false;
        out.append(buf, n);
        auto pos = out.find('\n');
        if (pos != std::string::npos) {
            out.resize(pos);
            return true;
        }
    }
}

RpcClient::RpcClient() = default;
RpcClient::~RpcClient() = default;

bool RpcClient::connect(const std::string& endpoint) {
    endpoint_ = endpoint;
    return true;
}

bool RpcClient::send_ping() {
    int sock;
    if (!open_connection(endpoint_, sock)) return false;
    std::string response;
    bool ok = send_line(sock, distributed::serialize_ping_request()) &&
              recv_line(sock, response) &&
              response == "PONG";
    close(sock);
    return ok;
}

bool RpcClient::send_reset_cache() {
    int sock;
    if (!open_connection(endpoint_, sock)) return false;
    std::string response;
    bool ok = send_line(sock, distributed::serialize_reset_cache_request()) &&
              recv_line(sock, response) &&
              response == "OK";
    close(sock);
    return ok;
}

bool RpcClient::send_generate_tokens(const GenerateTokensRequest& req,
                                     GenerateTokensResponse& resp) {
    int sock;
    if (!open_connection(endpoint_, sock)) return false;
    if (!send_line(sock, distributed::serialize_generate_request(req))) {
        close(sock);
        return false;
    }

    std::string response;
    bool ok = recv_line(sock, response);
    close(sock);
    if (!ok) return false;
    return distributed::deserialize_generate_response(response, resp);
}

bool RpcClient::send_tokens_to_hidden(const distributed::TokensToHiddenRequest& req,
                                      distributed::TokensToHiddenResponse& resp) {
    int sock;
    if (!open_connection(endpoint_, sock)) return false;
    if (!send_line(sock, distributed::serialize_tokens_to_hidden_request(req))) {
        close(sock);
        return false;
    }

    std::string response;
    bool ok = recv_line(sock, response);
    close(sock);
    if (!ok) return false;
    return distributed::deserialize_tokens_to_hidden_response(response, resp);
}

bool RpcClient::send_forward_stage(const StageForwardRequest& req,
                                   StageForwardResponse& resp) {
    int sock;
    if (!open_connection(endpoint_, sock)) return false;
    if (!send_line(sock, distributed::serialize_stage_request(req))) {
        close(sock);
        return false;
    }

    std::string response;
    bool ok = recv_line(sock, response);
    close(sock);
    if (!ok) return false;
    return distributed::deserialize_stage_response(response, resp);
}

bool RpcClient::send_hidden_to_token(const distributed::HiddenToTokenRequest& req,
                                     distributed::HiddenToTokenResponse& resp) {
    int sock;
    if (!open_connection(endpoint_, sock)) return false;
    if (!send_line(sock, distributed::serialize_hidden_to_token_request(req))) {
        close(sock);
        return false;
    }

    std::string response;
    bool ok = recv_line(sock, response);
    close(sock);
    if (!ok) return false;
    return distributed::deserialize_hidden_to_token_response(response, resp);
}
