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
    bool ok = send_line(sock, "PING|0|0|") && recv_line(sock, response) && response == "PONG|0";
    close(sock);
    return ok;
}

bool RpcClient::send_prefill(const PrefillRequest& req, PrefillResponse& resp) {
    int sock;
    if (!open_connection(endpoint_, sock)) return false;
    std::ostringstream oss;
    oss << "PREFILL|" << req.request_id << "|" << req.max_new_tokens << "|";
    for (size_t i = 0; i < req.input_ids.size(); ++i) {
        if (i) oss << ',';
        oss << req.input_ids[i];
    }
    if (!send_line(sock, oss.str())) {
        close(sock);
        return false;
    }

    std::string response;
    bool ok = recv_line(sock, response);
    close(sock);
    if (!ok) return false;

    std::istringstream rs(response);
    std::string status;
    std::getline(rs, status, '|');
    if (status != "OK") {
        std::string error_msg;
        std::getline(rs, error_msg, '|');
        resp.status = RequestStatus::ERROR;
        resp.message = error_msg;
        return true;
    }

    std::string request_id_str;
    std::getline(rs, request_id_str, '|');
    resp.request_id = std::stoull(request_id_str);
    resp.status = RequestStatus::OK;
    return true;
}

bool RpcClient::send_generate(const SessionId& session_id,
                              RequestId request_id,
                              const std::vector<int>& input_ids,
                              int max_new_tokens,
                              GenerationResult& result) {
    int sock;
    if (!open_connection(endpoint_, sock)) return false;
    std::ostringstream oss;
    oss << "GENERATE|" << request_id << "|" << max_new_tokens << "|";
    for (size_t i = 0; i < input_ids.size(); ++i) {
        if (i) oss << ',';
        oss << input_ids[i];
    }
    if (!send_line(sock, oss.str())) {
        close(sock);
        return false;
    }

    std::string response;
    bool ok = recv_line(sock, response);
    close(sock);
    if (!ok) return false;

    std::istringstream rs(response);
    std::string status;
    std::getline(rs, status, '|');
    if (status != "OK") {
        std::string err_msg;
        std::getline(rs, err_msg, '|');
        result.error_message = err_msg;
        return false;
    }

    std::string request_id_str;
    std::getline(rs, request_id_str, '|');
    std::string prefill_ms_str;
    std::getline(rs, prefill_ms_str, '|');
    std::string decode_ms_str;
    std::getline(rs, decode_ms_str, '|');
    std::string ids_str;
    std::getline(rs, ids_str, '|');

    result.prefill_ms = std::stof(prefill_ms_str);
    result.decode_ms = std::stof(decode_ms_str);
    result.output_ids.clear();
    std::istringstream iss(ids_str);
    std::string id_token;
    while (std::getline(iss, id_token, ',')) {
        if (id_token.empty()) continue;
        result.output_ids.push_back(std::stoi(id_token));
    }
    return true;
}
