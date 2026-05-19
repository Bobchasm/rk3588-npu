#include "network/worker_service.h"
#include "network/rpc_server.h"
#include <api/llm_engine.h>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

static std::vector<int> parse_ids(const std::string& token_list) {
    std::vector<int> ids;
    std::istringstream iss(token_list);
    std::string token;
    while (std::getline(iss, token, ',')) {
        if (token.empty()) continue;
        ids.push_back(std::stoi(token));
    }
    return ids;
}

static std::string serialize_ids(const std::vector<int>& ids) {
    std::ostringstream oss;
    for (size_t i = 0; i < ids.size(); ++i) {
        if (i) oss << ',';
        oss << ids[i];
    }
    return oss.str();
}

WorkerService::WorkerService()
    : loaded_(false)
{
}

WorkerService::~WorkerService() {
    rpc_server_.stop();
}

bool WorkerService::register_service(const std::string& address,
                                     int port,
                                     const std::string& model_dir) {
    if (!engine_.load(model_dir)) {
        std::cerr << "[WorkerService] failed to load model: " << model_dir << std::endl;
        return false;
    }
    loaded_ = true;
    return rpc_server_.start(address, port, [this](const std::string& request) {
        return handle_request(request);
    });
}

std::string WorkerService::handle_request(const std::string& request) {
    if (!loaded_) {
        return "ERR|0|model-not-loaded";
    }

    std::vector<std::string> parts;
    std::istringstream iss(request);
    std::string part;
    while (std::getline(iss, part, '|')) {
        parts.push_back(part);
    }

    if (parts.size() < 4) {
        return "ERR|0|invalid-request-format";
    }

    const std::string& command = parts[0];
    uint64_t request_id = std::stoull(parts[1]);
    int max_new_tokens = std::stoi(parts[2]);
    std::vector<int> input_ids = parse_ids(parts[3]);

    if (command == "PREFILL") {
        engine_.reset();
        GenerationConfig cfg;
        cfg.max_new_tokens = max_new_tokens;
        engine_.generate(input_ids, cfg, nullptr);
        return "OK|" + std::to_string(request_id) + "|0|0|";
    }

    if (command == "GENERATE") {
        engine_.reset();
        GenerationConfig cfg;
        cfg.max_new_tokens = max_new_tokens;
        GenerationResult result = engine_.generate(input_ids, cfg, nullptr);
        std::string payload = serialize_ids(result.output_ids);
        std::ostringstream resp;
        resp << "OK|" << request_id << "|" << result.prefill_ms << "|"
             << result.decode_ms << "|" << payload;
        return resp.str();
    }

    return "ERR|" + std::to_string(request_id) + "|unsupported-command";
}
