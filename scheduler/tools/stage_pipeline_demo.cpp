#include "network/rpc_client.h"
#include "scheduler_types.h"

#include <cstdlib>
#include <iostream>
#include <memory>
#include <cstring>
#include <string>
#include <vector>

namespace {

bool parse_endpoint(const std::string& text, std::string& host, int& port) {
    const auto pos = text.find(':');
    if (pos == std::string::npos) {
        return false;
    }
    host = text.substr(0, pos);
    port = std::atoi(text.substr(pos + 1).c_str());
    return port > 0;
}

std::vector<float> make_dummy_hidden(int seq, int hidden) {
    std::vector<float> out(static_cast<size_t>(seq) * hidden);
    for (int i = 0; i < seq * hidden; ++i) {
        const float v = static_cast<float>((i % 17) - 8) * 0.03125f;
        out[static_cast<size_t>(i)] = v;
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <seq> <hidden> <endpoint1> [endpoint2 ...]" << std::endl;
        return 1;
    }

    const int seq = std::atoi(argv[1]);
    const int hidden = std::atoi(argv[2]);
    if (seq <= 0 || hidden <= 0) {
        std::cerr << "invalid seq/hidden" << std::endl;
        return 1;
    }

    std::vector<std::string> endpoints;
    for (int i = 3; i < argc; ++i) {
        endpoints.emplace_back(argv[i]);
    }
    if (endpoints.empty()) {
        std::cerr << "no stage endpoints provided" << std::endl;
        return 1;
    }

    std::vector<float> hidden_state = make_dummy_hidden(seq, hidden);
    distributed::StageForwardRequest req;
    req.context.session_id = "stage-demo";
    req.context.request_id = 1;
    req.context.route.pos_base = 0;
    req.input_tensor.dtype = distributed::TensorDataType::kFloat32;
    req.input_tensor.shape = {seq, hidden};
    req.input_tensor.bytes.resize(hidden_state.size() * sizeof(float));
    std::memcpy(req.input_tensor.bytes.data(), hidden_state.data(), req.input_tensor.bytes.size());

    distributed::StageForwardResponse resp;
    for (size_t i = 0; i < endpoints.size(); ++i) {
        RpcClient client;
        if (!client.connect(endpoints[i])) {
            std::cerr << "connect failed: " << endpoints[i] << std::endl;
            return 1;
        }
        if (!client.send_forward_stage(req, resp)) {
            std::cerr << "stage rpc failed: " << endpoints[i] << std::endl;
            return 1;
        }
        if (!distributed::is_success(resp.status)) {
            std::cerr << "stage error: " << resp.message << std::endl;
            return 1;
        }
        req.context.route.pos_base += seq;
        req.input_tensor = resp.output_tensor;
    }

    std::cout << "stage pipeline ok, output_bytes=" << resp.output_tensor.bytes.size() << std::endl;
    return 0;
}
