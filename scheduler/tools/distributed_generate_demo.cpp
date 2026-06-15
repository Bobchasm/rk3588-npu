#include "network/rpc_client.h"
#include "tokenizer/tokenizer_adapter.h"

#include <distributed/protocol.h>

#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

bool is_stop_token(int token_id) {
    return token_id == 151645 || token_id == 151643;
}

std::string join_ids(const std::vector<int>& ids) {
    std::string out;
    for (size_t i = 0; i < ids.size(); ++i) {
        if (i != 0) out += " ";
        out += std::to_string(ids[i]);
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0]
                  << " <model_dir> <head_endpoint> <tail_endpoint> <max_new_tokens> <text> [stage_endpoint ...]"
                  << std::endl;
        return 1;
    }

    const std::string model_dir = argv[1];
    const std::string head_endpoint = argv[2];
    const std::string tail_endpoint = argv[3];
    const int max_new_tokens = std::atoi(argv[4]);
    const std::string text = argv[5];

    std::vector<std::string> stage_endpoints;
    for (int i = 6; i < argc; ++i) {
        stage_endpoints.emplace_back(argv[i]);
    }

    TokenizerAdapter tokenizer(model_dir);
    std::vector<int> prompt_ids;
    if (!tokenizer.encode(text, prompt_ids)) {
        std::cerr << "tokenize failed" << std::endl;
        return 1;
    }

    RpcClient head_client;
    RpcClient tail_client;
    if (!head_client.connect(head_endpoint) || !tail_client.connect(tail_endpoint)) {
        std::cerr << "connect head/tail failed" << std::endl;
        return 1;
    }
    std::vector<RpcClient> stage_clients(stage_endpoints.size());
    for (size_t i = 0; i < stage_endpoints.size(); ++i) {
        if (!stage_clients[i].connect(stage_endpoints[i])) {
            std::cerr << "connect stage failed: " << stage_endpoints[i] << std::endl;
            return 1;
        }
    }

    std::vector<int> generated_ids;
    std::vector<int> current_input = prompt_ids;

    for (int step = 0; step < max_new_tokens; ++step) {
        distributed::TokensToHiddenRequest head_req;
        head_req.context.session_id = "distributed-demo";
        head_req.context.request_id = static_cast<uint64_t>(step + 1);
        head_req.context.route.pos_base = static_cast<int32_t>(prompt_ids.size() + generated_ids.size() - current_input.size());
        head_req.input_token_ids.assign(current_input.begin(), current_input.end());

        distributed::TokensToHiddenResponse head_resp;
        if (!head_client.send_tokens_to_hidden(head_req, head_resp) ||
            !distributed::is_success(head_resp.status)) {
            std::cerr << "head failed: " << head_resp.message << std::endl;
            return 1;
        }

        distributed::TensorPayload hidden = head_resp.output_tensor;
        int32_t pos_base = head_req.context.route.pos_base;
        for (size_t i = 0; i < stage_clients.size(); ++i) {
            distributed::StageForwardRequest stage_req;
            stage_req.context = head_req.context;
            stage_req.context.route.hop_index = static_cast<int32_t>(i + 1);
            stage_req.context.route.pos_base = pos_base;
            stage_req.input_tensor = hidden;

            distributed::StageForwardResponse stage_resp;
            if (!stage_clients[i].send_forward_stage(stage_req, stage_resp) ||
                !distributed::is_success(stage_resp.status)) {
                std::cerr << "stage failed: " << stage_resp.message << std::endl;
                return 1;
            }
            hidden = stage_resp.output_tensor;
        }

        distributed::HiddenToTokenRequest tail_req;
        tail_req.context = head_req.context;
        tail_req.context.route.hop_index = static_cast<int32_t>(stage_clients.size() + 1);
        tail_req.context.route.pos_base = pos_base;
        tail_req.input_tensor = hidden;

        distributed::HiddenToTokenResponse tail_resp;
        if (!tail_client.send_hidden_to_token(tail_req, tail_resp) ||
            !distributed::is_success(tail_resp.status)) {
            std::cerr << "tail failed: " << tail_resp.message << std::endl;
            return 1;
        }

        const int token_id = tail_resp.output_token_id;
        generated_ids.push_back(token_id);
        std::cerr << "step=" << step << " token_id=" << token_id << std::endl;
        if (is_stop_token(token_id)) {
            break;
        }
        current_input.assign(1, token_id);
    }

    std::string output_text;
    if (!tokenizer.decode(generated_ids, output_text)) {
        std::cerr << "decode failed, ids=" << join_ids(generated_ids) << std::endl;
        return 1;
    }

    std::cout << output_text << std::endl;
    return 0;
}
