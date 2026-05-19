#include "coordinator.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

static std::string escape_shell_arg(const std::string& arg) {
    std::string escaped = "";
    for (char c : arg) {
        if (c == '\\' || c == '"') {
            escaped += '\\';
        }
        escaped.push_back(c);
    }
    return "\"" + escaped + "\"";
}

static bool run_command(const std::string& command, std::string& output) {
    FILE* pipe = popen(command.c_str(), "r");
    if (!pipe) return false;
    char buffer[256];
    output.clear();
    while (fgets(buffer, sizeof(buffer), pipe)) {
        output += buffer;
    }
    int status = pclose(pipe);
    return status == 0;
}

static bool encode_text(const std::string& model_dir,
                        const std::string& text,
                        std::vector<int>& out_ids) {
    std::string cmd = "python3 ../tools/tokenizer.py encode " +
                      escape_shell_arg(model_dir) + " " +
                      escape_shell_arg(text);
    std::string raw;
    if (!run_command(cmd, raw)) {
        std::cerr << "[scheduler_cli] tokenizer failed" << std::endl;
        return false;
    }
    std::istringstream iss(raw);
    int id;
    while (iss >> id) {
        out_ids.push_back(id);
    }
    return !out_ids.empty();
}

static std::string decode_ids(const std::string& model_dir,
                              const std::vector<int>& ids) {
    std::ostringstream arg;
    for (size_t i = 0; i < ids.size(); ++i) {
        if (i) arg << " ";
        arg << ids[i];
    }
    std::string cmd = "python3 scheduler/tools/tokenizer.py decode " +
                      escape_shell_arg(model_dir) + " " +
                      escape_shell_arg(arg.str());
    std::string raw;
    if (!run_command(cmd, raw)) {
        return "<decode failed>";
    }
    if (!raw.empty() && raw.back() == '\n') raw.pop_back();
    return raw;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model_dir> [remote_endpoint]" << std::endl;
        std::cerr << "Example: " << argv[0] << " qwen1.5b-instruct 127.0.0.1:5001" << std::endl;
        return 1;
    }

    std::string model_dir = argv[1];
    std::unique_ptr<WorkerInterface> worker;
    if (argc >= 3) {
        std::string endpoint = argv[2];
        worker = make_remote_worker(endpoint, "remote_worker");
        if (!worker) {
            std::cerr << "Failed to create remote worker at " << endpoint << std::endl;
            return 1;
        }
        std::cerr << "Using remote worker: " << endpoint << std::endl;
    } else {
        worker = make_local_worker(model_dir);
    }

    Coordinator coordinator(std::move(worker));
    SessionId session_id = coordinator.create_session();

    std::cout << "Scheduler CLI started. Enter text to generate, or type /exit." << std::endl;
    while (true) {
        std::cout << "> ";
        std::string line;
        if (!std::getline(std::cin, line)) break;
        if (line == "/exit" || line == "/quit") break;
        if (line.empty()) continue;

        std::vector<int> input_ids;
        if (!encode_text(model_dir, line, input_ids)) {
            std::cerr << "Failed to encode input text." << std::endl;
            continue;
        }

        GenerationResult result = coordinator.submit_request(
            session_id,
            input_ids,
            10,
            [](int step, int id, float elapsed_ms) {
                std::cout << "[token] " << id << " (" << elapsed_ms << " ms)" << std::endl;
            });

        std::cout << "Generate result: " << result.output_ids.size() << " token(s)" << std::endl;
        if (!result.output_ids.empty()) {
            std::cout << "token ids:";
            for (int id : result.output_ids) {
                std::cout << " " << id;
            }
            std::cout << std::endl;
            std::cout << "decoded: " << decode_ids(model_dir, result.output_ids) << std::endl;
        }
        std::cout << "prefill_ms=" << result.prefill_ms
                  << " decode_ms=" << result.decode_ms << std::endl;
    }

    return 0;
}
