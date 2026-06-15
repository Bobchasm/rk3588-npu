#include "tokenizer/tokenizer_adapter.h"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <utility>

namespace {

#ifndef SCHEDULER_REPO_ROOT
#define SCHEDULER_REPO_ROOT "."
#endif

std::string tokenizer_script_path() {
    return std::string(SCHEDULER_REPO_ROOT) + "/scheduler/tools/tokenizer.py";
}

std::string escape_shell_arg(const std::string& arg) {
    std::string escaped;
    escaped.reserve(arg.size() + 2);
    for (char ch : arg) {
        if (ch == '\\' || ch == '"') {
            escaped.push_back('\\');
        }
        escaped.push_back(ch);
    }
    return "\"" + escaped + "\"";
}

bool run_command(const std::string& command, std::string& output) {
    FILE* pipe = popen(command.c_str(), "r");
    if (!pipe) {
        return false;
    }

    output.clear();
    char buffer[256];
    while (fgets(buffer, sizeof(buffer), pipe)) {
        output += buffer;
    }

    const int status = pclose(pipe);
    return status == 0;
}

std::string json_escape(const std::string& text) {
    std::ostringstream oss;
    for (char ch : text) {
        switch (ch) {
        case '\\': oss << "\\\\"; break;
        case '"': oss << "\\\""; break;
        case '\n': oss << "\\n"; break;
        case '\r': oss << "\\r"; break;
        case '\t': oss << "\\t"; break;
        default: oss << ch; break;
        }
    }
    return oss.str();
}

}  // namespace

TokenizerAdapter::TokenizerAdapter(std::string model_dir)
    : model_dir_(std::move(model_dir)) {}

bool TokenizerAdapter::encode(const std::string& text, std::vector<int>& out_ids) const {
    out_ids.clear();

    const std::string cmd =
        "python3 " + escape_shell_arg(tokenizer_script_path()) + " encode " +
        escape_shell_arg(model_dir_) + " " +
        escape_shell_arg(text);

    std::string raw;
    if (!run_command(cmd, raw)) {
        std::cerr << "[TokenizerAdapter] encode command failed" << std::endl;
        return false;
    }

    std::istringstream iss(raw);
    int token_id = 0;
    while (iss >> token_id) {
        out_ids.push_back(token_id);
    }
    return !out_ids.empty();
}

bool TokenizerAdapter::encode_chat(const std::vector<ChatMessage>& messages,
                                   std::vector<int>& out_ids) const {
    out_ids.clear();

    std::ostringstream payload;
    payload << "[";
    for (size_t i = 0; i < messages.size(); ++i) {
        if (i != 0) {
            payload << ",";
        }
        payload << "{\"role\":\"" << json_escape(messages[i].role)
                << "\",\"content\":\"" << json_escape(messages[i].text) << "\"}";
    }
    payload << "]";

    const std::string cmd =
        "python3 " + escape_shell_arg(tokenizer_script_path()) + " encode_chat " +
        escape_shell_arg(model_dir_) + " " +
        escape_shell_arg(payload.str());

    std::string raw;
    if (!run_command(cmd, raw)) {
        std::cerr << "[TokenizerAdapter] encode_chat command failed" << std::endl;
        return false;
    }

    std::istringstream iss(raw);
    int token_id = 0;
    while (iss >> token_id) {
        out_ids.push_back(token_id);
    }
    return !out_ids.empty();
}

bool TokenizerAdapter::decode(const std::vector<int>& ids, std::string& out_text) const {
    std::ostringstream ids_stream;
    for (size_t i = 0; i < ids.size(); ++i) {
        if (i != 0) {
            ids_stream << ' ';
        }
        ids_stream << ids[i];
    }

    const std::string cmd =
        "python3 " + escape_shell_arg(tokenizer_script_path()) + " decode " +
        escape_shell_arg(model_dir_) + " " +
        escape_shell_arg(ids_stream.str());

    if (!run_command(cmd, out_text)) {
        std::cerr << "[TokenizerAdapter] decode command failed" << std::endl;
        return false;
    }

    if (!out_text.empty() && out_text.back() == '\n') {
        out_text.pop_back();
    }
    return true;
}
