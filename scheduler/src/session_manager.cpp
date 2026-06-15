#include "session_manager.h"

#include <sstream>

SessionManager::SessionManager() = default;

SessionId SessionManager::create_session(const std::string& system_prompt) {
    SessionId session_id = "session-" + std::to_string(next_session_index_++);
    SessionData data;
    if (!system_prompt.empty()) {
        data.history.push_back({"system", system_prompt});
    }
    sessions_.emplace(session_id, std::move(data));
    return session_id;
}

bool SessionManager::session_exists(const SessionId& session_id) const {
    return sessions_.find(session_id) != sessions_.end();
}

bool SessionManager::append_user_message(const SessionId& session_id, const std::string& text) {
    auto it = sessions_.find(session_id);
    if (it == sessions_.end()) return false;
    it->second.history.push_back({"user", text});
    return true;
}

bool SessionManager::append_assistant_message(const SessionId& session_id, const std::string& text) {
    auto it = sessions_.find(session_id);
    if (it == sessions_.end()) return false;
    it->second.history.push_back({"assistant", text});
    return true;
}

std::string SessionManager::get_prompt(const SessionId& session_id, int max_tokens) const {
    auto it = sessions_.find(session_id);
    if (it == sessions_.end()) return "";

    std::ostringstream oss;
    for (const auto& message : it->second.history) {
        oss << "<|im_start|>" << message.role << "\n"
            << message.text
            << "<|im_end|>\n";
    }
    // Qwen chat template expects the next assistant turn prefix without im_end.
    oss << "<|im_start|>assistant\n";
    std::string prompt = oss.str();
    if (max_tokens > 0 && (int)prompt.size() > max_tokens) {
        prompt = prompt.substr(prompt.size() - max_tokens);
    }
    return prompt;
}

std::vector<ChatMessage> SessionManager::get_history(const SessionId& session_id) const {
    auto it = sessions_.find(session_id);
    if (it == sessions_.end()) {
        return {};
    }
    return it->second.history;
}
