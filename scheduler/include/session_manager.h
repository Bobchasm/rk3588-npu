#pragma once

#include "scheduler_types.h"

#include <map>
#include <string>
#include <vector>

struct ChatMessage {
    std::string role;
    std::string text;
};

class SessionManager {
public:
    SessionManager();
    SessionId create_session(const std::string& system_prompt = "");
    bool session_exists(const SessionId& session_id) const;
    bool append_user_message(const SessionId& session_id, const std::string& text);
    bool append_assistant_message(const SessionId& session_id, const std::string& text);
    std::string get_prompt(const SessionId& session_id, int max_tokens = 2048) const;
    std::vector<ChatMessage> get_history(const SessionId& session_id) const;

private:
    struct SessionData {
        std::vector<ChatMessage> history;
    };

    std::map<SessionId, SessionData> sessions_;
    RequestId next_session_index_ = 1;
};
