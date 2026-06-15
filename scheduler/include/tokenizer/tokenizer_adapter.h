#pragma once

#include "session_manager.h"

#include <string>
#include <vector>

// TokenizerAdapter 负责把调度器的“文本世界”和 worker 的“token id 世界”隔离开。
// 当前实现复用项目里的 Python tokenizer 脚本，后续若改为常驻 tokenizer 服务，
// 只需要替换这一层，不需要改 Coordinator / WorkerInterface。
class TokenizerAdapter {
public:
    explicit TokenizerAdapter(std::string model_dir);

    bool encode(const std::string& text, std::vector<int>& out_ids) const;
    bool encode_chat(const std::vector<ChatMessage>& messages, std::vector<int>& out_ids) const;
    bool decode(const std::vector<int>& ids, std::string& out_text) const;

private:
    std::string model_dir_;
};
