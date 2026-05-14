// ============================================================
// qwen2_chat: 单轮对话 REPL（token id 级别）
//
// 协议（文本行）：
//   stdin  每行一条请求，空格分隔的 token id，例如：
//            "151644 872 108386 151645 198 151644 77091 198"
//          另外支持两个控制命令：
//            "EXIT"  退出
//            "RESET" 强制清空 KV Cache（一般由 Python 包装器控制）
//
//   stdout 每处理完一条请求输出：
//            "OK <gen_id1> <gen_id2> ... <gen_idM>"
//          遇到错误：
//            "ERR <message>"
//
// 设计目的：
//   模型只加载一次，常驻进程；文字级 tokenizer 在 Python 侧处理，
//   通过管道把 token id 送给本进程。用户可以在 Python 端实现
//   「真正 LLM 单轮对话」的交互。
//
// 命令行参数：
//   ./qwen2_chat <model_dir> [max_new_tokens]
//   - max_new_tokens 可选，默认 128
// ============================================================

#include "api/llm_engine.h"

#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

static LLMEngine* g_engine_ptr = nullptr;

static void sig_cleanup(int sig) {
    std::fprintf(stderr, "\n[信号%d] 释放 NPU 资源并退出\n", sig);
    if (g_engine_ptr) { g_engine_ptr->destroy(); g_engine_ptr = nullptr; }
    std::_Exit(0);
}

static std::vector<int> parse_ids(const std::string& line) {
    std::vector<int> ids;
    std::istringstream iss(line);
    int v;
    while (iss >> v) ids.push_back(v);
    return ids;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::fprintf(stderr, "Usage: %s <model_dir> [max_new_tokens]\n", argv[0]);
        return 1;
    }

    std::signal(SIGINT,  sig_cleanup);
    std::signal(SIGTERM, sig_cleanup);

    std::string model_dir       = argv[1];
    int         max_new_tokens  = (argc >= 3) ? std::atoi(argv[2]) : 128;

    std::fprintf(stderr, "[qwen2_chat] 加载模型: %s\n", model_dir.c_str());

    LLMEngine engine;
    g_engine_ptr = &engine;

    if (!engine.load(model_dir)) {
        std::fprintf(stderr, "[qwen2_chat] 模型加载失败\n");
        return 1;
    }
    std::fprintf(stderr, "[qwen2_chat] 就绪，等待 stdin 输入（每行一条请求；EXIT 退出）\n");
    std::fprintf(stderr, "[qwen2_chat] max_new_tokens = %d\n", max_new_tokens);

    // 就绪信号：让 Python 包装器知道可以开始送请求
    std::printf("READY\n");
    std::fflush(stdout);

    GenerationConfig cfg;
    cfg.max_new_tokens = max_new_tokens;

    std::string line;
    while (std::getline(std::cin, line)) {
        // 去除首尾空白
        size_t a = line.find_first_not_of(" \t\r\n");
        size_t b = line.find_last_not_of(" \t\r\n");
        if (a == std::string::npos) continue;
        line = line.substr(a, b - a + 1);

        if (line == "EXIT")  break;
        if (line == "RESET") { engine.reset(); std::printf("OK\n"); std::fflush(stdout); continue; }

        std::vector<int> input_ids = parse_ids(line);
        if (input_ids.empty()) {
            std::printf("ERR empty_input\n");
            std::fflush(stdout);
            continue;
        }

        // 单轮对话：每条请求都是独立会话，先清空 KV Cache
        engine.reset();

        auto result = engine.generate(input_ids, cfg, nullptr);

        // 输出：OK <ids...>
        std::printf("OK");
        for (int id : result.output_ids) std::printf(" %d", id);
        std::printf("\n");
        // 统计写到 stderr，不会干扰 stdout 协议
        std::fprintf(stderr,
                     "[qwen2_chat] prefill=%d tok %.0f ms, decode=%d tok %.0f ms"
                     " (%.2f tok/s)%s\n",
                     result.prefill_tokens, result.prefill_ms,
                     result.decode_tokens,  result.decode_ms,
                     result.decode_tokens > 0
                        ? result.decode_tokens / (result.decode_ms / 1000.0f) : 0.0f,
                     result.hit_stop ? " [hit_stop]" : "");
        std::fflush(stdout);
    }

    engine.destroy();
    std::fprintf(stderr, "[qwen2_chat] 正常退出\n");
    return 0;
}
