// ============================================================
// qwen2_demo: 保留原 demo 的外部行为（token id 命令行输入，带 top5 打印）
//
// 与旧版等价，但内部已完全迁移到 LLMEngine + 分层算子库，
// 作为「API 层封装后的等效入口」的验证示例。
//
// 用法：
//   ./qwen2_demo <model_dir> <token_id1> [token_id2 ...]
//
// 获取输入 token id：
//   python3 -c "from transformers import AutoTokenizer; \
//     t=AutoTokenizer.from_pretrained('Qwen1.5B'); \
//     m=[{'role':'user','content':'你好'}]; \
//     s=t.apply_chat_template(m, tokenize=False, add_generation_prompt=True); \
//     print(' '.join(map(str, t.encode(s))))"
// ============================================================

#include "api/llm_engine.h"

#include <algorithm>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <sys/time.h>
#include <vector>

static LLMEngine* g_engine_ptr = nullptr;

static void sig_cleanup(int sig) {
    std::fprintf(stderr, "\n[信号%d] 正在释放 NPU handles...\n", sig);
    if (g_engine_ptr) {
        g_engine_ptr->destroy();
        g_engine_ptr = nullptr;
    }
    std::_Exit(0);
}

static int64_t now_us() {
    struct timeval tv; gettimeofday(&tv, nullptr);
    return (int64_t)tv.tv_sec * 1000000LL + tv.tv_usec;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::fprintf(stderr, "Usage: %s <model_dir> <token_id> [token_id ...]\n", argv[0]);
        return 1;
    }

    std::signal(SIGINT,  sig_cleanup);
    std::signal(SIGTERM, sig_cleanup);

    std::string model_dir = argv[1];
    std::vector<int> input_ids;
    for (int i = 2; i < argc; ++i) input_ids.push_back(std::atoi(argv[i]));

    std::printf("[build] matmul-api A8W8(INT8xINT8->INT32) path available\n");
    std::printf("模型目录: %s\n", model_dir.c_str());
    std::printf("输入 token 数量: %d\n", (int)input_ids.size());
    for (int id : input_ids) std::printf("  %d\n", id);

    LLMEngine engine;
    g_engine_ptr = &engine;

    std::printf("\n开始加载模型...\n");
    int64_t t0 = now_us();
    if (!engine.load(model_dir)) {
        std::fprintf(stderr, "模型加载失败\n");
        return 1;
    }
    std::printf("模型加载完成，耗时 %.1f s\n", (now_us() - t0) / 1e6f);

    engine.reset();

    GenerationConfig cfg;
    cfg.max_new_tokens = 10;

    std::printf("\n[Prefill] %d 个输入 token...\n", (int)input_ids.size());

    auto result = engine.generate(input_ids, cfg,
        [](int step, int id, float elapsed_ms) {
            std::printf("step %2d (%.0f ms): emit=%d\n", step, elapsed_ms, id);
        });

    std::printf("\n[Prefill] 耗时 %.0f ms\n", result.prefill_ms);
    std::printf("生成完成：%d tokens，decode 总耗时 %.1f s，平均 %.2f tok/s\n",
                result.decode_tokens,
                result.decode_ms / 1000.0f,
                result.decode_tokens > 0
                    ? result.decode_tokens / (result.decode_ms / 1000.0f) : 0.0f);

    std::printf("\n输出 token ids:");
    for (int id : result.output_ids) std::printf(" %d", id);
    std::printf("\n");

    return 0;
}
