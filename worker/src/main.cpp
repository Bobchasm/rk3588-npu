#include "qwen2.h"
#include <cstdio>
#include <cstring>
#include <vector>
#include <string>
#include <algorithm>
#include <csignal>
#include <sys/time.h>

static Qwen2Model* g_model_ptr = nullptr;

static void sig_cleanup(int sig) {
    fprintf(stderr, "\n[信号%d] 正在释放 NPU handles...\n", sig);
    if (g_model_ptr) {
        g_model_ptr->layers.clear();   // 触发 ~TransformerLayer -> ~NpuLinear
        g_model_ptr->lm_head.destroy();
        g_model_ptr = nullptr;
    }
    _exit(0);
}

static int64_t now_us() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec * 1000000LL + tv.tv_usec;
}

// ============================================================
// 用法：
//   ./qwen2_demo <model_dir> <token_id1> [token_id2 ...]
//
// 例：先用 Python 获取 token id，再传入
//   python3 -c "from transformers import AutoTokenizer; \
//     t=AutoTokenizer.from_pretrained('model/Qwen1.5B'); \
//     ids=t.encode('你好'); print(' '.join(map(str,ids)))"
//   # 假设输出：151644 872 1773 151645
//   ./qwen2_demo ../model/Qwen1.5B 151644 872 1773 151645
// ============================================================
int main(int argc, char* argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model_dir> <token_id> [token_id ...]\n", argv[0]);
        return 1;
    }

    std::signal(SIGINT,  sig_cleanup);
    std::signal(SIGTERM, sig_cleanup);

    std::string model_dir = argv[1];
    std::vector<int> input_ids;
    for (int i = 2; i < argc; ++i)
        input_ids.push_back(atoi(argv[i]));

    printf("模型目录: %s\n", model_dir.c_str());
    printf("输入 token 数量: %d\n", (int)input_ids.size());
    for (int id : input_ids) printf("  %d\n", id);

    Qwen2Model model;
    g_model_ptr = &model;
    printf("\n开始加载模型...\n");
    int64_t t0 = now_us();
    if (!model.load(model_dir)) {
        fprintf(stderr, "模型加载失败\n");
        return 1;
    }
    printf("模型加载完成，耗时 %.1f s\n", (now_us() - t0) / 1e6f);

    const int max_new_tokens = 10;
    std::vector<int> all_tokens = input_ids;

    // ---- Prefill ----
    model.reset_kv_cache();
    printf("\n[Prefill] %d 个输入 token...\n", (int)input_ids.size());
    int64_t ts0 = now_us();
    auto logits = model.forward(input_ids);
    printf("[Prefill] 耗时 %.0f ms\n", (now_us() - ts0) / 1e3f);

    int next_id = greedy_sample(logits);

    // ---- Decode ----
    printf("\n[Decode]\n");
    int64_t t_infer_start = now_us();
    int generated = 0;
    for (int step = 0; step < max_new_tokens; ++step) {
        if (next_id == 151645 || next_id == 151643) {
            printf("step %2d: [EOS]\n", step);
            break;
        }

        all_tokens.push_back(next_id);
        ++generated;

        int64_t ts = now_us();
        logits = model.forward({next_id});   // 单 token decode
        int64_t elapsed = now_us() - ts;

        std::vector<std::pair<float, int>> top;
        for (int i = 0; i < (int)logits.size(); ++i)
            top.push_back({logits[i], i});
        std::partial_sort(top.begin(), top.begin() + 5, top.end(),
            [](const std::pair<float,int>& a, const std::pair<float,int>& b){ return a.first > b.first; });
        printf("step %2d (%.0f ms): emit=%d  top5=", step, elapsed / 1e3f, next_id);
        for (int i = 0; i < 5; ++i)
            printf("[%d:%.2f] ", top[i].second, top[i].first);
        printf("\n");

        next_id = greedy_sample(logits);
    }

    float total_s = (now_us() - t_infer_start) / 1e6f;
    printf("\n生成完成：%d tokens，总耗时 %.1f s，平均 %.2f tok/s\n",
           generated, total_s, generated > 0 ? generated / total_s : 0.0f);

    printf("\n输出 token ids:");
    for (int i = (int)input_ids.size(); i < (int)all_tokens.size(); ++i)
        printf(" %d", all_tokens[i]);
    printf("\n");

    return 0;
}
