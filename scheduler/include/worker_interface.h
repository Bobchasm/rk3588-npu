#pragma once

#include "scheduler_types.h"
#include <memory>
#include <string>
#include <vector>

#ifdef SCHEDULER_USE_WORKER_CORE
#include <api/llm_engine.h>
#endif

class WorkerInterface {
public:
    virtual ~WorkerInterface() = default;

    virtual bool register_node(const WorkerId& id) = 0;
    virtual void set_worker_id(const WorkerId& id) = 0;
    virtual WorkerId worker_id() const = 0;

    virtual bool supports_full_model() const = 0;
    virtual bool supports_stage() const = 0;

    // 当前单 worker 场景走 full-model 生成。
    // 后续若拆分为 head + 多个 stage，也仍可以由调度器先构造这类请求再做路由拆分。
    virtual GenerationResult generate_tokens(
        const GenerateTokensRequest& req,
        TokenCallback on_token = nullptr) = 0;

    // 预留给后续中间 stage worker 的 hidden-state 转发执行接口。
    virtual StageForwardResponse forward_stage(const StageForwardRequest& req) = 0;
};

std::unique_ptr<WorkerInterface> make_local_worker(const std::string& model_dir);
std::unique_ptr<WorkerInterface> make_remote_worker(const std::string& endpoint,
                                                    const WorkerId& worker_id);
