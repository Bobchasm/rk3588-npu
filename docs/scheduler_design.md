# 调度器与分布式 Worker 实现计划

## 1 目标

本计划针对“Coordinator 中心化调度、以调度器为核心”的分布式方案。目标是：

- 先实现一个可运行的中心化调度器（Coordinator）+ 本地 Worker 的调度链路；
- 预留出后续接入多会话、多 worker、stage/ shard 调度的结构；
- 保持 `worker/` 内部网络层独立，调度器在项目根目录独立；
- 设计模式清晰、模块化、可扩展、分层明确。

## 2 核心设计原则

### 2.1 分层与职责

- `Scheduler / Coordinator`：
  - 会话管理、请求队列、负载均衡、节点注册、路由决策；
  - 负责高层“用户请求 -> 生成任务”的调度；
  - 不直接承担模型前向计算。

- `Worker`：
  - 负责实际推理与运行时状态（KV cache、hidden state、权重）；
  - 只暴露 RPC 接口，不暴露调度逻辑；
  - 具体执行 `Prefill` / `Stage` / `Decode` / `Generate` 等。

- `LLMEngine`：
  - 继续保留作为本地 full-model worker 的包装器；
  - 作为 `Worker` 实现中的本机推理后端；
  - 可被 `LocalWorker` 或 `RPCWorker` 复用。

### 2.2 模块化与架构

- 每个模块文件功能专一；
- 采用接口抽象和适配器模式；
- 以 `WorkerInterface` 抽象不同 worker 类型；
- 以 `Scheduler` 抽象调度策略；
- 以 `SessionManager` 抽象会话历史与 prompt 管理；
- 以 `Transport` / `RpcClient` / `RpcServer` 抽象网络层。

### 2.3 不要混在一起写 Head/后续 Worker 逻辑

- Head worker 的输入与后续 worker 的输入本质不同：
  - Head worker 处理 `input_ids` / prompt prefill；
  - Stage worker 处理 `hidden_state` / stage tensor；
- 实现在 `worker/` 侧要严格分离：
  - `GenerateHandler` / `PrefillHandler` 负责 head worker 执行；
  - `StageHandler` / `DecodeHandler` 负责后续 worker 执行；
- 这样才能保持清晰、便于扩展、避免逻辑混乱。

## 3 当前接口与可演进策略

### 3.1 现有 `LLMEngine` 接口

当前 `worker/include/api/llm_engine.h` 提供：

```cpp
bool load(const std::string& model_dir, LinearBackend backend = LinearBackend::NPU);
void destroy();
void reset();
GenerationResult generate(const std::vector<int>& input_ids,
                          const GenerationConfig& cfg,
                          TokenCallback on_token = nullptr);
```

这是当前 `FullModelWorker` 的最核心接口。

### 3.2 向后兼容的接口演进

调度器实现时，应预留以下扩展点：

- `SessionId` 与 `RequestId` 分离；
- `WorkerInterface` 包装当前 full-model 调用并支持未来 stage 调用；
- `PrefillRequest` / `StageRunRequest` / `DecodeStepRequest` 等低层消息；
- `create_session()` / `generate_session()` 等可选后续接口；
- `RPCWorker` 与 `LocalWorker` 两种实现。

这样当第二个人后续在 worker 侧提供多会话接口时，调度器只需替换 `WorkerInterface` 实现，而不会重构会话管理与调度核心。

## 4 概念定义：Session 与 Request

### 4.1 Session

Session 代表“一个逻辑对话”“一个用户会话”“一个长会话上下文”。

- 由调度器 / SessionManager 创建；
- 保存聊天历史文本、系统指令、用户配置、request 元数据；
- 可以持续多轮请求；
- 例如：`session_id = "user-1234-2026-05-19"`。

### 4.2 Request

Request 代表“单次生成任务“或“单次 RPC 调度动作”。

- 每次用户发起一次文本输入、一次生成调用，都包含一个新的 `request_id`；
- `request_id` 是唯一的、单次的；
- 同一个 session 可包含多个 request；
- 例如：`request_id = 1001`。

### 4.3 为什么要分开

- `session_id` 用于会话状态与历史回溯；
- `request_id` 用于调度粒度、错误重试、日志追踪；
- 短期阶段内部数据传输也可复用同一 `request_id`，但仍保持 `session_id` 与 `request_id` 分离。

## 5 Coordinator 中心化调度方案

### 5.1 业务流程

1. 用户发送文本
2. 调度器使用 `SessionManager` 生成 / 查找 session
3. 调度器将用户文本加入 session 历史
4. 调度器将 session 历史拼成 prompt，并执行 tokenizer
5. 调度器创建 `request_id`
6. 调度器选择合适 worker 进行调度
7. worker 执行 `Prefill` / `Generate`
8. worker 结果返回给调度器
9. 调度器返回结果给客户端，并更新 session 历史

### 5.2 数据流方式

当前基础实现架构推荐：

- Worker 计算完成后，将结果上报给 Coordinator；
- Coordinator 决定下一个 worker；
- Coordinator 负责路由与失败回退；
- 数据平面在当前阶段走 Coordinator relay。

后续优化可演进为：

- Coordinator 下发 pipeline 拓扑；
- worker 之间直接转发隐藏态；
- Coordinator 保持控制平面、监控、调度决策。

### 5.3 为什么这样更稳妥

- 维护简单；
- 容错更容易；
- 便于实现负载均衡与统一监控；
- 适合先做“中心调度、后续优化”的路线。

## 6 调度器与 Worker 之间的数据结构

### 6.1 通用元数据

所有调度消息应至少包含：

- `session_id`：string
- `request_id`：uint64
- `stage_id`：int（可选，用于 shard / pipeline）
- `pos_base`：int（当前 token 位置 / hidden state 位置）
- `dtype`：enum（FP16/FP32）
- `shape`：list<int>
- `trace_id` / `priority` / `timeout`（可选）

### 6.2 Head Worker 输入

Head worker 的输入一般为：

- `input_ids`（用户 prompt token ids）
- `generation_cfg`
- `session_id` / `request_id`
- `inject_plan`（如使用 `generate_session` 或 full-model）

### 6.3 后续 Worker 输入

后续 worker 的输入应为：

- `hidden_state` 或 `stage_hidden` buffer
- `shape`（例如 `[batch, seq, hidden]`）
- `pos_base` / `sequence_length`
- `request_id` / `session_id`
- `stage_id`

Head/后续输入差异必须在 worker 代码中分离，不可混在一起。

### 6.4 示例消息

#### PrefillRequest

- `session_id`
- `request_id`
- `input_ids`
- `max_new_tokens`
- `generation_cfg`
- `timestamp`

#### PrefillResponse

- `request_id`
- `status`
- `next_token` / `logits`（可选）
- `hidden_state_bytes`
- `shape`
- `kv_meta`（若需要）

#### StageRunRequest

- `session_id`
- `request_id`
- `stage_id`
- `input_hidden_bytes`
- `shape`
- `pos_base`

#### StageRunResponse

- `request_id`
- `output_hidden_bytes`
- `shape`
- `logits` / `next_token`（如 decode step）
- `status`

## 7 网络通信实现计划

### 7.1 目录结构

```
worker/
  include/
    network/
      rpc_server.h
      rpc_client.h
      worker_service.h
  src/
    network/
      rpc_server.cpp
      rpc_client.cpp
      worker_service.cpp
  ...

scheduler/
  include/
    scheduler/
      session_manager.h
      worker_interface.h
      coordinator.h
      request_router.h
  src/
    scheduler/
      session_manager.cpp
      worker_interface.cpp
      coordinator.cpp
      request_router.cpp
  tools/
    scheduler_cli.cpp
    scheduler_http.cpp
```

### 7.2 设计要求

- `worker/` 内网络层独立，不能和算法/模型代码混杂；
- `scheduler/` 全部走根目录单独目录，负责高层业务逻辑；
- `scheduler/` 内部也要明确分块：
  - `network/` 负责 RPC 客户端/服务器、通信编码与传输；
  - `session/` 负责会话创建、历史管理、prompt 拼接；
  - `registry/` 负责 worker 节点注册、心跳与元信息；
  - `strategy/` 负责负载均衡、路由策略、调度决策；
- 每个目录只负责自己的职责：网络、会话、注册、策略、RPC；
- 用抽象接口避免调度器和底层 worker 直接耦合。

### 7.3 网络层建议技术栈

- 初期：gRPC + Protobuf；
- 后续：可优化为 custom binary message / zero-copy；
- 先完成控制层架构，再看性能优化。

## 8 调度器内部设计

### 8.1 `SessionManager`

职责：

- 创建 / 删除 session
- 保存聊天历史文本
- 拼接 prompt
- 计算 token limit / 截断策略
- 提供给调度器 `input_ids`

### 8.2 `WorkerInterface`

抽象：

```cpp
class WorkerInterface {
public:
  virtual ~WorkerInterface() = default;
  virtual bool register_node(const WorkerNodeInfo& info) = 0;
  virtual GenerationResult generate_local(...)=0;
  virtual PrefillResult prefill(const PrefillRequest& req)=0;
  virtual StageResult run_stage(const StageRunRequest& req)=0;
  virtual DecodeResult decode_step(const DecodeStepRequest& req)=0;
};
```

实现：

- `LocalWorker`：直接调用本地 `LLMEngine`；
- `RpcWorker`：通过网络调用远端 worker；
- `ShardWorker`：未来可基于 `run_stage` / `decode_step` 实现。

### 8.3 `Coordinator`

职责：

- 维护 `WorkerNode` 列表和状态；
- 维护请求队列；
- 选择可用 worker；
- 负载均衡策略；
- 路由 Prefill / Stage / Decode 调用。

### 8.4 负载均衡策略

初期可实现：

- 轮询（Round Robin）
- 最少活跃请求（Least Active）
- 估算延迟 / 空闲度打分（Latency-Aware）

后续可以扩展：

- 基于 stage 类型的能力分配；
- 基于请求优先级和会话类别；
- 基于 NPU/CPU 利用率动态调整。

## 9 代码实现计划

### 9.1 第一步：本地可运行调度器

- 在 `scheduler/include/` / `scheduler/src/` 下实现最小可运行模块；
- 用 `LocalWorker` 直接调用现有 `LLMEngine`；
- 实现简易 CLI/HTTP 接口，直接接收用户文本输入；
- 验证完整流程：文本->session->tokenizer->LLMEngine->生成->结果。

### 9.2 第二步：Worker RPC 桩

- 在 `worker/include/network/` / `worker/src/network/` 下实现 RPC 网络层；
- 提供 `Register` / `Heartbeat` / `Prefill` / `RunStage` / `DecodeStep` RPC；
- 让 worker 能被 Coordinator 发现并注册；
- 继续使用本地 `LLMEngine` 作为执行后端。

### 9.3 第三步：Coordinator 调度与负载均衡

- 实现 worker 注册与健康检查；
- 实现请求分配和响应收集；
- 让调度器支持多 worker 路由；
- 先做 Coordinator relay 模式。

### 9.4 第四步：Stage / Shard 接口适配

- 设计和实现 `PrefillRequest` / `StageRunRequest` / `DecodeStepRequest`；
- 让调度器能够根据 worker 类型选择 `FullModelWorker` 或 `StageWorker`；
- 让 worker 侧实现 head / stage 逻辑分离。

### 9.5 第五步：可选优化路径

- worker 之间直接转发隐藏态；
- 低延迟 binary transport；
- batch/merge decode requests；
- 复杂调度策略。

## 10 目录结构与文件清单

### 10.1 `scheduler/` 目录

```
scheduler/
  include/
    coordinator.h
    session_manager.h
    worker_interface.h
    request_router.h
    scheduler_types.h
    network/
      rpc_client.h
      rpc_server.h
    registry/
      node_registry.h
      node_info.h
    strategy/
      load_balancer.h
      schedule_policy.h
  src/
    coordinator.cpp
    session_manager.cpp
    worker_interface.cpp
    request_router.cpp
    network/
      rpc_client.cpp
      rpc_server.cpp
    registry/
      node_registry.cpp
    strategy/
      load_balancer.cpp
      schedule_policy.cpp
  tools/
    scheduler_cli.cpp
    scheduler_http.cpp
```

### 10.2 `worker/` 目录

```
worker/
  include/
    api/
    core/
    model/
    ops/
    backend/
    network/
      rpc_server.h
      rpc_client.h
      worker_service.h
  src/
    api/
    model/
    ops/
    backend/
    network/
      rpc_server.cpp
      rpc_client.cpp
      worker_service.cpp
    main.cpp
    chat_main.cpp
```

### 10.3 核心文件说明

- `scheduler/include/session_manager.h` / `scheduler/src/session_manager.cpp`
  - 会话生命周期、历史管理、prompt 拼接、tokenizer 入口。
- `scheduler/include/worker_interface.h` / `scheduler/src/worker_interface.cpp`
  - 抽象 worker 调用，支持 `LocalWorker` / `RpcWorker` / `ShardWorker`。
- `scheduler/include/coordinator.h` / `scheduler/src/coordinator.cpp`
  - 维护节点、队列、路由、调度决策、负载均衡。
- `scheduler/include/request_router.h` / `scheduler/src/request_router.cpp`
  - 请求流控、优先级、调度策略插件。
- `scheduler/include/network/rpc_client.h` / `scheduler/src/network/rpc_client.cpp`
  - 和远端 worker 的 RPC 调用封装。
- `scheduler/include/network/rpc_server.h` / `scheduler/src/network/rpc_server.cpp`
  - 可选提供 HTTP/ gRPC 控制面或本地管理接口。
- `scheduler/include/registry/node_registry.h` / `scheduler/src/registry/node_registry.cpp`
  - Worker 节点注册、心跳、健康状态、能力元数据。
- `scheduler/include/strategy/load_balancer.h` / `scheduler/src/strategy/load_balancer.cpp`
  - 轮询、最少活跃、延迟感知调度策略。
- `worker/include/network/worker_service.h` / `worker/src/network/worker_service.cpp`
  - worker 侧 RPC 服务定义与 handler。
- `worker/include/network/rpc_server.h` / `worker/src/network/rpc_server.cpp`
  - worker 侧网络服务框架。
- `worker/include/network/rpc_client.h` / `worker/src/network/rpc_client.cpp`
  - 可选用于 worker 自主调用下一段 pipeline 的直连接口。

## 11 结论与优势

- 这个目录结构把 `scheduler` 的网络、会话、注册和策略明确拆开；
- `scheduler/include/` / `scheduler/src/` 不再多套 `scheduler/` 目录层；
- `worker/` 的网络部分独立于模型/算子代码；
- 未来实现 `RpcWorker` / `ShardWorker` 时，只需补 `worker/network` 和 `scheduler/network` 的具体实现。 

---

## 12 备注

- 该文档仍以设计为准，实际代码可按此结构逐步落地；
- 若你希望，我也可以继续直接生成 `scheduler` 和 `worker` 的代码骨架。 

## 13 结语

这个更清晰的结构能让实现过程更可控，也便于你在团队中与第二个人分工对接。

- 当前方案符合你希望的“Coordinator 中心化调度”架构；
- 当前可实现的版本是“Coordinator relay + LocalWorker + 会话管理 + 负载均衡”；
- 该设计不会把 head worker 和后续 worker 的输入混在一起；
- 其核心优势是：模块分离清晰，`worker/` 和 `scheduler/` 目录各司其职；
- 未来接入多会话、多 stage、RPC worker 只需补充 `WorkerInterface` 和 `worker/ network` 层，而不会改大调度逻辑。

---

## 附录：建议目录结构

```
worker/
  include/
    api/
    core/
    model/
    ops/
    backend/
    network/
      rpc_server.h
      rpc_client.h
      worker_service.h
  src/
    api/
    model/
    ops/
    backend/
    network/
      rpc_server.cpp
      rpc_client.cpp
      worker_service.cpp
    main.cpp
    chat_main.cpp

scheduler/
  include/
    coordinator.h
    session_manager.h
    worker_interface.h
    request_router.h
    scheduler_types.h
    network/
      rpc_client.h
      rpc_server.h
    registry/
      node_registry.h
      node_info.h
    strategy/
      load_balancer.h
      schedule_policy.h
  src/
    coordinator.cpp
    session_manager.cpp
    worker_interface.cpp
    request_router.cpp
    network/
      rpc_client.cpp
      rpc_server.cpp
    registry/
      node_registry.cpp
    strategy/
      load_balancer.cpp
      schedule_policy.cpp
  tools/
    scheduler_cli.cpp
    scheduler_http.cpp

docs/
  scheduler_design.md
  rpc_interface.md
```

这个实现计划先以文档为准，后续即可照此拆分代码。