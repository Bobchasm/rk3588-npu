# 第一个人需要用到的接口清单

## 1. 目标

第一个人负责高层对话实现，主要关注“对话状态管理 + 调度 + 生成策略 + 上层接口封装”。

第一个人不需要直接操作底层模型算子和NPU细节，所有底层功能都应通过API接口调用和配置完成。

---

## 2. 当前已有接口

### 2.1 `LLMEngine` 公有方法

```cpp
class LLMEngine {
public:
    LLMEngine();
    ~LLMEngine();

    bool load(const std::string& model_dir,
              LinearBackend backend = LinearBackend::NPU);

    void destroy();

    void reset();

    GenerationResult generate(
        const std::vector<int>& input_ids,
        const GenerationConfig& cfg,
        TokenCallback on_token = nullptr);
};
```

### 2.2 `GenerationConfig`

```cpp
struct GenerationConfig {
    int  max_new_tokens = 10;
    bool greedy         = true;
    std::vector<int> stop_tokens = {151645, 151643};
    int repetition_window = 6;
    // 预留：float temperature;
    // 预留：int top_k;
    // 预留：float top_p;
};
```

### 2.3 `GenerationResult`

```cpp
struct GenerationResult {
    std::vector<int> output_ids;
    int   prefill_tokens = 0;
    int   decode_tokens  = 0;
    float prefill_ms     = 0.0f;
    float decode_ms      = 0.0f;
    bool  hit_stop       = false;
    bool  hit_repetition = false;
};
```

### 2.4 现有约定

- `LLMEngine::load()` 负责加载模型和底层资源。
- `LLMEngine::reset()` 在每一轮新会话前调用，清空KV Cache。
- `LLMEngine::generate()` 接收输入token序列，返回新生成token序列。
- `TokenCallback` 用于流式输出每个token。

---

## 3. 第一个人实际需要用到的接口

### 3.1 直接使用的已有接口

- `LLMEngine::load(model_dir, backend)`
- `LLMEngine::destroy()`
- `LLMEngine::reset()`
- `LLMEngine::generate(input_ids, cfg, on_token)`
- `GenerationConfig` 中的 `max_new_tokens`, `greedy`, `stop_tokens`, `repetition_window`
- `GenerationResult` 中的 `output_ids`, `prefill_tokens`, `decode_tokens`, `prefill_ms`, `decode_ms`, `hit_stop`, `hit_repetition`
- `TokenCallback` 用于高层流式输出。

这些接口构成了当前第一人可直接使用的最基础API。

### 3.2 首先需要新增的接口

第一个人所需功能已经超出了当前单会话、单次生成的能力，因此需要新增以下接口：

#### 3.2.1 会话管理接口

```cpp
using SessionId = std::string;

class LLMEngine {
public:
    SessionId create_session();
    void destroy_session(SessionId session_id);
    bool session_exists(SessionId session_id) const;
};
```

说明：第一个人负责多轮会话和并发会话管理，上层只需要会话ID，不直接管理KV Cache。

#### 3.2.2 会话级生成接口

```cpp
class LLMEngine {
public:
    GenerationResult generate_session(
        SessionId session_id,
        const std::vector<int>& input_ids,
        const GenerationConfig& cfg,
        TokenCallback on_token = nullptr);
};
```

说明：这个接口用于“同一会话内多轮对话”，第一个人只传会话ID和输入token即可。

#### 3.2.3 批处理接口

```cpp
struct BatchRequest {
    SessionId session_id;
    std::vector<int> input_ids;
    GenerationConfig config;
    TokenCallback callback;
};

class LLMEngine {
public:
    std::vector<GenerationResult> generate_batch(
        const std::vector<BatchRequest>& requests);
};
```

说明：用于高层调度多个请求、提升吞吐量。

#### 3.2.4 异步生成接口

```cpp
class LLMEngine {
public:
    std::future<GenerationResult> generate_async(
        SessionId session_id,
        const std::vector<int>& input_ids,
        const GenerationConfig& cfg,
        TokenCallback on_token = nullptr);

    void cancel_generation(SessionId session_id);
};
```

说明：高层需要能够提交请求后立即返回，不阻塞主线程；底层可以在CPU/NPU流水线中异步执行。

#### 3.2.5 采样策略配置接口

```cpp
struct GenerationConfig {
    int  max_new_tokens = 10;
    bool greedy         = true;
    std::vector<int> stop_tokens = {151645, 151643};
    int repetition_window = 6;

    float temperature = 1.0f;
    int   top_k       = 0;
    float top_p       = 1.0f;
    float repetition_penalty = 1.0f;
};
```

说明：第一个人负责选择策略，底层负责实现对应采样算子。

#### 3.2.6 性能监控接口

```cpp
struct PerformanceStats {
    float avg_prefill_latency_ms;
    float avg_decode_latency_ms;
    float throughput_tokens_per_sec;
    size_t peak_memory_usage_mb;
    size_t current_memory_usage_mb;
    float npu_utilization_percent;
    float cpu_utilization_percent;
};

class LLMEngine {
public:
    PerformanceStats get_performance_stats() const;
    void enable_performance_monitoring(bool enable);
};
```

说明：第一个人负责服务级指标展示和调度决策，底层负责统计采集。

---

## 4. 第一个人不需要直接调用的底层接口

以下接口不是第一个人直接使用的，而是第一个人依赖的底层实现接口：

- `KVCache` 相关优化接口（分页、量化、长序列支持）
- `op_sample_topk`, `op_sample_topp`, `op_sample_temperature` 等采样算子
- `op_fused_*` 或流水线算子
- NPU线程/异步执行内部实现

第一个人只需依赖上层API，不需要直接操作这些底层实现。

---

## 5. 推荐接口调用流程

### 单轮生成流程

1. 调用 `engine.load(model_dir)`
2. 调用 `engine.reset()` 或 `engine.create_session()` + `generate_session()`
3. 调用 `engine.generate(input_ids, cfg, callback)`
4. 处理 `GenerationResult`

### 多轮会话流程

1. 调用 `SessionId sid = engine.create_session()`
2. 每轮调用 `engine.generate_session(sid, input_ids, cfg, callback)`
3. 会话结束时调用 `engine.destroy_session(sid)`

### 并发批处理流程

1. 构造多个 `BatchRequest`
2. 调用 `engine.generate_batch(requests)`

### 异步调用流程

1. 调用 `auto future = engine.generate_async(sid, input_ids, cfg, callback)`
2. 后续使用 `future.get()` 获取结果
3. 如需取消调用，使用 `engine.cancel_generation(sid)`

---

## 6. 结论

- 当前现有接口对第一个人的基础工作够用，但不足以支持“多会话 + 批处理 + 异步 + 多种采样策略”。
- 需要新增的接口主要集中在 `LLMEngine` 上层封装和 `GenerationConfig` 扩展。
- 第一个人应当只依赖这些上层接口，底层优化由第二个人实现。
- 这样可以保持两人分工清晰：第一个人负责“对话与调度”，第二个人负责“性能与底层实现”。
