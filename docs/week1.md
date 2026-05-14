# Week1

|PRE

## 1 最终效果

使用模型：

```python
from edge_llm import DistributedLLM

# 初始化（指定调度器地址等）
model = DistributedLLM(coordinator="192.168.1.100:8080")
# 加载模型
model.load("qwen7b")
# 推理
result = model.generate(
    image="cat.jpg",
    prompt="这是什么动物？"
)
print(result)  # 输出结果
```

系统内部：

| 阶段           | 动作                                                                                                       |
| ------------ | -------------------------------------------------------------------------------------------------------- |
| `load()`     | 客户端通知调度器 → 调度器检查是否有该模型的记录 → 若无则触发部署（通知各 Worker 下载对应切片并加载）→ 各 Worker 回报就绪 → 调度器标记模型可用                     |
| `generate()` | 客户端发送图片/文本 → 调度器生成 embedding → 调度器依次调 Worker0 → Worker1 → Worker2 完成流水线推理 → 采样 token → 重复直到生成完整回答 → 返回结果 |

## 2 阶段划分

### 阶段1：单节点 RK3588 推理基线

**目标**

先让一个模型或模型片段在单个 RK3588 上稳定跑起来。

**主要工作**

1.模型转换

- RKLLM 路线
  
  ```
  HuggingFace 模型
    ↓
  RKLLM-Toolkit 转换
    ↓
  生成 .rkllm 模型
    ↓
  RKLLM Runtime 加载
    ↓
  RK3588 NPU 推理
  ```

- RKNN 路线
  
  ```
  PyTorch / ONNX
    ↓
  RKNN-Toolkit2 转换
    ↓
  生成 .rknn 模型
    ↓
  RKNN Runtime 加载
    ↓
  RK3588 NPU 推理
  ```

2.基础执行器

```cpp
class RKExecutor {
public:
    bool load_model(const std::string& model_path);
    bool init_runtime();
    Tensor forward(const Tensor& input);
    void release();
};
```

**阶段产出**

- 单节点推理 Demo
- RKNN/RKLLM Executor 初版
- 单节点性能基线

---

### 阶段2：单节点多 NPU Core 调度

**目标**

利用 RK3588 内部多个 NPU core，提高单节点吞吐或降低延迟。

**主要工作**

- 方式 A：单模型多核执行
  
  ```
  一个模型 context
    ↓
  绑定多个 NPU core
    ↓
  测试推理性能
  ```

- 方式 B：多 context 多核并行
  
  ```
  Context 0 → NPU Core 0
  Context 1 → NPU Core 1
  Context 2 → NPU Core 2
  ```

适合多个请求并发或多个 shard 并行。

**阶段产出**

- 单节点多核调度器
- 不同 core mask 策略性能对比
- 单节点并发推理 Demo

> 这一阶段对应课题目标中的：基于 RK3588 的 NPU 架构，实现单节点内多 NPU 核心的高性能推理。

---

### 阶段3：模型切分与 shard 执行

**目标**

把完整大模型拆成多个子模型 shard，为多节点部署做准备

**切分方式**

先做静态按层切分

```
Shard 0：Embedding + Layer 0 ~ 7
Shard 1：Layer 8 ~ 15
Shard 2：Layer 16 ~ 23
Shard 3：Layer 24 ~ 31 + LM Head
```

每个 shard 独立转换成 `.rknn` 或者 `.rkllm shard`，视工具支持情况而定。

**主要工作**

1. 分析目标模型结构
2. 标记每个 Transformer Block 输入输出
3. 将模型导出为多个 ONNX 子图
4. 使用 RKNN-Toolkit2 转换每个子图
5. 校验 shard 输入输出 shape
6. 在单机上串行执行多个 shard

先在一台机器上验证，再扩展到多机器

```
Shard 0 → Shard 1 → Shard 2 → Shard 3
```

**阶段产出**

- ModelPartitioner 初版
- 多个 RKNN/RKLLM shard
- 单机多 shard 串行推理 Demo
- shard 输出正确性验证报告

---

### 阶段 4：多节点 Worker 与通信框架

**目标**

让多个 RK3588 节点能够协同执行模型 shard

**架构**

Worker节点

```
RK3588 Worker
  ├── Worker ID
  ├── IP / Port
  ├── NPU Core 信息
  ├── 本地 shard 信息
  ├── RKNN/RKLLM Executor
  ├── Local KV Cache
  └── RPC Server
```

Coordinator

- 节点注册
- 模型部署计划
- 请求分发
- pipeline 调度
- 错误处理
- 性能统计

**通信方式**

先从 TCP/gRPC 入手

消息格式示例：

```cpp
struct TensorMessage {
    uint64_t request_id;
    uint32_t token_pos;
    uint32_t src_stage;
    uint32_t dst_stage;
    DataType dtype;
    std::vector<int64_t> shape;
    std::vector<uint8_t> payload;
};
```

节点间传输内容主要是：

- hidden states
- logits
- 控制消息

不建议频繁传输：

- 完整 KV Cache
- 模型权重

**阶段产出**

- RK3588 Worker 程序
- Coordinator 程序
- 节点注册与心跳机制
- Tensor 传输协议
- 两节点 shard 推理 Demo

---

### 阶段 5：多节点 Pipeline Parallel 推理

**目标**

实现真正的多节点协同推理。

**推理流程**

- Prefill 阶段
  
  ```
  Prompt tokens
    ↓
  Worker 0 执行 Shard 0
    ↓
  hidden states 发送给 Worker 1
    ↓
  Worker 1 执行 Shard 1
    ↓
  hidden states 发送给 Worker 2
    ↓
  Worker 2 执行 Shard 2
    ↓
  hidden states 发送给 Worker 3
    ↓
  Worker 3 执行 Shard 3
    ↓
  输出 logits
  ```

- Decode 阶段
  
  每生成一个 token：
  
  ```
  当前 token hidden state
    ↓
  Worker 0
    ↓
  Worker 1
    ↓
  Worker 2
    ↓
  Worker 3
    ↓
  logits
    ↓
  采样下一个 token
  ```

**KV Cache 管理**

每个 Worker 只维护自己的 KV Cache：

- Worker 0：Layer 0 ~ 7 KV Cache
- Worker 1：Layer 8 ~ 15 KV Cache
- Worker 2：Layer 16 ~ 23 KV Cache
- Worker 3：Layer 24 ~ 31 KV Cache

decode 阶段只需要传 hidden state。

**阶段产出**

- 2 节点 pipeline 推理
- 4 节点 pipeline 推理
- Prefill / Decode 分离调度
- Local KV Cache 管理
- 端到端生成 Demo

> 这一阶段对应课题目标中的：通过模型并行与流水线并行相结合的方式，将大模型分布到多个 RK3588 节点上协同执行。

---

### 阶段 6：vLLM-like 调度与服务接口

**目标**

在系统上层提供统一编程接口，屏蔽底层多节点和 NPU 异构细节。

**API 设计**

可以提供 OpenAI-compatible 接口：

```
POST /v1/chat/completions
POST /v1/completions
GET  /v1/models
```

也可以提供 C++ / Python SDK：

```python
engine = RKDistributedLLM(
    model_dir="./model_shards",
    cluster_config="./cluster.yaml"
)
output = engine.generate(
    prompt="介绍一下边缘侧分布式推理",
    max_new_tokens=128,
    temperature=0.7
)
```

配置文件：

```yaml
model:
  name: qwen-rknn-sharded
  num_layers: 32
  shard_policy: layerwise
cluster:
  - id: worker0
    ip: 192.168.1.10
    port: 9000
    layers: [0, 7]
    npu_cores: [0, 1, 2]
  - id: worker1
    ip: 192.168.1.11
    port: 9000
    layers: [8, 15]
    npu_cores: [0, 1, 2]
runtime:
  parallel: pipeline
  kv_cache: layer_local
  transport: tcp
  batch_size: 1
```

**调度设计**

参考 vLLM，但做轻量化实现：

```
Request Queue
  ↓
Prefill Scheduler
  ↓
Decode Scheduler
  ↓
Pipeline Runtime
  ↓
Streaming Output
```

**先支持**

- 单请求流式生成
- 多请求排队
- 简单 batch

**后续再支持**

- continuous batching
- chunked prefill
- prefix cache

**阶段产出**

- 统一 API Server
- Python/C++ 调用接口
- 流式输出 Demo
- vLLM-like 请求调度器

> 这一阶段对应课题目标中的：构建统一编程接口，屏蔽底层 NPU 异构性。

---

### 阶段 7：自动切分与性能优化

**目标**

从"手动切分"升级到"根据性能自动切分"。

**Profiling 数据**

每个 shard / layer 统计：

- 计算时间
- 输入 hidden state 大小
- 输出 hidden state 大小
- 权重大小
- KV Cache 大小
- 通信耗时
- NPU core 利用率
- 内存占用

**自动切分目标**

设每层代价为：

```
cost(layer_i) =
  α × compute_time_i
+ β × memory_usage_i
+ γ × communication_size_i
```

自动寻找切分点，使得：

- 各节点计算时间尽量均衡
- 节点内存不超过上限
- 节点间通信量尽量小
- 整体 pipeline bubble 尽量少

**优化方向**

1. hidden state 使用 FP16 传输
2. 通信 buffer 复用
3. 异步 send / recv
4. 双缓冲
5. Prefill chunking
6. Decode 队列调度
7. NPU core mask 自动选择

**阶段产出**

- Profiler
- 自动切分器
- 通信优化版本
- 多节点性能对比报告

---

## 3 分工

| No. | 分工            | 描述                         |
| --- | ------------- | -------------------------- |
| 1   | 模型切分/格式转换     | .rknn 生成、自动切分、模型正确性验证      |
| 2   | 执行器           | Executor实现，单板、多核推理...      |
| 3   | 网络通信          | TCP/gRPC 传输、Worker 通信集成、部署 |
| 4   | 调度器           | Coordinator、流水线调度          |
| 5   | api封装+kv缓存+测试 | 最上层api设计/封装、测试；kv缓存设计      |

代码结构类似于：

```
distributed-inference/
├── Makefile
├── scripts/                            # 脚本（运维/工具）
├── model/                              # 模型相关
├── common/                             # 公共模块
│   ├── message.h                       # 通信消息结构体定义
│   ├── tensor.h                        # Tensor 数据结构
│   └── rpc_interface.h                 # RPC 接口定义（proto 风格）
|
├── worker/                             # Worker 节点（RK3588）
│   ├── include/
│   │   ├── rknn_executor.h             # RKNN 执行器头文件
│   │   ├── rpc_server.h                # RPC 服务端头文件
│   │   ├── kv_cache.h                  # KV Cache 管理头文件
│   │   └── config.h                    # 配置解析头文件
│   ├── src/
│   │   ├── main.cpp                    # Worker 入口
│   │   ├── rknn_executor.cpp           # RKNN 执行器实现
│   │   ├── rpc_server.cpp              # RPC 服务端实现
│   │   └── kv_cache.cpp                # KV Cache 实现
│   └── test/
│
├── coordinator/                        # 调度器（PC）
│   ├── coordinator.py                  # 主控逻辑
│   ├── scheduler.py                    # 流水线调度逻辑
│   ├── worker_client.py                # Worker RPC 客户端
│   ├── vision_encoder.py               # 视觉编码（多模态）
│   ├── config.yaml                     # 集群配置
│   └── requirements.txt                # Python 依赖
│
├── api/                                # api封装
├── tests/
└── examples/                           # 示例程序
```
