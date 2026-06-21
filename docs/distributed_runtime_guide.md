# 分布式运行指南

## 自实现路径

当前文档只描述我们自己这套分布式运行流程：

- `scheduler_cli` 负责文本输入、session、tokenize、decode
- `worker_pc_rpc_server` 负责真正的推理执行
- 当前链路是 `head -> stage -> stage -> tail`

当前这套流程还没有接 Ray。Ray 相关内容单独留在文末占位。

### 1. 环境

```bash
conda activate rk3588
cd ~/rk3588-npu
```

### 2. 构建

先确保 `worker-pc` 和 `scheduler` 都已经编译：

```bash
cmake -S worker-pc -B worker-pc/build
cmake --build worker-pc/build -j4
```

```bash
cmake -S scheduler -B scheduler/build
cmake --build scheduler/build -j4
```

### 3. 启动 worker

建议开 4 个终端，分别启动 `head / stage / stage / tail`。

注意：

- 如果刚改过协议或 worker 代码，必须重启全部 worker
- 调度器不会替你拉起这些 worker，当前需要手动启动

```bash
./worker-pc/build/worker_pc_rpc_server \
  --device cpu \
  --mode head \
  --layer-begin 0 \
  --layer-end 4 \
  models/qwen1.5b-instruct/Qwen2-1.5B-Instruct \
  127.0.0.1:5001
```

```bash
./worker-pc/build/worker_pc_rpc_server \
  --device cpu \
  --mode stage \
  --layer-begin 4 \
  --layer-end 8 \
  models/qwen1.5b-instruct/Qwen2-1.5B-Instruct \
  127.0.0.1:5002
```

```bash
./worker-pc/build/worker_pc_rpc_server \
  --device cpu \
  --mode stage \
  --layer-begin 8 \
  --layer-end 12 \
  models/qwen1.5b-instruct/Qwen2-1.5B-Instruct \
  127.0.0.1:5003
```

```bash
./worker-pc/build/worker_pc_rpc_server \
  --device cpu \
  --mode tail \
  --layer-begin 12 \
  --layer-end 28 \
  models/qwen1.5b-instruct/Qwen2-1.5B-Instruct \
  127.0.0.1:5004
```

### 4. 启动调度器

再开一个终端，启动调度器主程序：

```bash
./scheduler/build/scheduler_cli \
  models/qwen1.5b-instruct/Qwen2-1.5B-Instruct \
  head:127.0.0.1:5001 \
  tail:127.0.0.1:5004 \
  stage:127.0.0.1:5002 \
  stage:127.0.0.1:5003
```

### 5. 输入文本

启动后直接在命令行输入文本，例如：

```text
你好
```

调度器内部会做这些事：

- 维护 session history
- 调用 tokenizer 生成 chat template 对应的 token ids
- 每轮请求前先 reset 整条 worker 链的 KV cache
- 把 token ids 送入 `head`
- 把中间 hidden state 依次送入各个 `stage`
- 把最后 hidden state 送入 `tail`
- 把输出 token ids decode 回文本

### 6. 参数说明

`scheduler_cli` 这一行里各参数含义如下：

- `models/qwen1.5b-instruct/Qwen2-1.5B-Instruct`
  模型目录，给 tokenizer 和调度器用
- `head:127.0.0.1:5001`
  `head worker` 地址
- `tail:127.0.0.1:5004`
  `tail worker` 地址
- `stage:127.0.0.1:5002`
  第一个中间 stage
- `stage:127.0.0.1:5003`
  第二个中间 stage

多个 `stage:` 会按命令行顺序串起来执行。

**默认行为**

- 当前默认 `max_new_tokens = 64`
- 当前生成策略是 greedy
- 当前 stop token 使用模型的 `<|im_end|>` / `<|endoftext|>` 对应 id
- 当前 worker 默认示例使用 `cpu`

如果后面你要测试 GPU，只需要把各个 worker 启动命令中的 `--device cpu` 改成 `--device gpu` 或 `--device auto`

**验证工具**

下面两个工具不是主流程，只是辅助验证：

```bash
./scheduler/build/stage_pipeline_demo 1 1536 127.0.0.1:5002 127.0.0.1:5003
```

```bash
./scheduler/build/distributed_generate_demo \
  models/qwen1.5b-instruct/Qwen2-1.5B-Instruct \
  127.0.0.1:5001 \
  127.0.0.1:5004 \
  10 \
  "你好" \
  127.0.0.1:5002 \
  127.0.0.1:5003
```

用途分别是：

- `stage_pipeline_demo`
  只验证 hidden state 在多个 stage 之间能否传通
- `distributed_generate_demo`
  验证 `head -> stage -> tail` 的最小生成链

## Ray 接入

下面是当前已经可用的 Ray 版本运行方式。

### 1. 先构建 Python binding

```bash
cmake -S bindings -B bindings/build
cmake --build bindings/build -j4
```

还需要重新编译一次 `scheduler`，因为 `ray:pc-distributed` 路径和 Ray Python 调用方式已经补充过：

```bash
cmake -S scheduler -B scheduler/build
cmake --build scheduler/build -j4
```

### 2. 环境配置

建议统一在 `rk3588` conda 环境下运行：

```bash
conda activate rk3588
cd ~/rk3588-npu
```

Ray 分布式版本当前有两个必须注意的环境变量：

```bash
export PYTHONPATH=bindings/python
export SCHEDULER_RAY_PYTHON=/home/deep/miniforge3/envs/rk3588/bin/python3.10
```

含义分别是：

- `PYTHONPATH=bindings/python`
  让 Ray actor 和 Python binding 能导入 `runtime.pc_engine`
- `SCHEDULER_RAY_PYTHON=...python3.10`
  让 `scheduler_cli` 内部调用 `scheduler/tools/ray_generate.py` 时，使用和 Ray 集群一致的 Python 3.10，而不是系统 `python3`

如果机器内存比较紧张，当前还建议额外加：

```bash
export RAY_memory_monitor_refresh_ms=0
```

这会关闭 Ray 的内存监控自动杀进程逻辑，避免 `tail actor` 在模型加载过程中因为单机内存接近阈值而被提前杀掉。

### 3. 启动 Ray distributed actor

```bash
PYTHONPATH=bindings/python python3 ray_runtime/serve_worker.py \
  models/qwen1.5b-instruct/Qwen2-1.5B-Instruct \
  --target pc \
  --device cpu \
  --mode distributed \
  --num-stages 1 \
  --actor-name pc-distributed \
  --object-store-memory-mb 80
```

如果是单卡 GPU 机器，不要直接用默认 GPU 资源声明。当前每个分段 actor 都需要共享同一张卡，建议显式设置：

```bash
PYTHONPATH=bindings/python python3 ray_runtime/serve_worker.py \
  models/qwen1.5b-instruct/Qwen2-1.5B-Instruct \
  --target pc \
  --device gpu \
  --mode distributed \
  --num-stages 1 \
  --actor-name pc-distributed \
  --object-store-memory-mb 80 \
  --gpu-fraction 0.25
```

集群连接：

```bash
PYTHONPATH=bindings/python python3 ray_runtime/serve_worker.py \
  models/qwen1.5b-instruct/Qwen2-1.5B-Instruct \
  --target pc \
  --device cpu \
  --mode distributed \
  --num-stages 1 \
  --actor-name pc-distributed \
  --ray-address 机器IP:6379
```

如果要把不同角色固定到不同机器，可以给不同节点声明自定义资源，再在创建 actor 时使用对应资源键。

示例：

- 机器 A 作为 `head` 节点，启动 Ray 时声明 `role_head`
- 机器 B 作为 `stage` 节点，启动 Ray 时声明 `role_stage`
- 机器 C 作为 `tail` 节点，启动 Ray 时声明 `role_tail`

例如节点加入集群时可以分别这样启动：

```bash
ray start --address='Head机器IP:6379' --resources='{"role_head": 1}'
```

```bash
ray start --address='Head机器IP:6379' --resources='{"role_stage": 1}'
```

```bash
ray start --address='Head机器IP:6379' --resources='{"role_tail": 1}'
```

然后创建 distributed actor 时显式指定：

```bash
PYTHONPATH=bindings/python python3 ray_runtime/serve_worker.py \
  models/qwen1.5b-instruct/Qwen2-1.5B-Instruct \
  --target pc \
  --device cpu \
  --mode distributed \
  --num-stages 1 \
  --actor-name pc-distributed \
  --ray-address Head机器IP:6379 \
  --head-resource role_head \
  --stage-resource role_stage \
  --tail-resource role_tail
```

这样当前这套 PC worker 的角色部署方式就和后续板子侧保持一致：都是“底层 engine 不关心节点位置，部署层通过角色资源决定 actor 落点”。

当前 pipeline 逻辑还支持两种模式选择：

- `--pipeline-mode centralized`
  由 `DistributedPipelineActor` 统一串联 `head -> stage -> tail`
- `--pipeline-mode p2p`
  当前先保留为可切换模式接口，底层仍复用同一套 actor 计算接口，后续会继续把 forwarding 逻辑做得更接近真正的 stage-to-stage 传递

示例：

```bash
PYTHONPATH=bindings/python python3 ray_runtime/serve_worker.py \
  models/qwen1.5b-instruct/Qwen2-1.5B-Instruct \
  --target pc \
  --device cpu \
  --mode distributed \
  --pipeline-mode centralized \
  --num-stages 1 \
  --actor-name pc-distributed \
  --ray-address Head机器IP:6379 \
  --head-resource role_head \
  --stage-resource role_stage \
  --tail-resource role_tail
```

参数含义：

- `--num-stages 1`
  当前 Ray 版本使用 `head + 1 stage + tail`
- `--object-store-memory-mb 80`
  降低本地 Ray object store 预留，适合低内存机器
- `--gpu-fraction 0.25`
  每个 GPU actor 向 Ray 申请 `0.25` 张卡，便于单卡机器同时调度多个 stage actor
- `--head-resource / --stage-resource / --tail-resource`
  为不同角色指定 Ray 自定义资源键，用于把不同角色固定到不同机器
- `--pipeline-mode centralized / p2p`
  选择当前 Ray distributed 逻辑模式。当前 `centralized` 更稳定，`p2p` 先用于统一接口和部署抽象

### 3.1 板子侧同步状态

当前这套角色部署和 pipeline mode 的抽象已经先在 `pc` 版本接好了，后续迁到 `rk3588` 时希望尽量保持同样模式：

- 上层仍然使用 `target + role + pipeline_mode + resource` 这一套部署参数
- 节点角色落点仍由部署层决定，而不是写死在 worker 内部

但需要注意：截至目前，`rk3588_engine` 还只有完整 `generate()` 能力，还没有补齐：

- `tokens_to_hidden`
- `hidden_forward`
- `hidden_to_token`

因此板子侧目前还不能直接复用 `head / stage / tail` 的分段 Ray pipeline，只能说部署抽象和入口形式已经开始和 PC 对齐，真正的分段能力还需要继续补齐。

如果日志里最终出现：

```text
[ray/serve_worker] distributed actor ready: {'actor_name': 'pc-distributed', ...}
[ray/serve_worker] service running, press Ctrl+C to stop
```

说明这条 Ray 分布式链已经真正拉起来了。

### 4. 直接发送 token 请求

如果只想先验证 Ray actor 本身能否返回 token，可以另开一个终端执行：

```bash
PYTHONPATH=bindings/python python3 ray_runtime/generate_request.py \
  --actor-name pc-distributed \
  151644 8948 198
```

### 5. 启动调度器，走文本输入到文本输出

如果想跑完整的“用户输入文本 -> 调度器 -> Ray distributed actor -> 文本输出”流程，再开一个终端：

```bash
export SCHEDULER_RAY_PYTHON=/home/deep/miniforge3/envs/rk3588/bin/python3.10
./scheduler/build/scheduler_cli \
  models/qwen1.5b-instruct/Qwen2-1.5B-Instruct \
  ray:pc-distributed
```

启动成功后会看到：

```text
Using Ray worker actor: pc-distributed
Scheduler CLI started. Enter text to generate, or type /exit.
```

然后就可以直接输入：

```text
你好
```

当前 `scheduler_cli` 也支持基础多会话管理，适合先做“多会话串行正确”验证。可用命令包括：

```text
/new
/new 你是一个简洁的中文助手。
/sessions
/switch session-2
/help
```

其中：

- `/new`
  创建一个新会话并切换过去
- `/new <system_prompt>`
  使用自定义 system prompt 创建新会话
- `/sessions`
  列出当前所有会话
- `/switch <session_id>`
  切换到指定会话

当前这层会话隔离主要在调度器上层完成，底层仍按单请求串行执行，因此适合先验证“不同会话历史不串”这一目标。

### 6. 当前注意事项

- CPU 版 Ray distributed 已经可以完成请求
- 单机低内存下，Ray distributed 仍然比较吃内存
- 单卡 GPU 下必须设置较小的 `--gpu-fraction`，否则 `head` 起完后 `stage/tail` 会因为 Ray 认为 GPU 不够而一直不调度
- 当前最外层 `DistributedPipelineActor` 不占 GPU，GPU 资源只分配给 `head/stage/tail`
- `scheduler_cli` 走 Ray 时，内部会调用 `scheduler/tools/ray_generate.py`，所以必须保证 `SCHEDULER_RAY_PYTHON` 指向和 Ray 集群一致的 Python 环境
- 如果前一次请求挂住，`DistributedPipelineActor` 由于 `max_concurrency=1` 会阻塞后续请求；这种情况下建议直接停掉当前 Ray 服务，重新启动整组 actor
