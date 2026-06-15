# 分布式运行指南

## 自实现

当前文档只描述我们自己这套分布式运行流程：

- `scheduler_cli` 负责文本输入、session、tokenize、decode
- `worker_pc_rpc_server` 负责真正的推理执行
- 当前链路是 `head -> stage -> stage -> tail`

当前这套流程还没有接 Ray。Ray 相关内容单独留在文末占位。

## 运行前提

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

### 1. 启动 worker

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

### 2. 启动调度器

再开一个终端，启动调度器主程序：

```bash
./scheduler/build/scheduler_cli \
  models/qwen1.5b-instruct/Qwen2-1.5B-Instruct \
  head:127.0.0.1:5001 \
  tail:127.0.0.1:5004 \
  stage:127.0.0.1:5002 \
  stage:127.0.0.1:5003
```

### 3. 输入文本

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

### 4. 参数说明

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

## 默认行为

- 当前默认 `max_new_tokens = 64`
- 当前生成策略是 greedy
- 当前 stop token 使用模型的 `<|im_end|>` / `<|endoftext|>` 对应 id
- 当前 worker 默认示例使用 `cpu`

如果后面你要测试 GPU，只需要把各个 worker 启动命令中的 `--device cpu` 改成 `--device gpu` 或 `--device auto`

## 验证工具

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

