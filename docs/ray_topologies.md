# Ray 分布式拓扑与部署命令

本文档单独说明当前 Ray 版本的两种 pipeline 模式：

- `centralized`
- `p2p`

并给出在“机器资源足够，每台机器只放一个角色节点”时的部署方式。

## 1. 先说当前支持状态

### 1.1 centralized

这是当前真正可用、已经打通过的模式。

它的特点是：

- 调度器只需要知道一个逻辑入口 actor
- `DistributedPipelineActor` 统一编排 `head -> stage -> tail`
- `head/stage/tail` 自己不需要知道彼此 IP
- 角色落到哪台机器，由 Ray resource 决定

### 1.2 p2p

当前代码里已经支持：

- `--pipeline-mode p2p`
- 上层接口抽象
- 独立的模式选择入口

但需要明确：

- 当前 `p2p` 还不是“真正的 worker 直连转发”
- 物理通信和控制流程仍主要由 Ray actor 编排
- 也就是说，它现在更像“为后续真实 P2P 演进预留的模式接口”

因此截至目前：

- `centralized`：可以作为正式可运行方案
- `p2p`：可以作为接口和部署抽象验证，但不要把它当成完全实现的去中心化链路

## 2. 两种模式的核心区别

### 2.1 centralized

数据流逻辑：

```text
Scheduler
  -> DistributedPipelineActor
  -> HeadWorkerActor
  -> StageWorkerActor
  -> TailWorkerActor
  -> DistributedPipelineActor
  -> Scheduler
```

特点：

- 中心 actor 保存完整拓扑信息
- `head/stage/tail` 不需要知道彼此地址
- 更容易调试
- 更适合当前版本

### 2.2 p2p

理想目标数据流：

```text
Scheduler
  -> HeadWorkerActor
  -> StageWorkerActor
  -> TailWorkerActor
  -> Scheduler
```

理想 P2P 特点：

- 每个阶段知道下一跳是谁
- 拓扑信息下沉到各个 stage 节点
- 中心节点不再反复接收和转发 hidden state

但当前实现并没有完全做到上述物理直连，因此现在它更准确的定位是：

- 接口模式已留好
- 真正 worker-to-worker 直连还需要继续实现

## 3. 角色划分

当前 `pc` 版分层是固定的：

- `head`: `layer 0-4`
- `stage-0`: `layer 4-8`
- `stage-1`: `layer 8-12`
- `tail`: `layer 12-28`

当 `--num-stages 1` 时：

- `head`: `0-4`
- `stage-0`: `4-12`
- `tail`: `12-28`

当 `--num-stages 2` 时：

- `head`: `0-4`
- `stage-0`: `4-8`
- `stage-1`: `8-12`
- `tail`: `12-28`

## 4. centralized 模式部署

### 4.1 三机最小版

推荐三机角色：

- 机器 A：`Ray head` + `pipeline` + `head`
- 机器 B：`stage`
- 机器 C：`tail`

这种分法适合先验证“角色真的落在不同机器上”。

#### 机器 A

启动 Ray head：

```bash
ray stop --force
ray start --head --port=6379 --resources='{"role_head": 1, "role_pipeline": 1}'
```

启动 distributed actor：

```bash
cd ~/rk3588-npu
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate rk3588

export PYTHONPATH=bindings/python
export SCHEDULER_RAY_PYTHON=/home/deep/miniforge3/envs/rk3588/bin/python3.10
export RAY_memory_monitor_refresh_ms=0

python3 ray_runtime/serve_worker.py \
  models/qwen1.5b-instruct/Qwen2-1.5B-Instruct \
  --target pc \
  --device cpu \
  --mode distributed \
  --pipeline-mode centralized \
  --num-stages 1 \
  --actor-name pc-distributed \
  --ray-address A机器IP:6379 \
  --head-resource role_head \
  --stage-resource role_stage \
  --tail-resource role_tail \
  --pipeline-resource role_pipeline
```

#### 机器 B

加入 Ray 集群并声明 stage 资源：

```bash
ray stop --force
ray start --address='A机器IP:6379' --resources='{"role_stage": 1}'
```

#### 机器 C

加入 Ray 集群并声明 tail 资源：

```bash
ray stop --force
ray start --address='A机器IP:6379' --resources='{"role_tail": 1}'
```

#### 请求验证

在机器 A 或任何能连到集群的机器上：

```bash
cd ~/rk3588-npu
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate rk3588
export PYTHONPATH=bindings/python

python3 scheduler/tools/ray_generate.py \
  --ray-address A机器IP:6379 \
  --actor-name pc-distributed \
  151644 8948 198
```

#### 文本入口

```bash
cd ~/rk3588-npu
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate rk3588
export SCHEDULER_RAY_PYTHON=/home/deep/miniforge3/envs/rk3588/bin/python3.10

./scheduler/build/scheduler_cli \
  models/qwen1.5b-instruct/Qwen2-1.5B-Instruct \
  ray:pc-distributed
```

### 4.2 四机版

如果你有 4 台机器，并且想让每台机器只放一个角色，建议用 `--num-stages 2`：

- 机器 A：`Ray head` + `pipeline` + `head`
- 机器 B：`stage-0`
- 机器 C：`stage-1`
- 机器 D：`tail`

#### 机器 A

```bash
ray stop --force
ray start --head --port=6379 --resources='{"role_head": 1, "role_pipeline": 1}'
```

```bash
cd ~/rk3588-npu
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate rk3588

export PYTHONPATH=bindings/python
export SCHEDULER_RAY_PYTHON=/home/deep/miniforge3/envs/rk3588/bin/python3.10
export RAY_memory_monitor_refresh_ms=0

python3 ray_runtime/serve_worker.py \
  models/qwen1.5b-instruct/Qwen2-1.5B-Instruct \
  --target pc \
  --device cpu \
  --mode distributed \
  --pipeline-mode centralized \
  --num-stages 2 \
  --actor-name pc-distributed \
  --ray-address A机器IP:6379 \
  --head-resource role_head \
  --stage-resource role_stage \
  --tail-resource role_tail \
  --pipeline-resource role_pipeline
```

#### 机器 B

```bash
ray stop --force
ray start --address='A机器IP:6379' --resources='{"role_stage": 1}'
```

#### 机器 C

```bash
ray stop --force
ray start --address='A机器IP:6379' --resources='{"role_stage": 1}'
```

#### 机器 D

```bash
ray stop --force
ray start --address='A机器IP:6379' --resources='{"role_tail": 1}'
```

注意：

- 当前两个 stage 都共用 `role_stage`
- 只要有两台机器都声明了 `role_stage`，Ray 就能把两个 stage actor 分别调度过去
- 如果后面你希望强制指定 `stage-0` 和 `stage-1` 到不同机器，可以进一步扩展为 `role_stage0` / `role_stage1`

## 5. p2p 模式部署

### 5.1 当前能怎么用

当前可以这样启动：

```bash
python3 ray_runtime/serve_worker.py \
  models/qwen1.5b-instruct/Qwen2-1.5B-Instruct \
  --target pc \
  --device cpu \
  --mode distributed \
  --pipeline-mode p2p \
  --num-stages 1 \
  --actor-name pc-distributed \
  --ray-address A机器IP:6379 \
  --head-resource role_head \
  --stage-resource role_stage \
  --tail-resource role_tail \
  --pipeline-resource role_pipeline
```

部署机器的命令和 centralized 基本相同：

- 机器 A：`role_head + role_pipeline`
- 机器 B：`role_stage`
- 机器 C：`role_tail`

即：

#### 机器 A

```bash
ray stop --force
ray start --head --port=6379 --resources='{"role_head": 1, "role_pipeline": 1}'
```

#### 机器 B

```bash
ray stop --force
ray start --address='A机器IP:6379' --resources='{"role_stage": 1}'
```

#### 机器 C

```bash
ray stop --force
ray start --address='A机器IP:6379' --resources='{"role_tail": 1}'
```

#### 机器 A 启动 p2p 模式 actor

```bash
cd ~/rk3588-npu
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate rk3588

export PYTHONPATH=bindings/python
export SCHEDULER_RAY_PYTHON=/home/deep/miniforge3/envs/rk3588/bin/python3.10
export RAY_memory_monitor_refresh_ms=0

python3 ray_runtime/serve_worker.py \
  models/qwen1.5b-instruct/Qwen2-1.5B-Instruct \
  --target pc \
  --device cpu \
  --mode distributed \
  --pipeline-mode p2p \
  --num-stages 1 \
  --actor-name pc-distributed \
  --ray-address A机器IP:6379 \
  --head-resource role_head \
  --stage-resource role_stage \
  --tail-resource role_tail \
  --pipeline-resource role_pipeline
```

### 5.2 但要明确的限制

当前 `p2p` 模式下：

- 还不是“每个 worker 自己保存下一跳 IP 并主动直连转发”
- 还没有真正完成“worker 之间完全去中心化通信”
- 更准确地说，它现在验证的是：
  - 模式切换入口
  - 角色部署抽象
  - 后续演进为真正 P2P 的上层接口

因此如果你问“现在能不能部署”：

- 可以部署
- 可以跑命令
- 但当前不要把它理解成已经实现了完整去中心化转发

## 6. 两种模式下，各机器需要知道的地址是否不同

### 6.1 centralized

在当前实现下：

- 调度器只需要知道 `Ray head` 地址和逻辑 actor 名
- 具体 `head/stage/tail` 在哪里，由 Ray 和中心 actor 管理
- 各 worker 不需要手工配置彼此 IP

因此拓扑信息主要集中在：

- Ray 集群
- `DistributedPipelineActor`

### 6.2 p2p

在理想完全体 P2P 设计中：

- `head` 要知道下一跳是谁
- `stage` 也要知道下一跳是谁
- 拓扑信息会下沉到各阶段节点

但当前实现还没有完全走到这一步，所以目前部署命令层面看起来和 centralized 很像。

## 7. 当前建议

如果你现在要真正部署和验证，建议按这个优先级：

1. `centralized + 单机 localhost`
2. `centralized + 多机`
3. `p2p` 作为接口模式验证
4. 后续再把 `p2p` 真正做成 worker-to-worker 转发

也就是说，当前适合真正拿来跑整条链路的是：

- `centralized`

而 `p2p` 现在更适合：

- 统一接口
- 统一部署抽象
- 为后续真正 P2P 留好结构
