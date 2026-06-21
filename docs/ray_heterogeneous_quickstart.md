# Ray 异构分布式推理完整手册

这份文档是当前仓库里关于 Ray 异构分布式推理的主文档，目标是把下面几件事一次讲清楚：

- 现在这版代码到底已经跑到什么程度
- 本地单机、Ray 单机、多机 Ray 分别怎么启动
- 当前实现里的系统架构是什么
- RK3588 板子现在接入到哪一步了
- 哪些能力已经验证，哪些能力是“按现实现应可跑通，但还缺实机验证”
- 如果机器数足够，不同角色应该怎么部署

如果你只想先跑起来，可以直接看“运行命令”部分。

---

## 1. 一句话结论

截至当前版本，下面这些结论和代码实现是一致的：

- `pc` 单机 Ray 分布式 pipeline 已经实际跑通
- `scheduler_cli -> Ray pipeline actor -> token 输出` 已经实际跑通
- `centralized` 是当前真正稳定、可作为正式方案使用的 Ray pipeline 模式
- `rk3588` 的分段推理接口已经补齐到和 `pc` 同一抽象层
- `serve_worker.py --target rk3588` 已经支持
- `pc + rk3588` 的异构 Ray 链路在架构和接口上已经打通
- 但 RK3588 板子的实机编译与端到端验证，当前仍受板端工具链 / 运行环境问题阻塞

更直白一点说：

- “PC 版分布式链路已经可跑、可演示、可讲清楚” 这句话可以成立
- “异构板端接口已经接上，但板子还没完成最终实机验收” 这句话也应该明确说出来

---

## 2. 当前能力边界

先把“已经验证”和“设计上已完成但未全量实测”分开。

### 2.1 已经验证通过

- 本机 `pc` target 的 Ray full-model actor
- 本机 `pc` target 的 Ray distributed pipeline
- `ray_generate.py` 对命名 actor 发起请求
- `scheduler_cli` 通过 `ray:<actor-name>` 调用 Ray 分布式链路
- `HeadWorkerActor / StageWorkerActor / TailWorkerActor / DistributedPipelineActor` 的 actor 编排
- `serve_worker.py --target pc --mode distributed`

已验证过的一组命令如下：

```bash
ray stop --force
ray start --head --port=6379 --resources='{"role_head": 1, "role_stage": 1, "role_tail": 1, "role_pipeline": 1}'
```

```bash
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
  --ray-address 127.0.0.1:6379 \
  --head-resource role_head \
  --stage-resource role_stage \
  --tail-resource role_tail \
  --pipeline-resource role_pipeline
```

```bash
python3 scheduler/tools/ray_generate.py \
  --ray-address 127.0.0.1:6379 \
  --actor-name pc-distributed \
  151644 8948 198
```

成功结果示例：

```text
STATUS OK
OUTPUT_IDS 40 2776 264 2409 8405 315 279 1473 330 785
```

### 2.2 已经补代码，但还缺 RK3588 板端实机验收

- `rk3588_engine` 分段加载接口
- `rk3588_engine.tokens_to_hidden`
- `rk3588_engine.hidden_forward`
- `rk3588_engine.hidden_to_token`
- `serve_worker.py --target rk3588`
- Ray actor 层对 `rk3588` target 的统一调度入口

也就是说，从代码结构上看，RK3588 已经不再只是“整模型单点推理”，而是已经进入了和 PC 一样的：

- `head`
- `stage`
- `tail`

分段执行模型。

### 2.3 当前仍然不能说“完全做完”的部分

- 板端环境还没有完成稳定构建
- 多台机器跨网络的 Ray 路径没有完成完整实机压测
- `p2p` 还不是严格意义上的 worker-to-worker 物理直连链路
- 分布式容错还没有做成完整故障恢复系统
- “每个角色使用不同 target，例如 head=pc、stage=rk3588、tail=pc” 的按角色混合 target 启动逻辑，目前还没有在 `serve_worker.py` 里做成一条命令直接配置

---

## 3. 系统架构总览

当前这套 Ray 分布式推理链路可以分成 4 层。

### 3.1 入口层

- `scheduler_cli`
- `scheduler/tools/ray_generate.py`

职责：

- 文本输入
- tokenizer / detokenizer
- 会话拼接
- 请求发起

### 3.2 调度控制层

- `DistributedPipelineActor`

职责：

- 持有整条 pipeline 的 actor handle
- 统一编排 `head -> stage -> tail`
- 对外暴露一个逻辑入口 actor

### 3.3 执行角色层

- `HeadWorkerActor`
- `StageWorkerActor`
- `TailWorkerActor`
- 以及非分段模式下的 `FullModelWorkerActor`

职责：

- `HeadWorkerActor`
  输入 token ids，输出第一段 hidden state
- `StageWorkerActor`
  输入 hidden state，输出下一段 hidden state
- `TailWorkerActor`
  输入 hidden state，输出最终 token id

### 3.4 底层推理引擎层

- `runtime.pc_engine`
- `runtime.rk3588_engine`

它们都通过：

- `bindings/python/runtime/engine.py`
- `runtime.create_engine(target)`

暴露统一接口。

---

## 4. 当前分布式数据流

### 4.1 centralized 模式

这是当前真正稳定可用的模式。

数据流可以理解成：

```text
Scheduler / ray_generate.py
  -> DistributedPipelineActor
  -> HeadWorkerActor
  -> StageWorkerActor(们)
  -> TailWorkerActor
  -> DistributedPipelineActor
  -> Scheduler / ray_generate.py
```

更细一点：

```text
输入 token ids
  -> head.tokens_to_hidden(...)
  -> stage.hidden_forward(...)
  -> stage.hidden_forward(...)
  -> tail.hidden_to_token(...)
  -> 生成一个 token
  -> 如果还要继续 decode，则重复上述过程
```

这里真正跨 actor 传的“数据面”是：

- token ids
- hidden state
- token id

而“控制面”由 `DistributedPipelineActor` 统一管理。

### 4.2 p2p 模式

当前仓库里已经有：

- `--pipeline-mode p2p`
- `DistributedPipelineActor._generate_p2p(...)`

但要明确：

- 现在的 `p2p` 仍然是 Ray actor 编排下的模式切换
- 不是裸 TCP 的 worker 直连数据平面
- 不是每个 stage 自己维护下游 socket 然后完全绕开中心 actor

所以当前准确说法是：

- `centralized`：正式可用
- `p2p`：接口和控制抽象已预留，但不是完整物理直连版

---

## 5. 角色划分与层切分

当前 `serve_worker.py` 对 distributed 模式支持：

- `--num-stages 1`
- `--num-stages 2`

对应切分如下。

### 5.1 `--num-stages 1`

```text
head    : layer 0  - 4
stage-0 : layer 4  - 12
tail    : layer 12 - 28
```

### 5.2 `--num-stages 2`

```text
head    : layer 0  - 4
stage-0 : layer 4  - 8
stage-1 : layer 8  - 12
tail    : layer 12 - 28
```

这套切分来自当前 `ray_runtime/serve_worker.py` 里的固定实现，因此文档和代码是一致的。

---

## 6. 本机非 Ray 分布式 RPC 路径

如果你想先验证“模型被切成多段、每一段走网络式 RPC”这件事，但又不想先碰 Ray，可以先跑这条路径。

### 6.1 构建

```bash
cd ~/rk3588-npu

cmake -S worker-pc -B worker-pc/build
cmake --build worker-pc/build -j4

cmake -S scheduler -B scheduler/build
cmake --build scheduler/build -j4
```

### 6.2 启动 4 个 worker 进程

终端 1：

```bash
./worker-pc/build/worker_pc_rpc_server \
  --device cpu \
  --mode head \
  --layer-begin 0 \
  --layer-end 4 \
  models/qwen1.5b-instruct/Qwen2-1.5B-Instruct \
  127.0.0.1:5001
```

终端 2：

```bash
./worker-pc/build/worker_pc_rpc_server \
  --device cpu \
  --mode stage \
  --layer-begin 4 \
  --layer-end 8 \
  models/qwen1.5b-instruct/Qwen2-1.5B-Instruct \
  127.0.0.1:5002
```

终端 3：

```bash
./worker-pc/build/worker_pc_rpc_server \
  --device cpu \
  --mode stage \
  --layer-begin 8 \
  --layer-end 12 \
  models/qwen1.5b-instruct/Qwen2-1.5B-Instruct \
  127.0.0.1:5003
```

终端 4：

```bash
./worker-pc/build/worker_pc_rpc_server \
  --device cpu \
  --mode tail \
  --layer-begin 12 \
  --layer-end 28 \
  models/qwen1.5b-instruct/Qwen2-1.5B-Instruct \
  127.0.0.1:5004
```

### 6.3 启动调度器

```bash
./scheduler/build/scheduler_cli \
  models/qwen1.5b-instruct/Qwen2-1.5B-Instruct \
  head:127.0.0.1:5001 \
  tail:127.0.0.1:5004 \
  stage:127.0.0.1:5002 \
  stage:127.0.0.1:5003
```

这个路径的意义是：

- 不依赖 Ray
- 先验证“分段执行 + hidden state 传递”的基本设计
- 是理解底层 worker 通信协议最直接的一条链路

---

## 7. 本机单机 Ray 路径

这是当前最推荐、也最完整的演示路径。

### 7.1 构建

```bash
cd ~/rk3588-npu
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate rk3588

cmake -S bindings -B bindings/build
cmake --build bindings/build -j4

cmake -S scheduler -B scheduler/build
cmake --build scheduler/build -j4
```

### 7.2 配置环境变量

```bash
export PYTHONPATH=bindings/python
export SCHEDULER_RAY_PYTHON=/home/deep/miniforge3/envs/rk3588/bin/python3.10
export RAY_memory_monitor_refresh_ms=0
```

说明：

- `PYTHONPATH`
  让 Ray worker 可以导入 `runtime`
- `SCHEDULER_RAY_PYTHON`
  让 `scheduler_cli` 调内部 Python 脚本时和当前 Ray 环境一致
- `RAY_memory_monitor_refresh_ms=0`
  防止单机小内存环境下 Ray 过早杀 actor

### 7.3 启动本机 Ray head

```bash
ray stop --force
ray start --head --port=6379 --resources='{"role_head": 1, "role_stage": 1, "role_tail": 1, "role_pipeline": 1}'
```

### 7.4 启动 distributed actor

单 stage 版本：

```bash
python3 ray_runtime/serve_worker.py \
  models/qwen1.5b-instruct/Qwen2-1.5B-Instruct \
  --target pc \
  --device cpu \
  --mode distributed \
  --pipeline-mode centralized \
  --num-stages 1 \
  --actor-name pc-distributed \
  --ray-address 127.0.0.1:6379 \
  --head-resource role_head \
  --stage-resource role_stage \
  --tail-resource role_tail \
  --pipeline-resource role_pipeline
```

双 stage 版本：

```bash
python3 ray_runtime/serve_worker.py \
  models/qwen1.5b-instruct/Qwen2-1.5B-Instruct \
  --target pc \
  --device cpu \
  --mode distributed \
  --pipeline-mode centralized \
  --num-stages 2 \
  --actor-name pc-distributed \
  --ray-address 127.0.0.1:6379 \
  --head-resource role_head \
  --stage-resource role_stage \
  --tail-resource role_tail \
  --pipeline-resource role_pipeline
```

### 7.5 最小 token 级验证

```bash
python3 scheduler/tools/ray_generate.py \
  --ray-address 127.0.0.1:6379 \
  --actor-name pc-distributed \
  151644 8948 198
```

### 7.6 文本级验证

```bash
./scheduler/build/scheduler_cli \
  models/qwen1.5b-instruct/Qwen2-1.5B-Instruct \
  ray:pc-distributed
```

启动后可直接输入：

```text
你好
```

---

## 8. 多台机器 Ray centralized 部署

这部分是当前实现下最符合代码实际、也最值得对外讲的“真实多机设计”。

核心思想是：

- 所有机器先加入同一个 Ray 集群
- `serve_worker.py` 在逻辑上创建整条 pipeline 的 actor
- 每个 actor 最终落到哪台机器，由 Ray resource 标签决定

也就是说：

- 不是每台机器手动启动一个 `head/stage/tail actor`
- 而是每台机器先作为 Ray 节点加入集群
- 然后由一个 launcher 在集群里创建 actor

### 8.1 三机最小版

推荐角色分配：

- 机器 A：Ray head + pipeline actor + head actor
- 机器 B：stage actor
- 机器 C：tail actor

这是最适合展示“角色确实分散到不同机器”的最小方案。

#### 机器 A

启动 Ray head：

```bash
ray stop --force
ray start --head --port=6379 --resources='{"role_head": 1, "role_pipeline": 1}'
```

启动 pipeline：

```bash
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

加入 Ray 集群：

```bash
ray stop --force
ray start --address='A机器IP:6379' --resources='{"role_stage": 1}'
```

#### 机器 C

加入 Ray 集群：

```bash
ray stop --force
ray start --address='A机器IP:6379' --resources='{"role_tail": 1}'
```

#### 发请求

在任何能连上 Ray head 的机器上：

```bash
export PYTHONPATH=bindings/python

python3 scheduler/tools/ray_generate.py \
  --ray-address A机器IP:6379 \
  --actor-name pc-distributed \
  151644 8948 198
```

### 8.2 四机版

如果机器数足够，并且想让每台机器只承载一个执行角色，可以用 `--num-stages 2`：

- 机器 A：Ray head + pipeline actor + head actor
- 机器 B：stage-0 actor
- 机器 C：stage-1 actor
- 机器 D：tail actor

#### 机器 A

```bash
ray stop --force
ray start --head --port=6379 --resources='{"role_head": 1, "role_pipeline": 1}'
```

```bash
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

### 8.3 为什么是 A 创建所有 actor

这点很容易误解，但一定要讲准确。

`serve_worker.py` 所在的那台机器，只是“发起 actor 创建请求”的 launcher。

它做的事情是：

1. 连接到 Ray 集群
2. 调用 `HeadWorkerActor.options(...).remote(...)`
3. 调用 `StageWorkerActor.options(...).remote(...)`
4. 调用 `TailWorkerActor.options(...).remote(...)`
5. 调用 `DistributedPipelineActor.options(...).remote(...)`

真正的 actor 运行在哪台机器，不是由 launcher 所在机器决定，而是由：

- Ray 集群里哪些节点在线
- 这些节点暴露了哪些 resource 标签

共同决定。

所以更准确的说法是：

- A 创建的是“actor 逻辑实例”
- Ray 决定它们落在哪些物理节点

---

## 9. Tailscale 双机版本

如果机器之间没有稳定公网互通，当前建议用 Tailscale。

### 9.1 已知地址示例

当前你们实际用过的地址是：

```text
PC_TS_IP=100.124.132.113
SERVER_TS_IP=100.66.163.79
```

### 9.2 推荐拓扑

- 服务器作为 Ray head
- 本地 PC 作为 Ray worker node
- 本地 PC 上启动 `serve_worker.py`

### 9.3 服务器

```bash
ray stop --force
ray start --head \
  --node-ip-address=100.66.163.79 \
  --port=6379 \
  --resources='{"role_stage": 1}'
```

### 9.4 本地 PC

```bash
ray stop --force
ray start \
  --address='100.66.163.79:6379' \
  --node-ip-address=100.124.132.113 \
  --resources='{"role_head": 1, "role_tail": 1, "role_pipeline": 1}'
```

```bash
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
  --ray-address 100.66.163.79:6379 \
  --head-resource role_head \
  --stage-resource role_stage \
  --tail-resource role_tail \
  --pipeline-resource role_pipeline
```

### 9.5 当前网络实测结论

你们之前的跨机尝试里出现过：

- `tailscale ping` 可以通
- 但 `nc -vz 100.66.163.79 6379` 报 `No route to host`
- 服务器 `tcpdump` 能看到 SYN 进来，但连接建不起来

这说明：

- 不是 Ray 逻辑本身有问题
- 而是当前这次跨机实验受网络路径 / 防火墙 / Tailscale TCP 可达性影响

因此当前最稳妥的对外表述是：

- 多机 Ray 方案已经完成架构设计和命令级部署方案
- 本地单机 Ray 已实证跑通
- 双机跨公网场景在这次环境里卡在网络连通性，不属于推理代码逻辑错误

---

## 10. RK3588 板子接入现状

这是当前异构链路最关键的一块。

### 10.1 当前已经做完的代码侧能力

RK3588 底层现在已经补齐：

- 分段 `load(...)`
- `tokens_to_hidden(...)`
- `hidden_forward(...)`
- `hidden_to_token(...)`

因此从抽象上看，RK3588 现在已经能扮演：

- `head`
- `stage`
- `tail`

任意一种角色。

### 10.2 这些能力复用了什么底层优化

这次不是重新写了一套简化版模型，而是复用了原本 worker 里的底层优化路径，包括：

- `Qwen2Model` 原有 forward 结构
- `qkv_proj` 融合投影
- `gate_up_proj` 融合投影
- `LinearBackend` 后端选择
- CPU / NPU / 单卡 / 分片等设备路径
- KV cache 逻辑
- `lm_head` 对应后端

也就是说，新加的分段接口是建立在已有优化实现之上的，不是绕开队友优化重新拼了个“能跑但不代表真实后端”的假接口。

### 10.3 板子当前为什么还没完全跑通

当前真实阻塞点不在推理接口设计，而在板端环境：

- `python3-config` 指到了 3.6，而系统 `python3` 实际是 3.8
- `cmake` / `apt` 因 `glibc` 与 `libstdc++` 版本不一致而异常
- 板子上实际可用的 Python include / lib 需要手工指定
- 之前上传到板子的代码子集不完整，导致无法生成 `runtime.rk3588_engine`

所以目前状态应该表述为：

- “RK3588 代码抽象已经接好”
- “板端工具链环境还没整理完”

### 10.4 板端最小 smoke test 目标

一旦板端 `runtime.rk3588_engine` 编译出来，可以先做这类 smoke test：

```python
from runtime import create_engine

engine = create_engine("rk3588")
engine.load(model_dir, "npu", 0, 4, True, False)
hidden = engine.tokens_to_hidden([151644, 8948, 198])
engine.destroy()
```

中间 stage：

```python
from runtime import create_engine

engine = create_engine("rk3588")
engine.load(model_dir, "npu", 4, 12, False, False)
hidden2 = engine.hidden_forward(hidden, 3, 0)
engine.destroy()
```

tail：

```python
from runtime import create_engine

engine = create_engine("rk3588")
engine.load(model_dir, "npu", 12, 28, False, True)
token = engine.hidden_to_token(hidden2, 3, 0)
engine.destroy()
```

只要这三步通过，就意味着 RK3588 已经具备被 Ray actor 封装为分布式角色的基础能力。

---

## 11. 当前实现下的“异构 Ray”到底到哪一步

这个问题最好讲得非常具体。

### 11.1 已经做到的

- `pc` 和 `rk3588` 都走统一的 `create_engine(target)` 抽象
- Ray actor 层不再只绑定 `pc`
- `serve_worker.py --target rk3588` 可以创建 RK3588 版 head/stage/tail actor
- RK3588 分段接口与 PC 的 actor 方法名一致

### 11.2 还差的最后一公里

当前 `serve_worker.py` 的一个 distributed pipeline 中，所有角色会共用同一个：

- `--target`
- `--device`

这意味着当前还不能通过一条命令直接表达：

```text
head = pc
stage = rk3588
tail = pc
```

所以“多 target 混合角色链路”从代码设计上已经具备基础，但还需要继续把 role-level target config 补到 launcher 层。

### 11.3 因此现在最准确的结论

当前可以说：

- Ray 分布式架构已经支持把 RK3588 纳入同一抽象体系
- RK3588 作为分段执行节点的底层代码已经补齐
- 真实的多机异构混合部署还差板端编译验证和 launcher 粒度继续细化

---

## 12. 各种配置下的运行命令汇总

这一节专门做成“查命令”用。

### 12.1 本机 full-model Ray actor

```bash
export PYTHONPATH=bindings/python

python3 ray_runtime/single_worker_demo.py \
  models/qwen1.5b-instruct/Qwen2-1.5B-Instruct \
  --target pc \
  --device cpu \
  --ray-address 127.0.0.1:6379 \
  --actor-name pc-full \
  151644 8948 198
```

### 12.2 本机 distributed Ray actor，1 stage

```bash
python3 ray_runtime/serve_worker.py \
  models/qwen1.5b-instruct/Qwen2-1.5B-Instruct \
  --target pc \
  --device cpu \
  --mode distributed \
  --pipeline-mode centralized \
  --num-stages 1 \
  --actor-name pc-distributed \
  --ray-address 127.0.0.1:6379 \
  --head-resource role_head \
  --stage-resource role_stage \
  --tail-resource role_tail \
  --pipeline-resource role_pipeline
```

### 12.3 本机 distributed Ray actor，2 stages

```bash
python3 ray_runtime/serve_worker.py \
  models/qwen1.5b-instruct/Qwen2-1.5B-Instruct \
  --target pc \
  --device cpu \
  --mode distributed \
  --pipeline-mode centralized \
  --num-stages 2 \
  --actor-name pc-distributed \
  --ray-address 127.0.0.1:6379 \
  --head-resource role_head \
  --stage-resource role_stage \
  --tail-resource role_tail \
  --pipeline-resource role_pipeline
```

### 12.4 切到 `p2p` 模式入口

```bash
python3 ray_runtime/serve_worker.py \
  models/qwen1.5b-instruct/Qwen2-1.5B-Instruct \
  --target pc \
  --device cpu \
  --mode distributed \
  --pipeline-mode p2p \
  --num-stages 1 \
  --actor-name pc-distributed \
  --ray-address 127.0.0.1:6379 \
  --head-resource role_head \
  --stage-resource role_stage \
  --tail-resource role_tail \
  --pipeline-resource role_pipeline
```

注意：

- 这个命令可以运行
- 但当前 `p2p` 不是完整物理直连版

### 12.5 本机文本入口

```bash
./scheduler/build/scheduler_cli \
  models/qwen1.5b-instruct/Qwen2-1.5B-Instruct \
  ray:pc-distributed
```

### 12.6 RK3588 target actor 启动模板

这是代码层已经支持的命令模板：

```bash
python3 ray_runtime/serve_worker.py \
  /root/22_04_rootfs/root/matmul/worker_test/Qwen1.5B \
  --target rk3588 \
  --device npu \
  --mode distributed \
  --pipeline-mode centralized \
  --num-stages 1 \
  --actor-name rk3588-distributed \
  --ray-address 127.0.0.1:6379 \
  --head-resource role_head \
  --stage-resource role_stage \
  --tail-resource role_tail \
  --pipeline-resource role_pipeline
```

但前提是板端已经具备：

- 可导入的 `runtime.rk3588_engine`
- 正常工作的 RK3588 worker 依赖库

---

## 13. 哪些组件必须同机，哪些可以拆机

如果机器数量不限，并且想让每台机器尽量只承担最少角色，可以这样理解。

### 13.1 必须同机的

从当前实现看，没有强制要求以下角色必须物理同机：

- `head`
- `stage`
- `tail`

它们可以分别落在不同 Ray 节点。

### 13.2 最好同机的

以下两个通常放在同一台机器最合理：

- Ray head
- `DistributedPipelineActor`

原因不是“代码必须”，而是：

- 管理方便
- 故障定位方便
- 命名 actor 和控制入口集中

### 13.3 最省机器的最小部署

3 台机器就能覆盖“真正多机角色分离”的最小演示：

- A：Ray head + pipeline + head
- B：stage
- C：tail

### 13.4 最清晰的完全分角色部署

4 台机器：

- A：Ray head + pipeline
- B：head
- C：stage
- D：tail

但当前 `serve_worker.py` 里 `head` 和 `pipeline` 常常都由 launcher 所在节点发起并比较自然地放在 A，这也是为什么很多示例里写成：

- A：Ray head + pipeline + head

---

## 14. 故障处理与容错现状

这部分一定不要说过头。

### 14.1 现在已经有的

- actor `metadata()` 可查询
- launcher 会周期性检查 actor 是否还活着
- `reset()` 链路已经有
- actor `shutdown()` 已有
- Ray 本身提供基础 actor 生命周期管理

### 14.2 现在还没有完整做完的

- stage actor 崩溃后的自动重建与重新挂接
- 请求级别的透明重试
- hidden state 级别的断点续传
- 某个 stage 失效后的自动切流
- 多副本容灾
- 统一 heartbeat / health-check / failover 管理器

### 14.3 所以当前最准确的说法

当前分布式链路已经有：

- 基础运行能力
- 基础状态可观测性
- 基础生命周期控制

但还不能称为“完整容错分布式系统”。

如果面试里被问到“容错做了吗”，比较稳的回答是：

- 目前主线先把分段执行、Ray actor 编排、异构 target 抽象统一起来
- 容错目前只做到基础存活检查和 actor 生命周期控制
- 真正的 failover / retry / replica 还属于下一阶段工作

---

## 15. 你可以怎么描述当前项目完成度

如果你想用一句比较稳、又不虚的说法，可以这样表述：

> 当前这套系统在 PC 侧已经把分段推理、Ray 分布式编排、统一 actor 抽象和文本入口打通了；RK3588 侧也已经补齐到同样的分段接口抽象，异构链路在架构上是闭合的。剩下主要卡在板端工具链和多机场景的最终实机验收，而不是核心执行链路设计本身。

如果对方继续追问“那你们算没算跑通”，更细一点可以这样说：

- 单机 PC Ray distributed pipeline：已跑通
- 文本到文本 Ray 推理链路：已跑通
- RK3588 segmented engine 接口：已实现
- RK3588 板端端到端：待板端环境恢复后验收
- 多机 Ray：部署方案和 actor 资源约束已完成，当前这次跨网测试受网络环境影响

---

## 16. 相关代码位置

如果你要对着代码讲，最关键的是这些文件：

- [ray_runtime/serve_worker.py](/home/deep/rk3588-npu/ray_runtime/serve_worker.py)
  Ray launcher，负责创建 full/distributed actor

- [ray_runtime/actors.py](/home/deep/rk3588-npu/ray_runtime/actors.py)
  `FullModelWorkerActor`、`HeadWorkerActor`、`StageWorkerActor`、`TailWorkerActor`、`DistributedPipelineActor`

- [ray_runtime/ray_common.py](/home/deep/rk3588-npu/ray_runtime/ray_common.py)
  Ray 初始化、runtime env、resource 选项

- [bindings/python/runtime/engine.py](/home/deep/rk3588-npu/bindings/python/runtime/engine.py)
  target 到具体 engine 的统一入口

- [bindings/src/rk3588_engine_module.cpp](/home/deep/rk3588-npu/bindings/src/rk3588_engine_module.cpp)
  RK3588 Python binding

- [worker/include/api/llm_engine.h](/home/deep/rk3588-npu/worker/include/api/llm_engine.h)
- [worker/src/api/llm_engine.cpp](/home/deep/rk3588-npu/worker/src/api/llm_engine.cpp)
  统一推理接口层

- [worker/include/model/qwen2_model.h](/home/deep/rk3588-npu/worker/include/model/qwen2_model.h)
- [worker/src/model/qwen2_model.cpp](/home/deep/rk3588-npu/worker/src/model/qwen2_model.cpp)
  RK3588 底层模型执行和分段 forward 扩展

---

## 17. 最后建议

如果你的目标是“对外展示这套项目已经基本打通”，当前最稳妥的演示顺序是：

1. 先展示本机 Ray distributed pipeline 真正返回 token
2. 再解释 actor 拆分与 resource 调度机制
3. 再讲 RK3588 分段接口已经补齐
4. 最后把板端受阻点明确归因到工具链环境，而不是系统设计缺陷

这会比一上来硬讲“多机异构已经完全实机跑满”更稳，也更经得住细问。
