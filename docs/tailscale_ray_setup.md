# Tailscale + Ray 双机部署说明

本文档单独说明如何把本地 PC 和公网服务器通过 Tailscale 接成可互通节点，并用于当前这套 Ray 分布式推理链路。

## 1. 当前机器信息

根据当前实际环境：

- 本地 PC 主机名：`chasm`
- 本地 PC Tailscale IP：`100.124.132.113`
- 服务器主机名：`vm-24-4-ubuntu`
- 服务器 Tailscale IP：`100.66.163.79`

后续统一约定：

```text
PC_TS_IP=100.124.132.113
SERVER_TS_IP=100.66.163.79
```

## 2. ping 应该用哪个 IP

服务器上测试本地 PC：

```bash
ping 100.124.132.113
```

本地 PC 上测试服务器：

```bash
ping 100.66.163.79
```

不要使用：

- `127.0.0.1`
- 校园网内网 IP
- 公网 NAT 地址

双机部署时统一使用 Tailscale 的 `100.x.x.x` 地址。

## 3. 两台机器安装 Tailscale

本地 PC 和服务器都执行：

```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo systemctl enable --now tailscaled
sudo tailscale up
```

执行完成后检查：

```bash
tailscale status
tailscale ip -4
tailscale netcheck
```

## 4. 连通性检查

### 4.1 先看两边是否在线

本地 PC：

```bash
tailscale status
```

服务器：

```bash
tailscale status
```

理想情况是两边互相都能看到对方在线。

如果服务器看到 PC 是：

```text
offline, last seen ...
```

先不要启动 Ray，先确认：

- 对端机器没有休眠
- `tailscaled` 服务在运行
- 两边都执行过 `sudo tailscale up`
- 两边登录的是同一个 Tailscale 账号

### 4.2 再做 ping

服务器：

```bash
ping 100.124.132.113
```

本地 PC：

```bash
ping 100.66.163.79
```

如果 `status` 在线但 `ping` 表现不稳定，也可以先继续，很多情况下 Tailscale 会通过 DERP 中继维持可达性。

## 5. 当前推荐部署拓扑

当前推荐：

- 服务器做 `Ray head`
- 本地 PC 加入 Ray 集群
- 本地 PC 负责 `head / tail / pipeline coordinator`
- 服务器负责一个中间 `stage`

这样做的原因是：

- 本地 PC 在校园网里，更适合主动连出去
- 服务器有公网环境，更适合承担集群入口
- 你当前先验证一条最小双机链路，这个角色分配最稳

## 6. 服务器启动 Ray head

服务器上执行：

```bash
cd /home/ubuntu/rk3588/rk3588-npu
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rk3588

ray stop --force
ray start --head \
  --node-ip-address=100.66.163.79 \
  --port=6379 \
  --resources='{"role_stage": 1}'
```

启动后可检查：

```bash
ray status --address=100.66.163.79:6379
```

## 7. 本地 PC 加入 Ray 集群

本地 PC 上执行：

```bash
cd /home/deep/rk3588-npu
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate rk3588

ray stop --force
ray start \
  --address='100.66.163.79:6379' \
  --node-ip-address=100.124.132.113 \
  --resources='{"role_head": 1, "role_tail": 1, "role_pipeline": 1}'
```

## 8. 本地 PC 启动 distributed actor

本地 PC 上执行：

```bash
cd /home/deep/rk3588-npu
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
  --ray-address 100.66.163.79:6379 \
  --head-resource role_head \
  --stage-resource role_stage \
  --tail-resource role_tail \
  --pipeline-resource role_pipeline
```

如果日志里出现：

```text
[ray/serve_worker] distributed actor ready: ...
[ray/serve_worker] service running, press Ctrl+C to stop
```

说明这组 actor 已经创建成功。

## 9. 验证 actor

本地 PC 上执行：

```bash
cd /home/deep/rk3588-npu
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate rk3588

export PYTHONPATH=bindings/python

python3 ray_runtime/actor_status.py \
  --ray-address 100.66.163.79:6379 \
  --actor-name pc-distributed
```

## 10. 最小 token 请求验证

本地 PC 上执行：

```bash
cd /home/deep/rk3588-npu
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate rk3588

export PYTHONPATH=bindings/python
export SCHEDULER_RAY_PYTHON=/home/deep/miniforge3/envs/rk3588/bin/python3.10

python3 scheduler/tools/ray_generate.py \
  --ray-address 100.66.163.79:6379 \
  --actor-name pc-distributed \
  151644 8948 198
```

## 11. 跑完整文本链路

本地 PC 上执行：

```bash
cd /home/deep/rk3588-npu
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate rk3588

export SCHEDULER_RAY_PYTHON=/home/deep/miniforge3/envs/rk3588/bin/python3.10

./scheduler/build/scheduler_cli \
  models/qwen1.5b-instruct/Qwen2-1.5B-Instruct \
  ray:pc-distributed
```

启动后可以直接输入：

```text
你好
```

## 12. 停止 actor

本地 PC 上执行：

```bash
cd /home/deep/rk3588-npu
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate rk3588

python3 ray_runtime/stop_worker.py \
  --ray-address 100.66.163.79:6379 \
  --actor-name pc-distributed
```

## 13. 常见问题

### 13.1 服务器看到本地 PC 是 offline

先检查：

```bash
sudo systemctl status tailscaled
tailscale status
```

必要时两边重新执行：

```bash
sudo tailscale up
```

### 13.2 本地 PC 没有公网 IP，是否还能跑

可以，只要 Tailscale 两边都在线，后续就通过虚拟内网地址通信，不依赖本地 PC 具有公网 IP。

### 13.3 服务器和本地 PC 是否必须双向可达

是的，至少对 Ray 这类分布式运行时来说，不能只满足“本地 PC 能访问服务器”这一半。  
Tailscale 的意义就是把两边放进同一个可互通虚拟网络中，尽量解决校园网 NAT、无公网 IP、回连困难这些问题。
