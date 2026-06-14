# ray_runtime

`ray_runtime/` 是 Ray 接入层，负责：

- actor 生命周期管理
- 请求路由与后续 orchestrator 扩展
- Python binding 到底层 worker engine 的衔接

当前已跑通链路是：

`client -> Ray actor -> Python binding -> worker-pc LLMEngine`
这里先只保留 `pc` 路径，底层推理由这些目录负责：

- `worker-pc/`：PC 版本
- `bindings/`：Python <-> C++ 接口层

## 当前能力

- 单模型常驻 actor
- 模型只加载一次，后续请求重复复用
- `cpu` / `gpu` / `auto` 设备选择
- 为后续多 actor、多 stage、多节点拆分保留统一 actor 接口

## 运行前提

- 已构建 `bindings/`
- 已设置 `PYTHONPATH=bindings/python`

## 1. 单次验证

```bash
PYTHONPATH=bindings/python python3 ray_runtime/single_worker_demo.py \
  models/qwen1.5b-instruct/Qwen2-1.5B-Instruct \
  --target pc \
  --device auto \
  151644 8948 198
```

## 2. 启动常驻服务

这会启动一个命名 actor，并让模型只加载一次：

```bash
PYTHONPATH=bindings/python python3 ray_runtime/serve_worker.py \
  models/qwen1.5b-instruct/Qwen2-1.5B-Instruct \
  --target pc \
  --device auto \
  --actor-name pc-full-model
```

如果想让 actor 创建后直接留在 Ray 里、启动脚本自己退出，可以加：

```bash
--detach-only
```

## 3. 发送推理请求

另开一个终端：

```bash
PYTHONPATH=bindings/python python3 ray_runtime/generate_request.py \
  --actor-name pc-full-model \
  151644 8948 198
```

返回结果是 JSON，包含：

- `output_ids`
- `prefill_ms`
- `decode_ms`
- `elapsed_ms`
- `request_count`

## 4. 查看状态

```bash
PYTHONPATH=bindings/python python3 ray_runtime/actor_status.py \
  --actor-name pc-full-model
```

## 5. 停止服务

前台启动时可以直接 `Ctrl+C`。

如果 actor 是 detach 方式保留下来的，或者想从另一个终端主动停掉：

```bash
PYTHONPATH=bindings/python python3 ray_runtime/stop_worker.py \
  --actor-name pc-full-model
```

## 6. GPU 说明

如果 `--device gpu` 或 `--device auto`，Ray 会为 actor 申请 `num_gpus=1`。

若 actor 日志仍显示：

```text
requested device=gpu unavailable, fallback to cpu
```

说明问题在运行环境侧，比如：

- Ray 进程没有看到 CUDA 设备
- 当前 shell / WSL 没正确暴露 GPU
- `worker-pc` 构建时没有启用 CUDA

在 WSL 内如果 Ray 因内存阈值误杀 actor，可以临时加：

```bash
RAY_memory_monitor_refresh_ms=0
RAY_memory_usage_threshold=0.99
```

## 7. Ray 和原 RPC Server 的关系

Ray 不是简单把原来的 `worker_rpc_server` 重写一遍，它更像是更高一层的运行时：

- `worker_rpc_server`：更像单机单进程的自定义推理服务入口
- Ray actor：更像可调度、可命名、可常驻、可扩展到多节点的执行单元

当前单机阶段，两者都能承担“接请求然后跑模型”的作用；Ray 的主要优势在于后续扩展时更自然：

- 更容易把模型拆到多个 actor / 多个节点
- 更容易做常驻 worker 生命周期管理
- 更容易做资源声明，例如 GPU / CPU / 后续 NPU 节点标签
- 更容易把上层调度和底层执行分开

所以现在可以把 Ray 理解成“比原 RPC server 更适合继续往分布式方向扩”的那一层。

## 8. 后续扩展方向

当前 `FullModelWorkerActor` 是“完整模型单节点 actor”。后面扩展分布式时，可以继续沿这个接口拆成：

- `StageWorkerActor`
- `PrefillWorkerActor`
- `DecodeWorkerActor`
- `PipelineCoordinator`

这样可以保持上层调度逻辑稳定，把平台差异继续留在 `bindings/` 和底层 engine 实现里。
