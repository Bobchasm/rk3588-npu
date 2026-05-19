# 远程运行指南 — 在 WSL 上运行调度器，板子（RK3588）上运行 worker

本文档给出最小可复现的流程：在 WSL（或任意 Linux 主机）上运行调度器（scheduler），在 RK3588 开发板上运行 worker 的 RPC 服务。包含构建、运行、快速检查、SSH 隧道以及后续分层 worker 的扩展说明。

概览
- 调度器在 WSL 上运行并向 worker 发送简单的行文本 TCP RPC（默认端口 5001）。
- worker 在 RK3588 板子上运行真实推理（使用 RKNN），负责执行 `PREFILL` / `GENERATE`（可扩展成 `STAGE` / `DECODE`）。
- 分词（tokenizer）在调度器端本地执行，使用 `scheduler/tools/tokenizer.py`（需要 `transformers`）。

相关可执行/脚本
- 调度器：`scheduler/build/scheduler_cli`
- worker RPC 服务：`worker/build/worker_rpc_server`
- Tokenizer：`scheduler/tools/tokenizer.py`

先决条件
- 板子（RK3588）：已配置 RKNN SDK，可在板子上编译 `worker` 并链接 RKNN（`-DBUILD_WITH_RKNN=ON`）。
- WSL：安装 `cmake`、`g++`、Python 3、`pip`，并能访问板子网络端口（或使用 SSH 隧道）。

在 RK3588（板子）上构建并运行 worker
1. 将仓库拷贝到板子并确保交叉/本地编译工具链与 RKNN 可用。
2. 构建 worker（启用示例程序与 RKNN）：

```bash
cd /home/deep/rk3588-npu/worker
mkdir -p build && cd build
cmake .. -DBUILD_WORKER_APPS=ON -DBUILD_WITH_RKNN=ON
cmake --build . -- -j4
```

3. 启动 worker RPC 服务（示例在所有接口监听 5001 端口）：

```bash
# 在板子上运行
./worker_rpc_server /path/to/model_dir 0.0.0.0:5001
```

成功启动后应看到类似输出：

```
# 运行准备与一步步操作（WSL 主机 + RK3588 板子）

下面是一份可直接执行的、按步骤整理的运行说明（复制粘贴即可）。将所有示例路径中的 `~/rk3588-npu`、`/path/to/model_dir`、`192.168.1.50` 等替换为你实际的路径与 IP。

重要前提
- 本说明假定本地（WSL）有一个 conda 环境名为 `rk3588`，后续 Python 测试均使用该环境。
- 板子上需要 RKNN 运行时（用于 NPU 加速），在板子上构建 worker 时使用 `-DBUILD_WITH_RKNN=ON`。

仓库中新增文件（已提交）
- `requirements.txt`：项目 Python 依赖，位于仓库根目录。
- `scripts/setup_python_deps.sh`：自动在 `rk3588` conda 环境安装依赖的脚本。

步骤 1 — 在 WSL 上安装 Python 依赖（仅需一次）

```bash
cd ~/rk3588-npu
chmod +x scripts/setup_python_deps.sh
./scripts/setup_python_deps.sh
```

说明：脚本会使用 `conda`，若 `rk3588` 环境不存在会自动创建（Python 3.8），然后激活并通过 `pip install -r requirements.txt` 安装依赖。

步骤 2 — 在板子（RK3588）上构建并启动 worker

在板子终端（假设仓库在 `/home/deep/rk3588-npu`）：

```bash
cd /home/deep/rk3588-npu/worker
mkdir -p build && cd build
cmake .. -DBUILD_WORKER_APPS=ON -DBUILD_WITH_RKNN=ON
cmake --build . -- -j4

# 启动 RPC 服务（替换 model_dir 与端口）
./worker_rpc_server /path/to/model_dir 0.0.0.0:5001
```

成功示例输出：

```
Worker RPC server listening on 0.0.0.0:5001
Loaded model from /path/to/model_dir
```

步骤 3 — 在 WSL 上构建并运行调度器（使用 `rk3588` 环境）

在 WSL：

```bash
cd ~/rk3588-npu
conda activate rk3588

# 检查本地 tokenizer
python3 scheduler/tools/tokenizer.py --help || true

# 构建 scheduler（不需要 RKNN）
cd scheduler
mkdir -p build && cd build
cmake .. -DSCHEDULER_USE_WORKER_CORE=OFF
cmake --build . -- -j4

# 运行调度器 CLI，指向板子 IP（示例：192.168.1.50:5001）
cd ~/rk3588-npu/scheduler/build
./scheduler_cli ~/rk3588-npu/models/qwen1.5b-instruct/Qwen2-1.5B-Instruct 192.168.1.50:5001
```

步骤 4 — 一次端到端示例（输入 → 输出）

在 `scheduler_cli` prompt 输入：

```
Hello, how are you?
```

流程概览：本地 tokenizer 将文本转为 token ids → scheduler 发送 `GENERATE` RPC 到板子 → 板子推理返回 token ids → scheduler 使用本地 tokenizer 解码并显示文本。

步骤 5 — 手动检查 RPC 协议（快速验证）

```bash
# 检查端口
nc -zv 192.168.1.50 5001

# PING（期望返回 PONG|0）
printf 'PING|0|0|\n' | nc 192.168.1.50 5001

# 手动 GENERATE 请求示例
printf 'GENERATE|1|4|1,2,3\n' | nc 192.168.1.50 5001
```

示例返回：

```
OK|1|12.3|45.6|100,101,102
```

说明字段含义：`OK|<request_id>|<prefill_ms>|<decode_ms>|<comma-separated-token-ids>`。

步骤 6 — 使用 SSH 隧道（当主机不能直接访问板子网络时）

在 WSL 建立隧道并在本地运行：

```bash
ssh -L 5001:localhost:5001 user@<board-ip>
# 然后指向 localhost:5001
cd ~/rk3588-npu/scheduler/build && ./scheduler_cli ~/rk3588-npu/models/qwen1.5b-instruct/Qwen2-1.5B-Instruct localhost:5001
```

常见问题快速排查
- 如果出现连接超时/被拒绝：确认 `worker_rpc_server` 是否正在运行并监听端口，检查板子防火墙。
- 如果模型加载失败：检查 `model_dir` 内容并确认 RKNN 可用。
- tokenizer 解码错误：确保 `local_tokenizer_dir` 与模型相匹配。

后续我可以继续：
- A) 添加自动化 end-to-end smoke-test（WSL 脚本），或
- B) 实现 `STAGE` / `DECODE` RPC 命令以支持分层 worker。

请选择 A 或 B，我将继续实现。