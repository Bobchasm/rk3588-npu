本文档描述当前项目的运行步骤，包括本地环境设置、将文字转换为 token id、在 RK3588 板子上测试推理、以及将输出解码回文字。

# 单板 Worker

## 1. 本地环境设置

### 编译工具

本地安装交叉编译器(wsl-ubuntu22.04)

```bash
sudo apt-get install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu
```

### 依赖安装 (或者使用conda环境)

- Python 3.8+
- transformers 库：`pip install transformers`
- torch（可选，用于本地测试）：`pip install torch`

### 模型准备

模型位于 `models/qwen1.5b-instruct/Qwen2-1.5B-Instruct/`，包含：
- model.safetensors（权重）
- tokenizer_config.json 等 tokenizer 文件

## 2. 将文字转换为 Token ID

在本地使用 Python 将输入的文字转换为 token id：

```bash
python3 -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('models/qwen1.5b-instruct/Qwen2-1.5B-Instruct')
msgs = [{'role':'user','content':'你好'}]
text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
ids = tok.encode(text)
print('./qwen2_demo Qwen1.5B', ' '.join(map(str, ids)))
"
```

输出示例：
```
输入文本: <|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
你好，请介绍一下自己<|im_end|>
<|im_start|>assistant

Token IDs: [151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645, 151644, 872, 198, 108386, 101055, 107748, 104968, 151645, 151644, 77091, 198]
```

## 3. 在 RK3588 板子上测试推理

### 当前默认后端

当前 worker 仍然只接收 token id，不做 tokenizer 和文本解码。默认计算路径如下：

- NPU：qkv fused projection、o_proj、gate/up fused projection、down_proj、lm_head
- CPU：Embedding、RMSNorm、RoPE、Attention/Softmax、SiLU、残差、采样、格式转换
- 多核 NPU：默认使用自动分块器，超过规模阈值且 N 维满足对齐的矩阵乘会按 N 维切成 3 片，分别绑定 3 个 NPU core 并行执行
- RKNN matmul 仍按 `M=1` 逐 token 调用；不要直接改成 `M=seq` 批量输入，否则可能得到错误 logits

### 编译和部署

1. 在本地交叉编译：
   ```bash
   # /worker
   ./build-linux.sh
   ```

2. 每次重新编译完将 `/worker/build/aarch64/install/` 下除.so的文件替换板子上 `/22_04_rootfs/root/matmul/worker_test/` 。

3. 设置权限
   ```bash
   # /root/matmul/worker_test
   chmod +x qwen2_demo
   chmod +x qwen2_chat
   ```

### 运行推理

使用 `qwen2_demo` 进行单次推理测试：

```bash
cd /root/matmul/worker_test
# 一次性的
./qwen2_demo Qwen1.5B 151644 8948 198 2610 525 264 10950 17847 13 151645 151644 872 198 108386 101055 107748 104968 151645 151644 77091 198
# 加载好后可以多次输入token ids
./qwen2_chat Qwen1.5B
# 然后可以输：151644 8948 198 2610 525 264 10950 17847 13 151645 151644 872 198 108386 101055 107748 104968 151645 151644 77091 198
```

参数说明：
- 第一个参数：模型目录路径
- 后续参数：输入 token id 列表 (步骤2生成)

程序会输出生成的 token id 序列。

### 性能 profile

开启 `RKLLM_PROFILE=1` 可以打印每次 forward 的阶段耗时：

```bash
cd /root/matmul/worker_test
RKLLM_PROFILE=1 ./qwen2_demo Qwen1.5B 151644 8948 198 2610 525 264 10950 17847 13 151645 151644 872 198 108386 101055 107748 104968 151645 151644 77091 198
```

重点关注字段：
- `qkv`：融合后的 q/k/v projection
- `gate_up`：融合后的 gate/up projection
- `down`：MLP down projection
- `lm_head`：输出词表 projection
- `attention`、`silu_mul`：当前仍在 CPU 上执行

当前优化后的参考结果会随板子负载波动。以 27 个输入 token、生成 10 个 token 为例，decode 通常在 `4 tok/s` 以上；若输出 token ids 与基准不一致，优先检查是否误用了 `M=seq` 批量 matmul 路径。

自动分块器可以通过环境变量临时调整：

```bash
# 关闭自动分片，默认 NPU 后端退回单核
RKLLM_NPU_AUTO_SHARD=0 ./qwen2_demo Qwen1.5B ...

# 调整触发三核分片的 K*N 阈值，默认 3000000
RKLLM_NPU_SHARD_MIN_OPS=3000000 ./qwen2_demo Qwen1.5B ...
```

### lm_head 后端选择

默认 `lm_head` 使用 NPU 自动后端。由于 `lm_head` 矩阵很大，自动分块器会选择三核 NPU 分片：

```bash
./qwen2_demo Qwen1.5B ...
```

如果板子出现 NPU/CMA 内存分配失败，可以临时切到 CPU fallback，速度会明显下降，但有助于验证稳定性：

```bash
RKLLM_LM_HEAD_BACKEND=CPU ./qwen2_demo Qwen1.5B ...
```

可选值：
- `NPU`：自动规划，默认
- `NPU_SHARDED`：强制三核 NPU 分片
- `NPU_SINGLE`：单 NPU matmul，上板排查时使用
- `CPU`：CPU fallback，主要用于避开 lm_head 大块 NPU 内存分配失败

### 常见问题

如果加载或运行时报下面这类错误：

```text
failed to allocate handle
[NpuLinear] rknn_create_mem(B) failed
failed to convert handle to fd
Too many open files
```

可以先在板子当前 shell 检查并提高 fd 限制后再运行。注意：`ulimit` 单独执行显示的是文件大小限制，不是 open files；这里必须看 `ulimit -n`。

```bash
ulimit -n
ulimit -n 4096
export LD_LIBRARY_PATH=/root/matmul/worker_test:$LD_LIBRARY_PATH
RKLLM_PROFILE=1 ./qwen2_demo Qwen1.5B ...
```

如果仍然失败，先重启板子释放 NPU handle/CMA，再用 `RKLLM_LM_HEAD_BACKEND=CPU` 判断是否是 `lm_head` 大权重分配问题。

## 4. 将板子输出解码回文字

### 本地解码

将板子输出的 token id 复制到本地，使用 Python 解码：

```bash
python3 -c "
from transformers import AutoTokenizer
tkz = AutoTokenizer.from_pretrained('models/qwen1.5b-instruct/Qwen2-1.5B-Instruct')
ids = [108386,6313,104139,109944,100364,103929,101037,11319]  # 替换为实际输出
print(tkz.decode(ids))
```

## 5. 本地基准测试

在本机上运行，得到标注token输出对比

```bash
# /rk3588-npu
python3 -c "
import torch, time
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
tkz = AutoTokenizer.from_pretrained('models/qwen1.5b-instruct/Qwen2-1.5B-Instruct')
model = AutoModelForCausalLM.from_pretrained('models/qwen1.5b-instruct/Qwen2-1.5B-Instruct', torch_dtype=torch.bfloat16, device_map='cpu', low_cpu_mem_usage=True)
msgs = [{'role':'user','content':'你好'}]
text = tkz.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
ids = tkz.encode(text, return_tensors='pt')
streamer = TextStreamer(tkz, skip_prompt=True)
t0 = time.time()
with torch.no_grad():
    out = model.generate(ids, max_new_tokens=10, do_sample=False, temperature=None, top_p=None, pad_token_id=tkz.eos_token_id, streamer=streamer)
t1 = time.time()
new_ids = out[0][ids.shape[1]:].tolist()
print('token ids:', new_ids)
print(f'耗时: {t1-t0:.1f}s，平均 {10/(t1-t0):.2f} tok/s')
"
```


# 引入调度器后的系统运行

## 1. 环境

使用脚本安装 Python 依赖

```bash
chmod +x scripts/setup_python_deps.sh
./scripts/setup_python_deps.sh
```


## 2. 编译运行 worker

### 本地编译

```bash
cd ~/worker
./build-linux.sh
```

输出目录：
- `~/rk3588-npu/worker/install/qwen2_demo`
- `~/rk3588-npu/worker/install/qwen2_chat`
- `~/rk3588-npu/worker/install/worker_rpc_server`
- `~/rk3588-npu/worker/install/librknnrt.so`

> 如果 `./build-linux.sh` 报 "请设置 GCC_COMPILER"，请先执行：
> ```bash
> export GCC_COMPILER=aarch64-linux-gnu
> ```

### 将产物传到板子

`worker/install/worker_rpc_server` 传到板子 `/root/matmul/worker_test/` 下



### 板子上运行 worker

```bash
./worker_rpc_server Qwen1.5B 0.0.0.0:5001
```

## 3. 本地运行调度器

### conda环境

```
# 方式一：直接用脚本
./scripts/setup_python_deps.sh

# 方式二：手动
conda create -y -n rk3588 python=3.10
conda activate rk3588
pip install -r requirements.txt
```


### 在本地 WSL 构建 scheduler

```bash
cd scheduler
mkdir -p build && cd build
cmake .. -DSCHEDULER_USE_WORKER_CORE=OFF
cmake --build . -- -j4
```

### 运行 scheduler CLI 指向板子 RPC 服务

可能需要在win上开一个隧道，win的cmd输入以下命令后输入密码然后不动终端

```bash
ssh -N -g -L 0.0.0.0:5001:127.0.0.1:5001 root@172.28.9.59
```

然后运行，出现 `>` 后可以直接输入文本

```bash
cd /scheduler/build
./scheduler_cli ~/rk3588-npu/models/qwen1.5b-instruct/Qwen2-1.5B-Instruct 172.26.0.1:5001
```

### 多会话使用方法

这条 `scheduler_cli -> worker_rpc_server` 路径可以直接在一个 CLI 里使用多个会话。

不需要：

- 多起几个调度器
- 多起几个 worker

保持前面的启动方式不变，进入 CLI 后直接使用下面这些命令即可。

CLI 每轮会打印：

- `output_tokens`
- `prefill_ms`
- `decode_ms`
- `decode_tok_s`

其中 `decode_tok_s` 的计算口径与 `qwen2_demo` 一致，都是：

```text
decode_tokens / (decode_ms / 1000)
```

### 常用命令

- `/new`
  创建一个新会话，并自动切换到新会话

- `/new <system_prompt>`
  创建一个带自定义 system prompt 的新会话

- `/sessions`
  查看当前已有的全部会话

- `/switch <session_id>`
  切换到指定会话，会话id格式为提示符前面的字符串

- `/history`
  查看当前会话的历史消息

- `/reset`
  重置当前会话，恢复默认 system prompt

- `/reset <system_prompt>`
  重置当前会话，并替换为新的 system prompt

- `/max_tokens <n>`
  设置后续每轮请求的最大生成 token 数

### 基本使用流程

启动后默认会进入第一个会话，例如：

```text
Scheduler CLI started. Session=session-1. Enter text to generate, or type /help.
[session-1] >
```

此时可以直接输入文本开始对话。

如果想新开一个独立会话：

```text
/new
```

创建后会自动切到新会话，例如：

```text
Created session session-2
[session-2] >
```

如果想查看当前有哪些会话：

```text
/sessions
```

如果想切回旧会话：

```text
/switch session-1
```

如果想看当前会话里之前问过什么：

```text
/history
```

如果想把当前会话清空重新开始：

```text
/reset
```


# 全模型 Ray 方案启动

这一节只说明当前已经跑通的 `pc` 路径，也就是：

```text
用户文本 -> scheduler / client -> Ray actor -> Python binding -> worker-pc -> token 输出
```

当前 `rk3588` 路径仍在验证中，不放进这里的主运行流程。

## 1. 环境准备

建议使用 `rk3588` conda 环境：

```bash
conda activate rk3588
pip install -r requirements.txt
```

如果还没有环境：

```bash
conda create -y -n rk3588 python=3.10
conda activate rk3588
pip install -r requirements.txt
```

## 2. 构建 PC 版 binding

```bash
rm -rf bindings/build
rm -f bindings/python/runtime/pc_engine*.so
cmake -S bindings -B bindings/build
cmake --build bindings/build -j4
```

成功后，`bindings/python/runtime/` 下应出现 `pc_engine` 对应的 `.so` 文件。

## 3. 先直接验证 worker-pc

```bash
PYTHONPATH=bindings/python python3 - <<'PY'
from runtime import create_engine
engine = create_engine("pc")
engine.load("models/qwen1.5b-instruct/Qwen2-1.5B-Instruct", "gpu")
result = engine.generate([151644, 8948, 198], max_new_tokens=4)
print(result)
engine.destroy()
PY
```

如果本机 GPU 不可用，可以先把 `"gpu"` 换成 `"cpu"` 或 `"auto"`。

## 4. 启动 Ray 常驻 worker

```bash
PYTHONPATH=bindings/python \
RAY_memory_monitor_refresh_ms=0 \
RAY_memory_usage_threshold=0.99 \
python3 ray_runtime/serve_worker.py \
  models/qwen1.5b-instruct/Qwen2-1.5B-Instruct \
  --target pc \
  --device gpu \
  --actor-name pc-full-model
```

说明：

- `--target pc`：当前只跑通 PC 实现
- `--device gpu`：明确走 GPU
- `--device auto`：自动探测 GPU，没有则回退 CPU
- `pc-full-model`：这个 actor 的名字，可自定义

首次启动会有模型冷加载，耗时明显长于后续请求，属于正常现象。

## 5. 从另一个终端发送请求

```bash
PYTHONPATH=bindings/python python3 ray_runtime/generate_request.py \
  --actor-name pc-full-model \
  151644 8948 198
```

如果想直接用固定测试输入，也可以：

```bash
PYTHONPATH=bindings/python python3 ray_runtime/generate_request.py \
  --actor-name pc-full-model \
  151644 8948 198 2610 525 264 10950 17847 13 151645 198 151644 872 198 108386 37945 109432 107828 151645 198 151644 77091 198
```

返回结果是 JSON，重点看：

- `output_ids`
- `prefill_ms`
- `decode_ms`
- `elapsed_ms`
- `request_count`

如果 `request_count` 递增而 Ray 服务端没有重新打印 `load begin`，说明模型确实只加载了一次。

## 6. 查看和停止 actor

查看状态：

```bash
PYTHONPATH=bindings/python python3 ray_runtime/actor_status.py \
  --actor-name pc-full-model
```

停止服务：

```bash
PYTHONPATH=bindings/python python3 ray_runtime/stop_worker.py \
  --actor-name pc-full-model
```

如果 `serve_worker.py` 是前台启动的，也可以直接在那个终端按 `Ctrl+C`。

## 7. 一次性验证

如果只是想快速验证整条 Ray 链，而不需要常驻加载，可以直接用：

```bash
PYTHONPATH=bindings/python python3 ray_runtime/single_worker_demo.py \
  models/qwen1.5b-instruct/Qwen2-1.5B-Instruct \
  --target pc \
  --device gpu \
  151644 8948 198
```

它会启动临时 actor，跑完一次请求后退出。
