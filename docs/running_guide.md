# RK3588 NPU Worker 运行指南

本文档描述当前项目的运行步骤，包括本地环境设置、将文字转换为 token id、在 RK3588 板子上测试推理、以及将输出解码回文字。

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
- 多核 NPU：qkv、gate/up、down、lm_head 使用手动 N 维分片，分别绑定 3 个 NPU core 并行执行
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

### lm_head 后端选择

默认 `lm_head` 使用三核 NPU 分片后端：

```bash
./qwen2_demo Qwen1.5B ...
```

如果板子出现 NPU/CMA 内存分配失败，可以临时切到 CPU fallback，速度会明显下降，但有助于验证稳定性：

```bash
RKLLM_LM_HEAD_BACKEND=CPU ./qwen2_demo Qwen1.5B ...
```

可选值：
- `NPU` / `NPU_SHARDED`：三核 NPU 分片，默认
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

可以先在板子当前 shell 提高 fd 限制后再运行：

```bash
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
