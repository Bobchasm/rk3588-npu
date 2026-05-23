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
# /root/matmul/worker_test
# 一次性的
./qwen2_demo Qwen-1.5B 151644 8948 198 2610 525 264 10950 17847 13 151645 151644 872 198 108386 101055 107748 104968 151645 151644 77091 198
# 加载好后可以多次输入token ids
./qwen2_chat Qwen-1.5B
# 然后可以输：151644 8948 198 2610 525 264 10950 17847 13 151645 151644 872 198 108386 101055 107748 104968 151645 151644 77091 198
```

参数说明：
- 第一个参数：模型目录路径
- 后续参数：输入 token id 列表 (步骤2生成)

程序会输出生成的 token id 序列。

### 可选：板子空间不足时，直接远端随机读取模型

1.用 Docker 在本地PC上启动 Nginx (/models/qwen1.5b-instruct/Qwen2-1.5B-Instruct换成自己存权重的路径，wsl/win都可以)

```bash
docker run -d \
  --name rk3588-files \
  -p 8585:80 \
  -v /models/qwen1.5b-instruct/Qwen2-1.5B-Instruct:/usr/share/nginx/html:ro \
  nginx:stable-alpine
```
2.win上开一个终端，电脑上建立反向隧道

```bash
ssh -N -R 9000:127.0.0.1:8585 root@172.28.9.59
# 然后输密码就ok
```

3.板子上推理
```bash
./qwen2_demo http://127.0.0.1:9000/model.safetensors 151644 8948 198 ...
./qwen2_chat http://127.0.0.1:9000/model.safetensors
```

#### 远端随机读取相关说明

- 这种方式缓解的是“板子磁盘空间不够”的问题，不会减少模型加载到内存/NPU 时的内存占用。
- 当前实现会在内存中做小块缓存，不会把整个模型落盘到板子磁盘。
- 如果程序运行在 `chroot` 环境中，宿主机里的 `curl/wget` 不一定对程序可见；当前实现已经支持：
  - `curl/wget` 命令读取
  - 明文 `http://` 的内建 socket fallback
- 如果后续改回“板子本地目录”或“板子挂载的网络文件系统目录”，继续传原来的本地路径即可，不需要改上层接口。
- 远端读取相关调试与缓存参数：

```bash
export RKLLM_HTTP_DEBUG=1
export RKLLM_HTTP_CACHE_BLOCK_MB=1
export RKLLM_HTTP_CACHE_MAX_BLOCKS=16
```

含义：
- `RKLLM_HTTP_DEBUG=1`：打印远端读取调试信息
- `RKLLM_HTTP_CACHE_BLOCK_MB`：每次远端读取的块大小，默认 `1MB`
- `RKLLM_HTTP_CACHE_MAX_BLOCKS`：最多缓存多少个块，默认 `16`

### 可选：使用逻辑模型名切换本地/远端来源

如果你不想每次都改命令行参数，可以把第一个参数固定写成逻辑模型名，例如 `Qwen1.5B`，
然后通过环境变量切换模型来源。下面仍然以 `qwen2_demo / qwen2_chat` 为例：

```bash
# 本地目录模式
export RKLLM_MODEL_SOURCE_MODE=local
export RKLLM_MODEL_LOCAL_ROOT=/root/matmul/worker_test
./qwen2_demo Qwen1.5B 151644 8948 198 ...
```

```bash
# 远端 HTTP 模式
export RKLLM_MODEL_SOURCE_MODE=http
export RKLLM_MODEL_REMOTE_BASE=http://127.0.0.1:9000
./qwen2_demo Qwen1.5B 151644 8948 198 ...
```

此时 worker 会自动把 `Qwen1.5B` 映射为：
- 本地：`/root/matmul/worker_test/Qwen1.5B/model.safetensors`
- 远端：`http://127.0.0.1:9000/Qwen1.5B/model.safetensors`


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
