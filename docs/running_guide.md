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

### 在本地 WSL 构建 scheduler

```bash
cd scheduler
mkdir -p build && cd build
cmake .. -DSCHEDULER_USE_WORKER_CORE=OFF
cmake --build . -- -j4
```

### 运行 scheduler CLI 指向板子 RPC 服务

```bash
cd /scheduler/build
./scheduler_cli /models/qwen1.5b-instruct/Qwen2-1.5B-Instruct <rk3588 ip>:5001
```