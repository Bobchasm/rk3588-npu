# Week2

| 单节点 RK3588 NPU 推理基线，以qwen1.5B为例

## 1 路线选择

本阶段采用 **rknn_matmul_api 路线**（而非 RKNN 全模型转换路线），原因如下：

| 对比项 | RKNN 全模型路线 | MatMul API 路线（本项目） |
|---|---|---|
| 输入 shape | 编译时固定 | 运行时动态，支持任意 seq_len |
| LLM 适用性 | 差（静态 shape 是 RKNN compiler 的软件限制） | 好 |
| 格式转换 | 需要 HuggingFace → ONNX → .rknn | 不需要，直接读 safetensors |
| NPU 调用方式 | `rknn_run()` | `rknn_matmul_run()` |
| 计算图管理 | RKNN runtime 自动 | C++ 手动实现 |

## 2 调用栈

```
应用层（token id 输入）
    ↓
C++ 推理执行器（qwen2.cpp）
    ├── CPU 算子：RMSNorm / RoPE / Softmax / SiLU / 残差
    └── NPU 算子：所有 Linear 层（通过 rknn_matmul_run）
        ↓
librknnrt.so（RKNN 运行时）
        ↓
RK3588 NPU 硬件
```

---

## 3 模型结构分析（Qwen1.5B）

- 架构：`Qwen2ForCausalLM`
- 层数：28 层 Transformer Block
- 关键参数：

| 参数 | 值 |
|---|---|
| hidden_size | 1536 |
| num_attention_heads | 12（GQA） |
| num_kv_heads | 2 |
| head_dim | 128 |
| intermediate_size | 8960 |
| vocab_size | 151936 |

## 4 NPU / CPU 算子分布（每层）

| 算子 | 运行位置 | 备注 |
|---|---|---|
| q/k/v_proj | NPU | 有 bias，单独加 |
| o_proj | NPU | 无 bias |
| gate/up_proj | NPU | 无 bias |
| down_proj | NPU | 无 bias |
| RMSNorm | CPU | |
| RoPE | CPU | 使用绝对位置编码 |
| Attention score（QK^T） | CPU | seq × total_len |
| Softmax | CPU | |
| score × V | CPU | |
| SiLU | CPU | |
| 残差加法 | CPU | |
| lm_head | NPU | K=1536, N=151936 |

**K 维度对齐验证**（NPU 硬件要求 K % 32 == 0）：

| 矩阵乘 | K | 对齐 |
|---|---|---|
| q/k/v/o_proj | 1536 | ✅ |
| gate/up_proj | 1536 | ✅ |
| down_proj | 8960 | ✅ |
| lm_head | 1536 | ✅ |

---

## 5 Worker 代码结构

```
worker/
├── build-linux.sh          # 一键交叉编译脚本
├── CMakeLists.txt
├── include/
│   ├── half.h              # BF16 / FP16 / FP32 互转
│   ├── weight_loader.h     # safetensors 解析接口
│   ├── cpu_ops.h           # CPU 算子接口
│   ├── npu_matmul.h        # NpuLinear 封装
│   └── qwen2.h             # 模型结构 / KV Cache
└── src/
    ├── weight_loader.cpp   # safetensors 解析（无第三方依赖）
    ├── cpu_ops.cpp         # RMSNorm / RoPE / Softmax / SiLU
    ├── npu_matmul.cpp      # rknn_matmul_create/run/destroy
    ├── qwen2.cpp           # 28 层前向推理 + KV Cache
    └── main.cpp            # 入口，Prefill + Decode 分离
```

### 5.1 NpuLinear

每个 Linear 层封装为一个 `NpuLinear` 对象：

- 构造时：`rknn_matmul_create` → 加载权重（BF16 → FP16 → native B layout）
- 推理时：`rknn_matmul_run`（M=1，每 token 单独调用）
- 析构时：自动 `rknn_matmul_destroy`

**重要约束**：NPU 上下文以 M=1 创建，不能批量输入多行，否则只计算第一行，其余为零。

### 5.2 KV Cache

```
kv_cache.k_cache[layer][pos * kv_dim + d]  // FP16
kv_cache.v_cache[layer][pos * kv_dim + d]  // FP16
kv_cache.cur_pos                            // 当前已写入位置数
```

- Prefill：输入全部 token，写入 cache，返回第一个生成 token
- Decode：每步输入 1 个 token，读取完整历史 cache，写入新 K/V

### 5.3 Prefill / Decode 分离

```
main.cpp 流程：

[Prefill] forward(input_ids)   → next_id
    ↓
[Decode] for step in range(max_new_tokens):
    emit next_id
    forward({next_id})         → next_id（单 token，约 380ms/step）
```

---

## 6 关键问题与解决方案

1. M=1 约束导致全量输出错误

**现象**：所有 step 输出相同 top-5，logits 值固定不变。

**原因**：`NpuLinear` 以 M=1 创建上下文，喂入 M=20 时只计算第一行，其余为零。

**修复**：对 seq 中每个 token 单独调 NPU（循环 M=1 调用）。

2. NPU handle 泄漏导致反复运行失败

**现象**：第 1-2 次运行正常，第 3+ 次在加载 lm_head 时报 `failed to allocate handle`。

**原因**：
1. `rknn_create_mem(B)` 失败时，`ctx` 未调用 `rknn_matmul_destroy` → 每次失败泄漏 1 个 handle
2. lm_head 的 B 矩阵需要 **467MB 连续 CMA 内存**，内核 CMA 回收不及时

**修复**：在 `NpuLinear::init()` 的错误路径中显式 `rknn_matmul_destroy(ctx)`。

3. 无 KV Cache 时性能退化

**现象**：每步耗时从 6s 增长到 9s（随 token 增加）。

**原因**：每步重算全部历史 token，O(n²) 复杂度。

**修复**：实现 KV Cache，decode 阶段每步只处理 1 个新 token，耗时稳定在 ~380ms。

4. 精度差异导致后续 token 不一致

**现象**：板子与 WSL 前 1-2 个 token 相同，后续不同。

**原因**：权重从 BF16 转换为 FP16 时精度损失，误差逐步累积。

**结论**：属于正常精度差异，不是 bug。验证方法：对应步骤的 token 互在对方 top-5 中即视为通过。

---

5. 性能测试结果

测试输入：`你好`（~20 input tokens），生成 10 个新 token。

| 环境 | 耗时 | 吞吐量 |
|---|---|---|
| RK3588 NPU（本项目） | 3.8 s | **2.6 tok/s** |
| WSL CPU（PyTorch BF16） | 28.4 s | 0.35 tok/s |
| **加速比** | - | **7.4x** |

> WSL 使用多核 x86 CPU，故加速比约 7x（而非理论峰值 200x）。

---

## 7 常用命令

### 7.1 环境配置

```bash
# 激活 conda 环境，wsl有python/conda环境就行，主要用于正确性对照
source ~/miniforge3/etc/profile.d/conda.sh && conda activate rk3588

# 安装交叉编译器
sudo apt-get install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu
```

### 7.2 编译与部署

```bash
# 交叉编译（在 WSL 中执行）
cd /home/deep/d-rk3588-npu/worker
./build-linux.sh
```

上传 `worker/install/qwen2_demo` 到板子

板子上目录结构

```
/root/matmul/worker_test/
  ├── qwen2_demo  # 重新编译后替换
  ├── librknnrt.so
  └── Qwen1.5B/
      └── model.safetensors
```

### 7.3 Tokenizer 工具

```bash
# 获取输入 token id（供板子使用）
source ~/miniforge3/etc/profile.d/conda.sh && conda activate rk3588
cd /home/deep/d-rk3588-npu
python3 -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('model/Qwen1.5B')
msgs = [{'role':'user','content':'你好'}]
text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
ids = tok.encode(text)
print('./qwen2_demo Qwen1.5B', ' '.join(map(str, ids)))
"

# 解码板子输出的 token id
python3 -c "
from transformers import AutoTokenizer
tkz = AutoTokenizer.from_pretrained('model/Qwen1.5B')
ids = [108386, 3837, 35946, 85106, 46944]  # 替换为实际输出
print(tkz.decode(ids))
"
```

### 7.4 板子上运行

```bash
cd /root/matmul/worker_test
export LD_LIBRARY_PATH=/root/matmul/worker_test:$LD_LIBRARY_PATH
./qwen2_demo Qwen1.5B 151644 8948 198 ...  # 替换为实际 token ids
```

### 7.5 NPU 故障恢复

```bash
# handle 耗尽时重载驱动（板子上执行）
rmmod rknpu && modprobe rknpu

# 若 rmmod 提示 module in use，先杀进程
pkill qwen2_demo
sleep 2
rmmod rknpu && modprobe rknpu

# 以上均不行则重启
reboot
```

### 7.6 WSL CPU 基准测试

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate rk3588
cd /home/deep/d-rk3588-npu
python3 -c "
import torch, time
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
tkz = AutoTokenizer.from_pretrained('model/Qwen1.5B')
model = AutoModelForCausalLM.from_pretrained('model/Qwen1.5B', torch_dtype=torch.bfloat16, device_map='cpu', low_cpu_mem_usage=True)
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

---

## 8 下一步

- [ ] lm_head 迁移到 CPU（解决 467MB CMA 分配不稳定问题）
- [ ] 多 NPU core 调度（RK3588 有 3 个 core，理论再提速 2-3x）
- [ ] 模型切分（Shard 0~3，为多节点做准备）
- [ ] Worker RPC 通信框架（阶段 4）
