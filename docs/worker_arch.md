# RK3588 NPU Worker 架构文档

> 本文档描述 `/worker/` 目录的完整代码架构、各层职责、关键数据流与接口约定，供后续分工优化使用。

---

## 一、总体架构

整个 worker 分为 **5 层**，自底向上依次为：

```
┌─────────────────────────────────────────────────────────────────┐
│  入口层  │  qwen2_demo (main.cpp)   qwen2_chat (chat_main.cpp)  │
│          │  scripts/chat.py（Python 文字包装器）                 │
├─────────────────────────────────────────────────────────────────┤
│  API 层  │  LLMEngine + GenerationConfig / GenerationResult     │
│          │  include/api/   src/api/                             │
├─────────────────────────────────────────────────────────────────┤
│  模型层  │  Qwen2Model + KVCache + TransformerLayer + 权重加载  │
│          │  include/model/ src/model/                           │
├─────────────────────────────────────────────────────────────────┤
│  算子层  │  op_rmsnorm / op_rope / op_attention / op_linear ... │
│          │  include/ops/   src/ops/                             │
├──────────────────────────────────┬──────────────────────────────┤
│  后端层  │  NpuLinear（rknn_matmul）│ （预留 CPU GEMM 等）       │
│          │  include/backend/  src/backend/                      │
├──────────────────────────────────┴──────────────────────────────┤
│  Core    │  half.h  data_types.h  model_config.h                │
│          │  include/core/                                        │
└─────────────────────────────────────────────────────────────────┘
```

所有源文件按层放在对应子目录，不同层之间 **只向下依赖**，不向上引用。

---

## 二、目录结构

```
worker/
├── include/
│   ├── core/
│   │   ├── half.h              # BF16 / FP16 / FP32 互转（纯 header）
│   │   ├── data_types.h        # 全局类型别名（f16_t / weight_t / act_t）
│   │   └── model_config.h      # Qwen2Config 结构体
│   │
│   ├── ops/
│   │   ├── op_linear.h         # ILinearOp 抽象接口 + make_linear 工厂
│   │   ├── op_rmsnorm.h
│   │   ├── op_rope.h
│   │   ├── op_softmax.h
│   │   ├── op_silu.h
│   │   ├── op_elementwise.h
│   │   ├── op_cast.h
│   │   ├── op_embedding.h
│   │   ├── op_attention.h
│   │   └── op_sampling.h
│   │
│   ├── backend/
│   │   └── npu_linear.h        # NpuLinear : ILinearOp（rknn_matmul 后端）
│   │
│   ├── model/
│   │   ├── weight_loader.h     # safetensors 解析、张量加载
│   │   ├── kv_cache.h          # KVCache 类
│   │   ├── transformer_layer.h # TransformerLayer 权重容器
│   │   └── qwen2_model.h       # Qwen2Model（28 层 forward）
│   │
│   └── api/
│       ├── generation_config.h # GenerationConfig / GenerationResult
│       └── llm_engine.h        # LLMEngine（对外唯一入口类）
│
├── src/
│   ├── ops/        op_rmsnorm.cpp  op_rope.cpp  op_softmax.cpp
│   │               op_silu.cpp     op_elementwise.cpp  op_cast.cpp
│   │               op_embedding.cpp  op_attention.cpp  op_sampling.cpp
│   ├── backend/    npu_linear.cpp
│   ├── model/      weight_loader.cpp  kv_cache.cpp  qwen2_model.cpp
│   ├── api/        llm_engine.cpp
│   ├── main.cpp         # qwen2_demo 入口
│   └── chat_main.cpp    # qwen2_chat 入口（REPL）
│
├── CMakeLists.txt
└── build-linux.sh
```

---

## 三、Core 层

### `include/core/half.h`

纯 header，提供 BF16 / FP16 / FP32 之间的转换函数，**不依赖任何第三方库**：

| 函数 | 说明 |
|---|---|
| `bf16_to_f32(uint16_t)` | BF16 → FP32（高 16 位直接复制） |
| `f32_to_bf16(float)` | FP32 → BF16 |
| `f16_to_f32(uint16_t)` | IEEE 754 FP16 → FP32 |
| `f32_to_f16(float)` | FP32 → FP16（含次正规/Inf 处理） |

### `include/core/data_types.h`

集中定义全局类型别名，未来切换底层存储格式（如 BF16 换 FP16）只改这一处：

```cpp
namespace rk {
    using f16_t    = uint16_t;   // FP16 以 uint16_t 承载
    using weight_t = f16_t;      // 权重存储格式
    using act_t    = float;      // 激活在 CPU 上用 FP32
}
```

### `include/core/model_config.h`

描述 Qwen2 系列模型结构的超参数，以默认值覆盖 Qwen2-1.5B-Instruct：

```cpp
struct Qwen2Config {
    int   hidden_size         = 1536;
    int   num_hidden_layers   = 28;
    int   num_attention_heads = 12;
    int   num_kv_heads        = 2;     // GQA
    int   head_dim            = 128;
    int   intermediate_size   = 8960;
    int   vocab_size          = 151936;
    float rms_norm_eps        = 1e-6f;
    float rope_theta          = 500000.0f;
    int   max_position        = 512;   // KV Cache 最大容量

    int kv_dim() const { return num_kv_heads * head_dim; }  // = 256
};
```

> 新增模型时在 `include/core/` 下新建 `xxx_config.h` 即可，不影响现有代码。

---

## 四、算子层（ops）

### 设计原则

- 每个算子独占一个 `op_*.h` + `src/ops/op_*.cpp` 文件，**高内聚**；
- 所有算子都是**无状态纯函数**（除 `ILinearOp` 是有状态对象），便于单独测试和替换；
- CPU 算子直接操作 `float*`，NPU 接口统一经过 `ILinearOp` 抽象。

### 算子清单

#### `op_rmsnorm`
```cpp
void op_rmsnorm(const float* x, const float* weight, float* y,
                int seq, int hidden, float eps = 1e-6f);
```
- 实现 Root Mean Square LayerNorm：`y = x / rms(x) * weight`
- `rms(x) = sqrt(mean(x²) + eps)`
- 输入输出形状均为 `[seq, hidden]`

#### `op_rope`
```cpp
void op_rope(float* q, float* k,
             int n_heads, int n_kv_heads, int head_dim,
             int pos, float theta = 500000.0f);
```
- 对 Q 和 K 施加旋转位置编码（RoPE），**原地修改**；
- 支持 GQA：q 有 `n_heads` 个 head，k 只有 `n_kv_heads` 个；
- 按对相邻维度 `(2i, 2i+1)` 做旋转，频率 `freq_i = 1 / theta^(2i / head_dim)`。

#### `op_softmax`
```cpp
void op_softmax(float* x, int rows, int cols);
```
- 按行做数值稳定的 softmax（先减行内最大值），**原地修改**。

#### `op_silu`
```cpp
void op_silu(float* x, int n);
```
- SiLU 激活 `y = x / (1 + exp(-x))`，**原地修改**，用于 FFN gate。

#### `op_elementwise`
```cpp
void op_vec_add(float* dst, const float* src, int n);         // 残差累加
void op_vec_add_bias(float* x, const float* bias, int rows, int cols); // 加 bias
```

#### `op_cast`
```cpp
void op_f16_to_f32(const uint16_t* src, float* dst, int n);
void op_f32_to_f16(const float*    src, uint16_t* dst, int n);
```
- NPU 边界处的批量格式转换，**单独成文件**便于后续做 SIMD/NEON 向量化加速。

#### `op_embedding`
```cpp
void op_embedding_lookup(const uint16_t* table, const std::vector<int>& ids,
                         float* out, int hidden);
```
- 从 FP16 嵌入表中查行，转为 FP32 输出；
- `table` 形状 `[vocab, hidden]`，`out` 形状 `[seq, hidden]`。

#### `op_attention`
```cpp
void op_attention(
    const float*    q,          // [seq, n_heads * head_dim]  FP32
    const uint16_t* k_cache,    // [total_len, kv_dim]        FP16
    const uint16_t* v_cache,    // [total_len, kv_dim]        FP16
    float*          out,        // [seq, n_heads * head_dim]  FP32
    int seq, int total_len,
    int n_heads, int n_kv_heads, int head_dim,
    int pos_base);
```
- 实现因果 GQA 多头注意力（CPU）；
- 内部计算 `scores = (q · k^T) * scale`，施加 causal mask（`q_pos < k_pos` 时置 `-inf`），softmax，再乘 v；
- K/V 读自 KVCache（FP16），中间 scores 矩阵按需分配在栈/堆上。

#### `op_sampling`
```cpp
int op_greedy_sample(const std::vector<float>& logits);
```
- 贪心采样（argmax），预留位置后续加 `op_sample_topk` / `op_sample_topp` 等。

#### `op_linear`（抽象接口）

```cpp
class ILinearOp {
public:
    virtual bool init(int K, int N, const uint16_t* weight_kn) = 0;
    virtual bool forward(const uint16_t* input_f16, int M, uint16_t* output_f16) = 0;
    virtual void destroy() = 0;
};

enum class LinearBackend { NPU /*, NPU_MULTI_CORE, CPU_GEMM */ };

std::unique_ptr<ILinearOp> make_linear(LinearBackend backend = LinearBackend::NPU);
```

- 所有模型内的 Linear 层**都用** `ILinearOp` 声明，后端切换只改 `make_linear` 的参数；
- 权重约定：`[K, N]` FP16，即 PyTorch `[out_features, in_features]` 的**转置**。

---

## 五、后端层（backend）

### `NpuLinear : ILinearOp`

文件：`include/backend/npu_linear.h`，`src/backend/npu_linear.cpp`

基于 `rknn_matmul_api` 实现 FP16 矩阵乘法，计算 `C = A × B`：

| 张量 | 形状 | 数据类型 |
|---|---|---|
| A（输入激活） | `[M, K]` | FP16 |
| B（权重，初始化时固定） | `[K, N]` | FP16，native layout |
| C（输出） | `[M, N]` | FP16 |

**初始化流程（`init`）：**

1. `rknn_matmul_create` 创建上下文（初始 `M=1`）；
2. `rknn_create_mem` 分配 B 内存；
3. `rknn_B_normal_layout_to_native_layout` 将权重重排为 NPU native layout（性能更好）；
4. `rknn_matmul_set_io_mem` 绑定 B；
5. `rebuild_ac(1)` 预分配 A / C 内存。

**前向流程（`forward`）：**

1. 若 `M` 与当前不同，调用 `rebuild_ac(M)` 重建 A / C；
2. `memcpy` 将输入写入 A 内存；
3. **`rknn_matmul_run`** — 实际 NPU 执行；
4. `memcpy` 读出 C 内存到输出 buffer。

> 每个 `NpuLinear` 实例持有一个独立的 `rknn_matmul_ctx_t`，析构或 `destroy()` 时释放。

**工厂函数**（同文件实现）：
```cpp
std::unique_ptr<ILinearOp> make_linear(LinearBackend backend) {
    // 目前只有 NPU 后端
    return std::unique_ptr<ILinearOp>(new NpuLinear());
}
```

---

## 六、模型层（model）

### `weight_loader`

文件：`include/model/weight_loader.h`，`src/model/weight_loader.cpp`

**safetensors 格式**（无第三方依赖）：
```
[0, 8)       uint64  header_size
[8, 8+hs)    JSON 头（含每个张量的 dtype/shape/data_offsets）
[8+hs, ...)  所有张量二进制数据（连续存储）
```

提供两类函数：

```cpp
// 只解析文件头（不读张量数据），返回 name -> TensorMeta 映射
TensorMap load_safetensors_meta(const std::string& path);

// 读取指定张量，统一转为 FP16（支持 BF16/FP16/FP32 输入）
// transpose=true 时将 [rows, cols] 转置为 [cols, rows]（PyTorch 权重转 KN 格式）
std::vector<uint16_t> load_tensor_f16(const std::string& path,
                                      const TensorMeta& meta,
                                      bool transpose = false);

// 读取指定张量，转为 FP32（用于 norm weight 等标量参数）
std::vector<float>    load_tensor_f32(const std::string& path,
                                      const TensorMeta& meta);
```

### `KVCache`

文件：`include/model/kv_cache.h`，`src/model/kv_cache.cpp`

以 FP16 存储所有历史 K/V，每层各一块连续内存：

```
k_cache_[layer] : [capacity * kv_dim]  FP16
v_cache_[layer] : [capacity * kv_dim]  FP16
```

| 接口 | 说明 |
|---|---|
| `init(num_layers, capacity, kv_dim)` | 分配内存，`capacity` 对应 `max_position` |
| `reset()` | 清零所有 K/V，`cur_pos = 0`（新会话） |
| `cur_pos()` | 已写入的 token 位置数 |
| `k_ptr(layer)` / `v_ptr(layer)` | 各层底层指针（attention 算子直接访问） |

> **可替换性**：只要保持上述接口，即可替换为 PagedKVCache（block 粒度分配）、QuantizedKVCache（INT8 压缩）、SlidingWindowKVCache 等，上层代码无需改动。

### `TransformerLayer`

文件：`include/model/transformer_layer.h`

单层权重容器，**只存数据，不含 forward 逻辑**：

```cpp
struct TransformerLayer {
    // Attention
    std::vector<float>          input_layernorm;     // [hidden]
    std::unique_ptr<ILinearOp>  q_proj;              // [hidden → hidden]
    std::unique_ptr<ILinearOp>  k_proj;              // [hidden → kv_dim]
    std::unique_ptr<ILinearOp>  v_proj;              // [hidden → kv_dim]
    std::unique_ptr<ILinearOp>  o_proj;              // [hidden → hidden]
    std::vector<float>          q_bias, k_bias, v_bias;

    // FFN
    std::vector<float>          post_attention_layernorm;  // [hidden]
    std::unique_ptr<ILinearOp>  gate_proj;           // [hidden → intermediate]
    std::unique_ptr<ILinearOp>  up_proj;             // [hidden → intermediate]
    std::unique_ptr<ILinearOp>  down_proj;           // [intermediate → hidden]
};
```

所有 Linear 均为 `ILinearOp`，切换后端只需改 `make_linear` 调用参数，层结构本身不变。

### `Qwen2Model`

文件：`include/model/qwen2_model.h`，`src/model/qwen2_model.cpp`

**对外接口：**

```cpp
bool load(const std::string& model_dir, LinearBackend backend = LinearBackend::NPU);
void destroy();
void reset_kv_cache();
std::vector<float> forward(const std::vector<int>& tokens);
const Qwen2Config& config() const;
```

**`forward` 内部的单层计算流程**（共 28 次循环）：

```
① Input LayerNorm          CPU  op_rmsnorm
② Q/K/V proj               NPU  ILinearOp::forward（FP32→FP16→NPU→FP16→FP32）
③ 加 Q/K/V bias             CPU  op_vec_add_bias
④ RoPE                     CPU  op_rope（原地，每 token 独立计算）
⑤ 写入 KV Cache            CPU  f32_to_f16 写入当前位置
⑥ 因果 GQA Attention       CPU  op_attention（Q·Kᵀ / √d → causal mask → softmax → ·V）
⑦ O proj                   NPU  ILinearOp::forward
⑧ Residual add             CPU  op_vec_add
⑨ Post-Attn LayerNorm      CPU  op_rmsnorm
⑩ Gate/Up proj             NPU  ILinearOp::forward（各一次）
⑪ SiLU(gate) * up         CPU  op_silu + 逐元素乘
⑫ Down proj               NPU  ILinearOp::forward
⑬ Residual add             CPU  op_vec_add
```

**最终步骤（循环结束后）：**
```
⑭ Final LayerNorm（仅最后一个 token）  CPU  op_rmsnorm
⑮ lm_head                              NPU  ILinearOp::forward → [vocab_size] FP16
⑯ FP16 → FP32 logits                  CPU  op_f16_to_f32
⑰ 更新 kv_cache.cur_pos
```

**CPU / NPU 分工汇总：**

| 算子 | 执行位置 |
|---|---|
| q_proj / k_proj / v_proj / o_proj | NPU |
| gate_proj / up_proj / down_proj   | NPU |
| lm_head                           | NPU |
| RMSNorm / RoPE / Softmax / SiLU   | CPU |
| Attention score（Q·Kᵀ 及 ·V）    | CPU |
| Embedding lookup / cast / add     | CPU |
| 采样（argmax）                    | CPU |

---

## 七、API 层

### `GenerationConfig` / `GenerationResult`

文件：`include/api/generation_config.h`

```cpp
struct GenerationConfig {
    int  max_new_tokens = 10;
    bool greedy         = true;
    std::vector<int> stop_tokens = {151645, 151643};  // <|im_end|>, <|endoftext|>
    // 预留：float temperature; int top_k; float top_p;
};

struct GenerationResult {
    std::vector<int> output_ids;
    int   prefill_tokens, decode_tokens;
    float prefill_ms, decode_ms;
    bool  hit_stop;
};
```

### `LLMEngine`

文件：`include/api/llm_engine.h`，`src/api/llm_engine.cpp`

**对外接口：**

```cpp
bool             load(const std::string& model_dir,
                      LinearBackend backend = LinearBackend::NPU);
void             destroy();
void             reset();   // 清空 KV Cache，新会话前调用
GenerationResult generate(
    const std::vector<int>& input_ids,
    const GenerationConfig& cfg,
    TokenCallback on_token = nullptr);  // 流式回调（可选）
```

**`generate` 执行逻辑：**

```
Prefill：
  model.forward(input_ids)  → logits
  greedy_sample(logits)     → next_id

Decode 循环（最多 max_new_tokens 步）：
  if next_id in stop_tokens → hit_stop = true, break
  output_ids.push_back(next_id)
  model.forward({next_id})  → logits
  on_token 回调（流式输出）
  greedy_sample(logits)     → next_id
```

`TokenCallback` 签名：
```cpp
using TokenCallback = std::function<void(int step, int id, float elapsed_ms)>;
```

> **扩展说明**：  
> - vLLM-like 调度器只需调 `LLMEngine::generate`，无需了解 NPU 细节；  
> - 多请求 batch 在此层实现，不需要修改模型层；  
> - 采样策略在此层替换（greedy / top-k / top-p / beam search）。

---

## 八、入口层

### `qwen2_demo`（`src/main.cpp`）

保留原 demo 行为，供**验证分层实现正确性**：

```bash
./qwen2_demo <model_dir> <token_id1> [token_id2 ...]
```

内部直接构造 `LLMEngine`，调用 `generate`，打印每步耗时与最终 token id。

获取输入 token id：
```bash
python3 -c "
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained('path/to/Qwen2-1.5B-Instruct')
m = [{'role':'user','content':'你好'}]
s = t.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
print(' '.join(map(str, t.encode(s))))
"
```

### `qwen2_chat`（`src/chat_main.cpp`）

REPL 模式，**模型只加载一次，常驻进程**，通过 stdin/stdout 管道与 Python 端交互：

**stdin 协议（每行一条）：**

| 输入 | 含义 |
|---|---|
| `151644 872 108386 ...` | 空格分隔的 token id，触发一次生成 |
| `RESET` | 强制清空 KV Cache |
| `EXIT` | 退出进程 |

**stdout 协议：**

| 输出 | 含义 |
|---|---|
| `READY` | 模型加载完毕，开始接受请求 |
| `OK <id1> <id2> ...` | 生成完成，返回 token id |
| `ERR <message>` | 出错 |

统计信息（prefill/decode 耗时、tok/s）输出到 **stderr**，不污染 stdout 协议。

```bash
./qwen2_chat <model_dir> [max_new_tokens]
# 默认 max_new_tokens = 128
```

---

## 九、编译

### 依赖

| 依赖 | 说明 |
|---|---|
| aarch64 交叉编译工具链 | `aarch64-linux-gnu-g++` |
| RKNN SDK | `worker/lib/rknpu2/`（已内置） |
| CMake ≥ 3.10 | 构建系统 |

### 构建

```bash
cd worker
bash build-linux.sh
# 产物：install/qwen2_demo  install/qwen2_chat  install/librknnrt.so
```

### CMake 结构

```
worker_core（静态库）
├── OPS_SOURCES      # 9 个算子 .cpp
├── BACKEND_SOURCES  # npu_linear.cpp（rknn_matmul）
├── MODEL_SOURCES    # weight_loader / kv_cache / qwen2_model
└── API_SOURCES      # llm_engine.cpp

qwen2_demo → worker_core
qwen2_chat → worker_core
```

---

## 十、数据格式约定

| 位置 | 格式 | 说明 |
|---|---|---|
| safetensors 文件 | BF16（PyTorch 默认） | weight_loader 统一转为 FP16 |
| NPU 输入/输出（A/C） | FP16（uint16_t） | rknn_matmul 要求 |
| NPU 权重（B） | FP16 native layout | `rknn_B_normal_layout_to_native_layout` 预处理 |
| CPU 激活（hidden state） | FP32 | 精度优先 |
| KV Cache | FP16（uint16_t） | 节省内存；attention 内部临时转 FP32 计算 |
| logits 输出 | FP32 | lm_head 输出 FP16 后转换 |

---
