# RKLLM 环境变量说明

本文档记录当前 worker 中已经接入的环境变量。变量来源主要在：

- `worker/src/model/qwen2_model.cpp`
- `worker/src/ops/op_linear.cpp`
- `worker/src/ops/op_attention.cpp`
- `worker/src/backend/npu_linear.cpp`
- `worker/src/backend/npu_linear_w8.cpp`
- `worker/src/backend/npu_linear_i4.cpp`
- `worker/src/backend/npu_weight_cache.cpp`

## 常用组合

默认正确性测试：

```bash
./qwen2_chat Qwen1.5B 64 < cases.txt > out.txt 2> log.txt
```

prefill 速度测试常用组合：

```bash
RKLLM_LINEAR_BATCH=23 RKLLM_SHARD_DYNAMIC_M=1 RKLLM_NPU_ATTENTION=1 \
./qwen2_chat Qwen1.5B 64 < cases.txt > out.txt 2> log.txt
```

带 profile 的排查组合：

```bash
RKLLM_PROFILE=1 RKLLM_LINEAR_BATCH=23 RKLLM_SHARD_DYNAMIC_M=1 RKLLM_NPU_ATTENTION=1 \
./qwen2_chat Qwen1.5B 64 < cases.txt > out.txt 2> log.txt
```

注意：`RKLLM_BATCH_TRACE`、`RKLLM_NPU_ATTENTION_TRACE`、`RKLLM_NPU_ATTENTION_VERIFY` 会增加日志和额外 CPU 计算，不适合正式测速。

## 日志和 Profile

### `RKLLM_PROFILE`

- 默认：关闭。
- 作用：打印每次 `forward` 的阶段耗时。
- 位置：`qwen2_profile_enabled()`。
- 开启方式：任意非空且不是精确 `0` 的值，例如 `RKLLM_PROFILE=1`。
- 输出字段包括 `embedding`、`rmsnorm`、`qkv`、`attention`、`o_proj`、`gate_up`、`silu_mul`、`down`、`lm_head`。

### `RKLLM_BATCH_TRACE`

- 默认：关闭。
- 作用：打印 prefill linear 是否走 prepared、batch、fallback 等路径。
- 位置：`qwen2_batch_trace_enabled()`。
- 开启方式：任意非空且不是精确 `0` 的值。
- 用途：确认 `qkv_proj`、`o_proj`、`gate_up_proj`、`down_proj` 是否实际跑了 `M>1`。

### `RKLLM_MLP_PROFILE`

- 默认：关闭。
- 作用：打印 MLP 热点的细粒度耗时，专门分析 `gate_up`、`SwiGLU`、`down_proj` 是否还能继续优化。
- 开启方式：任意非空且不是 `0`、`false`、`FALSE`、`off`、`OFF` 的值。
- 输出内容：
  - `gate_up_input`：post-attention RMSNorm 写入 gate_up 输入 buffer 的耗时。
  - `sharded_prepared`：三核 sharded prepared 路径的共享 A buffer、输入复制和 NPU run 耗时。
  - `gate_up_linear`：模型层看到的 gate_up 总耗时。
  - `swiglu`：CPU `SiLU(gate) * up` 和 down input prepare 耗时。
  - `down_kshard`：`RKLLM_DOWN_KSPLIT=1` 时 down K-shard 的输入切片、三核 run、partial sum 和 residual add 耗时。
  - `down_total`：模型层看到的 down 总耗时。
- 注意：该变量会增加日志和少量计时开销，不适合正式测速。

### `RKLLM_MLP_PROFILE_LAYER`

- 默认：未设置，打印所有层。
- 作用：配合 `RKLLM_MLP_PROFILE=1`，只打印指定 Transformer 层的 MLP 细节。
- 示例：`RKLLM_MLP_PROFILE_LAYER=0` 只打印第 0 层。
- 用途：避免 28 层日志过多，先抽样判断瓶颈落点。

### `RKLLM_LOAD_PROFILE`

- 默认：关闭。
- 作用：打印模型加载总耗时，并开启 native cache 汇总日志。
- 位置：`npu_weight_cache::configure_for_model()` 和 `Qwen2Model::load()`。
- 开启值：`1`、`true`、`TRUE`、`on`、`ON`。

### `RKLLM_NATIVE_CACHE_PROFILE`

- 默认：关闭。
- 作用：只控制 native weight cache 的 profile 汇总。
- 开启值：`1`、`true`、`TRUE`、`on`、`ON`。

## Prefill Linear 批处理

### `RKLLM_LINEAR_BATCH`

- 默认：`1`。
- 作用：设置 prefill 阶段 linear 的目标 batch 行数，也就是 NPU matmul 的 `M`。
- 典型值：prompt 有 23 个 token 时用 `RKLLM_LINEAR_BATCH=23`。
- 影响路径：
  - `run_linear_batched_or_throw()` 使用它决定按多少行分块。
  - `NpuLinear`、`NpuLinearW8`、`NpuLinearI4` 用它创建 dynamic shape。
- 注意：
  - decode 阶段仍是 `M=1`。
  - `M>1` 需要后端支持 dynamic shape，否则会 fallback 到逐行。
  - 对三核分片后端，通常还需要配合 `RKLLM_SHARD_DYNAMIC_M=1`。
  - 改变该值会改变 native cache header flags，可能导致旧缓存 miss。

### `RKLLM_SHARD_DYNAMIC_M`

- 默认：关闭。
- 作用：允许带 core mask 的 NPU shard 使用 dynamic shape `M>1`。
- 位置：`shard_dynamic_m_enabled()`。
- 开启方式：任意非空且不是 `0`、`false`、`FALSE`、`off`、`OFF`。
- 注意：
  - 三核分片每个 shard 都绑定单独 NPU core。不开这个变量时，core-masked shard 通常只能稳定跑 `M=1`。
  - 当前 `NpuLinear` 中 dynamic M 还要求 `N <= 32768`，因此超大 `lm_head` shard 不一定支持 prefill batch。

## Linear 后端和三核分片

### `RKLLM_NPU_AUTO_SHARD`

- 默认：开启。
- 作用：控制自动三核分片 planner。
- 关闭值：`0`、`false`、`FALSE`、`off`、`OFF`。
- 关闭后，`LinearBackend::NPU` 自动规划会退回单核 NPU。

### `RKLLM_NPU_SHARD_MIN_OPS`

- 默认：`3000000`。
- 作用：自动分片的 `K*N` 最小阈值。
- 只有 `K*N >= RKLLM_NPU_SHARD_MIN_OPS` 且其他条件满足时，planner 才倾向三核分片。

### `RKLLM_NPU_SHARD_MIN_N`

- 默认：`16`。
- 作用：自动分片要求输出维度 `N` 至少达到该值。

### `RKLLM_NPU_SHARD_MIN_LARGE_DIM`

- 默认：`4096`。
- 作用：自动分片要求 `K` 或 `N` 至少有一个维度达到该阈值。
- 这是为了避免小矩阵因为分片调度开销反而变慢。

### `RKLLM_NPU_SHARD_SCOPE`

- 默认：`auto`。
- 作用：强制指定哪些 linear 进入三核分片，不再只依赖自动阈值。
- 可选值：
  - `auto`：按默认 planner。
  - `qkv`：只强制 fused qkv。
  - `o_proj`：只强制 attention output projection。
  - `attn`：强制 qkv 和 o_proj。
  - `mlp`：强制 gate_up 和 down。
  - `lm_head`：强制 lm_head。
  - `all`：所有可分片 linear。
  - `off` 或 `none`：不强制。
- 未识别值会打印警告并按 `auto` 处理。

### `RKLLM_HYBRID_CPU_SHARD`

- 默认：关闭。
- 作用：实验性地把 MLP 的三核 NPU 分片改成 `3 个 NPU shard + 1 个 CPU shard`。
- 可选值：
  - `gate_up`：只作用于 fused gate_up。
  - `down` 或 `down_proj`：只作用于 down_proj。
  - `mlp`：同时作用于 gate_up 和 down。
  - `all`：当前等价于 `mlp`，不会作用于 lm_head。
  - `off` 或 `none`：关闭。
- 注意：
  - 这是速度上限实验路径，默认不开启。
  - CPU shard 使用专门快路径：加载时把该 shard 权重预转为 FP32，运行时在输出维用 NEON 累加。
  - 开启后命中的层会跳过 sharded native cache warm-load，冷加载重新构造 CPU shard 权重。
  - CPU shard 太大会拖慢整层，因为最后仍要等待 CPU 和 3 个 NPU shard 都完成。
  - 当前不支持 lm_head hybrid，避免大词表 argmax 路径复杂化。

### `RKLLM_HYBRID_CPU_RATIO`

- 默认：`5`。
- 范围：`1` 到 `25`。
- 作用：设置 CPU shard 占逻辑输出维度的比例，单位是百分比。
- 示例：`RKLLM_HYBRID_CPU_SHARD=gate_up RKLLM_HYBRID_CPU_RATIO=5` 表示 gate_up 的约 5% intermediate slice 由 CPU 计算。
- 建议：从 `5` 开始测。如果 `gate_up` 或 `down` profile 没下降，说明 CPU tail 拖慢，应关闭或降低比例。

### `RKLLM_DOWN_KSPLIT`

- 默认：关闭。
- 作用：实验性地把 `down_proj` 改成按 K 维三核分片。
- 开启值：任意非空且不是 `0`、`false`、`FALSE`、`off`、`OFF`、`none`、`NONE`。
- 触发条件：
  - 只作用于 `role=down`。
  - `K > 8192`。
  - 只作用于 FP16 sharded 路径，不和 A8W8/A4W4/hybrid CPU shard 混用。
- 用途：绕开普通 N 维分片后每个 shard 变成 `K=8960,N=512,M=23` 的 RKNN 大 K 慢路径。
- 计算方式：
  - 三个 NPU core 分别计算 `[M,K_i] * [K_i,N] -> [M,N]`。
  - 每个 partial 输出使用 `RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32`。
  - CPU 将三个 `[M,N]` partial sum 累加到 residual FP32。
- 注意：
  - 这是 down 专用实验路径，默认不开启。
  - 会改变 native cache key；首次 cold load 会重新写 K-shard cache。

### `RKLLM_DOWN_KSPLIT_K`

- 默认：`4480`。
- 范围：`16` 到 `8192`。
- 作用：旧的 per-N-shard K-split 实验参数。
- 当前 `RKLLM_DOWN_KSPLIT=1` 优先使用 K 维三核分片，因此该值暂时不影响主实验路径。

### `RKLLM_NPU_AUTOTUNE`

- 默认：关闭。
- 作用：加载时对部分 linear 同时 benchmark 单核和三核，再选择更快的后端。
- 开启值：`1`、`true`、`TRUE`、`on`、`ON`。
- 限制：
  - 只对 `3,000,000 <= K*N <= 50,000,000` 的矩阵生效。
  - 会增加冷加载时间。
  - native cache warm-load 时没有原始权重，无法完整 autotune，只能按 planner 和 cache 命中情况选择。

### `RKLLM_LM_HEAD_BACKEND`

- 默认：`NPU_SHARDED`。
- 作用：单独控制 `lm_head` 后端。
- 可选值：
  - `CPU`
  - `NPU`
  - `NPU_SINGLE` 或 `SINGLE_NPU`
  - `NPU_SHARDED` 或 `SHARDED_NPU`
- 用途：
  - `NPU_SHARDED` 是默认高性能路径。
  - `CPU` 可用于排查 `lm_head` 大权重引起的 NPU/CMA 分配失败。
  - `NPU_SINGLE` 可用于排查三核分片问题。

## 量化后端

### `RKLLM_NPU_WEIGHT_DTYPE`

- 默认：FP16。
- 作用：选择 NPU linear 的实验量化后端。
- A8W8 可选值：`int8`、`INT8`、`a8w8`、`A8W8`、`w8`、`W8`。
- A4W4 可选值：`int4`、`INT4`、`a4w4`、`A4W4`、`w4`、`W4`。
- 不支持值：`w8a16`、`W8A16`。RK3588 当前 matmul runtime 不支持 FP16 x INT8，所以代码会打印警告并使用 FP16。
- 注意：
  - A8W8 使用 `RKNN_INT8_MM_INT8_TO_INT32`，输出再按 scale 反量化。
  - 默认 A4W4 会尝试 `RKNN_INT8_MM_INT4_TO_INT32`；当前 RK3588 runtime 若返回 unsupported，会自动 fallback FP16。
  - 可用 `RKLLM_INT4_KSPLIT=1` 走真实 `RKNN_INT4_MM_INT4_TO_INT16` 分块实验路径，见下文。
  - 这是速度上限实验路径，A8W8/A4W4 开启时可能因为量化误差累积导致生成 token 与 FP16 基准不一致。

### `RKLLM_NPU_INT8_SCOPE`

- 默认：`gate_up`。
- 作用：启用 A8W8 时，限制哪些 linear 使用 int8。
- 可选值：
  - `gate_up`
  - `mlp`：gate_up 和 down。
  - `attn`：qkv 和 o_proj。
  - `lm_head`
  - `all`
  - `off` 或 `none`
- 未识别值会打印警告并回到 `gate_up`。

### `RKLLM_NPU_INT4_SCOPE`

- 默认：若未设置，则复用 `RKLLM_NPU_INT8_SCOPE`；两者都未设置时为 `gate_up`。
- 作用：启用 A4W4 时，限制哪些 linear 使用 int4。
- 可选值同 `RKLLM_NPU_INT8_SCOPE`。
- 注意：A4W4 的对齐约束更强，部分 shard 可能因为 `N` 不满足 64 对齐而 fallback FP16。
- 默认路径下，当前 RK3588 runtime 不支持 `RKNN_INT8_MM_INT4_TO_INT32` 时，int4 请求会整体 fallback FP16；日志会出现 `I4 shard ... fallback FP16`。

### `RKLLM_INT4_KSPLIT`

- 默认：关闭。
- 作用：启用真实 A4W4 K 维分块实验路径。
- 开启值：`1`、`true`、`TRUE`、`on`、`ON`。
- 实现：
  - 每个 K chunk 使用 `RKNN_INT4_MM_INT4_TO_INT16`。
  - 每个 chunk 的 int16 输出按输入 scale 和权重 scale 反量化后累加到 FP32。
  - 权重传给 `rknn_B_normal_layout_to_native_layout` 时使用 RKNN 期望的一字节一个 int4 元素 normal layout，再由 RKNN 转成 native layout。
- 注意：
  - 该路径是实验性能路径，不是默认正确性路径。
  - 当前只在 `gate_up` 范围做过小样本验证：真实 INT4 无 fallback，16 条样例 exact `2/16`、首 token `10/16`，平均 decode 约 `6.21 tok/s`；同批 FP16 约 `5.22 tok/s`。
  - 精度仍明显低于 FP16/A8W8，适合继续调量化策略，不适合作为默认生产路径。

### `RKLLM_INT4_KSPLIT_K`

- 默认：`512`。
- 作用：设置 `RKLLM_INT4_KSPLIT=1` 时每个 K chunk 的大小。
- 约束：
  - 会按 32 对齐。
  - 最大限制为 `640`，避免 `640 * 7 * 7` 超过 int16 安全累加范围。
  - 无效值会打印警告并回到默认值。

### `RKLLM_NPU_INT8_LAYERS`

- 默认：全部 transformer layer 允许量化。
- 作用：限制 int8/int4 量化只作用于指定层。
- 语法：
  - `all`：全部层。
  - `none` 或 `off`：不量化 transformer 层。
  - `0,1,2`：只量化指定层。
  - `0-7,16-27`：支持范围。
- 注意：
  - 这个变量也被 int4 路径复用。
  - `lm_head` 没有 transformer layer index，因此不受层号限制。

### `RKLLM_A8W8_NEON_QUANT`

- 默认：关闭。
- 作用：A8W8 输入激活量化时使用 NEON 优化路径。
- 开启值：`1`、`true`、`TRUE`、`on`、`ON`。

### `RKLLM_A8W8_HADAMARD`

- 默认：关闭。
- 作用：A8W8 路径对权重和输入使用 Hadamard 变换后再量化。
- 开启值：`1`、`true`、`TRUE`、`on`、`ON`。
- 注意：
  - 会改变实际 matmul 的 `K_matmul`、scale 和 native cache flags。
  - 属于实验精度/速度路径。

### `RKLLM_A8W8_HADAMARD_BLOCK`

- 默认：未设置时使用 full Hadamard block。
- 作用：设置 A8W8 Hadamard block 大小。
- 要求：
  - 正整数。
  - 2 的幂。
  - 不大于 `K`。
  - 能整除 `K`。
- 无效值会打印警告，并回到 full block。

## Attention

### `RKLLM_ATTENTION_THREADS`

- 默认：`min(std::thread::hardware_concurrency(), 6)`，如果硬件线程数未知则按 4。
- 作用：decode CPU attention 的常驻线程池线程数。
- 范围：`1` 到 `16`。

### `RKLLM_ATTENTION_PARALLEL_MIN_LEN`

- 默认：`64`。
- 作用：decode 阶段只有 `total_len >= 该值` 时才使用 CPU head 并行线程池。
- 范围：`1` 到 `4096`。

### `RKLLM_NPU_ATTENTION`

- 默认：关闭。
- 作用：开启 prefill attention 的 NPU 快路径。
- 开启方式：任意非空且不是 `0`、`false`、`FALSE`、`off`、`OFF`。
- 当前默认模式：
  - 不设置 `RKLLM_NPU_ATTENTION_MODE` 时走 `pv-only`。
  - CPU 仍计算 FP32 QK 和 softmax。
  - NPU 只计算 `P @ V`。
  - pv-only 会按 GQA group 批头，Qwen2 1.5B 的 `seq=23` 时 trace 应显示 `rows=138`。
- decode 阶段 `seq=1` 不走该 prefill NPU attention 路径。

### `RKLLM_NPU_ATTENTION_MODE`

- 默认：`pv-only`。
- 可选 full 值：`full`、`FULL`、`qk_pv`、`QK_PV`。
- full 模式含义：QK 和 PV 都走 NPU。
- 注意：
  - full 模式会把 Q 从 FP32 转 FP16 后做 QK，softmax 对小误差敏感，实测可能导致重复输出或提前停止。
  - 速度上限实验可以用 full；正确性测试不要默认使用 full。

### `RKLLM_NPU_ATTENTION_GROUP_HEADS`

- 默认：关闭。
- 作用：full 模式下，将同一个 KV head 对应的多个 query head 合成一个更大的 `M=seq*group` 运行。
- 注意：pv-only 当前默认已经批头，不依赖该变量。

### `RKLLM_NPU_ATTENTION_DYNAMIC`

- 默认：关闭。
- 作用：NPU attention 的 runtime matmul 使用 dynamic shape 创建。
- 用途：实验不同 `M` 的 attention matmul。
- 注意：这是 attention 专用开关，不等同于 linear 的 `RKLLM_SHARD_DYNAMIC_M`。

### `RKLLM_NPU_ATTENTION_TRACE`

- 默认：关闭。
- 作用：打印 NPU attention 的模式和 shape。
- 典型输出：

```text
[op_attention] NPU prefill attention mode=pv seq=23 total_len=23 aligned=32 heads=12 kv_heads=2 group=6 rows=138 head_dim=128
```

### `RKLLM_NPU_ATTENTION_VERIFY`

- 默认：关闭。
- 作用：对 NPU attention 输出做 CPU reference 校验。
- 行为：
  - 打印 `verify_pv` 或 `verify_qk` 的 matmul 误差。
  - 打印最终 attention 输出误差。
  - 若最大绝对误差超过阈值，会用 CPU reference 覆盖输出，并禁用后续 NPU attention。
- 注意：会增加大量 CPU 计算，不适合正式测速。

### `RKLLM_NPU_ATTENTION_VERIFY_TOL`

- 默认：`0.05`。
- 作用：设置 NPU attention verify 的最大绝对误差阈值。
- 无效或小于等于 0 的值会回到 `0.05`。

## Native Weight Cache

### `RKLLM_NATIVE_CACHE`

- 默认：开启。
- 作用：控制 RKNN native-layout 权重缓存。
- 关闭值：`0`、`false`、`FALSE`、`off`、`OFF`。
- 行为：
  - 冷加载时将 native B 写入缓存。
  - warm-load 时跳过 safetensors 大权重读取、转置、融合、shard copy 和 native layout 转换。

### `RKLLM_NATIVE_CACHE_DIR`

- 默认：`<model_dir>/.rknn_native_cache`。
- 作用：覆盖 native cache 根目录。
- 用途：板端多个测试目录共用同一份 cache，或者临时切换 cache 目录排查加载失败。

缓存 key 会包含模型文件大小和 mtime，以及后端 kind、K/N、dynamic M flags、A8W8/A4W4 flags 等。改变量化、batch 或 dynamic shape 相关配置后，旧 cache 可能 miss，这是正常行为。

## 推荐排查方式

确认 prefill linear 是否真的批处理：

```bash
RKLLM_LINEAR_BATCH=23 RKLLM_SHARD_DYNAMIC_M=1 RKLLM_BATCH_TRACE=1 \
./qwen2_chat Qwen1.5B 64 < cases.txt > out.txt 2> log.txt
```

期望看到：

```text
[batch] qkv_proj rows=23 runM=23 path=prepared
[batch] o_proj rows=23 runM=23 path=f32_accumulate
[batch] gate_up_proj rows=23 runM=23 path=prepared
[batch] down_proj rows=23 runM=23 path=prepared
```

确认 NPU attention pv-only 是否批头：

```bash
RKLLM_NPU_ATTENTION=1 RKLLM_NPU_ATTENTION_TRACE=1 \
./qwen2_chat Qwen1.5B 64 < cases.txt > out.txt 2> log.txt
```

期望 Qwen2 1.5B 的 `seq=23` trace 是 `rows=138`，不是 `rows=23`。

排查 attention 数值：

```bash
RKLLM_NPU_ATTENTION=1 RKLLM_NPU_ATTENTION_VERIFY=1 RKLLM_NPU_ATTENTION_TRACE=1 \
./qwen2_chat Qwen1.5B 64 < cases.txt > out.txt 2> log.txt
```

正式测速时去掉 `VERIFY` 和 `TRACE`。
