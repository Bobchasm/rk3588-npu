"""
模型层布局分析
- 不加载权重到内存
- 标注每个算子运行在 NPU 还是 CPU
- 检查 RKNN MatMul API 的 32 对齐限制
- 输出用于模型拆分的完整信息
"""
import os, json, sys, io
from safetensors import safe_open

MODEL_PATH = "./qwen1.5b/Qwen1.5B"
OUTPUT_FILE = "./qwen1.5b/qwen1.5b_layout.txt"

class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
    def flush(self):
        for s in self.streams:
            s.flush()

_file = open(OUTPUT_FILE, "w", encoding="utf-8")
sys.stdout = Tee(sys.__stdout__, _file)

# ────────────────────────────────────────────
# 1. 读取配置
# ────────────────────────────────────────────
with open(os.path.join(MODEL_PATH, "config.json")) as f:
    cfg = json.load(f)

H   = cfg["hidden_size"]          # 1536
I   = cfg["intermediate_size"]    # 8960
NH  = cfg["num_attention_heads"]  # 12
NKV = cfg["num_key_value_heads"]  # 2
HD  = H // NH                     # 128  (head_dim)
KVD = NKV * HD                    # 256  (total K/V proj dim)
NL  = cfg["num_hidden_layers"]    # 28
VS  = cfg["vocab_size"]           # 151936
ACT = cfg["hidden_act"]           # silu

# ────────────────────────────────────────────
# 2. 读取所有权重 key（只读索引，不读数据）
# ────────────────────────────────────────────
sf_path = os.path.join(MODEL_PATH, "model.safetensors")
with safe_open(sf_path, framework="pt", device="cpu") as f:
    all_keys = set(f.keys())

def has_bias(prefix):
    return (prefix + ".bias") in all_keys

# ────────────────────────────────────────────
# 3. 工具函数
# ────────────────────────────────────────────
NPU = "🟢 NPU"
CPU = "🔵 CPU"
WARN = "⚠️ "

def align_check(k, name=""):
    ok = k % 32 == 0
    tag = "" if ok else f"  {WARN}K={k} 不是32的倍数！注意对齐"
    return ok, tag

def matmul_row(name, M, K, N, note=""):
    ok, tag = align_check(K)
    bias_note = f" + bias(CPU)" if note == "bias" else ""
    return f"    {NPU}  MatMul  {name:20s}  [{M}×{K}] @ [{K}×{N}]{bias_note}{tag}"

def cpu_row(name, desc):
    return f"    {CPU}  {name:28s}  {desc}"

# ────────────────────────────────────────────
# 4. 打印完整图
# ────────────────────────────────────────────
print("=" * 75)
print("  Qwen1.5B (Qwen2ForCausalLM) 完整层布局  NPU/CPU 分配")
print("=" * 75)
print(f"  配置: {NL}层 | hidden={H} | heads={NH} | KV_heads={NKV}(GQA)")
print(f"         head_dim={HD} | intermediate={I} | vocab={VS} | act={ACT}")
print("=" * 75)

print()
print("┌─ 输入")
print(cpu_row("Tokenizer", f"文本 → token ids  (vocab={VS})"))
print(cpu_row("Embedding (embed_tokens)", f"[seq, {VS}] → [seq, {H}]  (CPU, 大表查找)"))

for layer_idx in range(NL):
    print()
    print(f"├─ TransformerBlock [{layer_idx:02d}/{NL-1}]")

    # --- input_layernorm ---
    print(cpu_row(f"  input_layernorm (RMSNorm)", f"[seq, {H}]  eps=1e-6"))

    # --- Attention ---
    print(f"  │  [Self-Attention  GQA  NH={NH} NKV={NKV}]")

    # q/k/v proj — q和k有bias，v有bias
    q_bias = has_bias(f"model.layers.{layer_idx}.self_attn.q_proj")
    k_bias = has_bias(f"model.layers.{layer_idx}.self_attn.k_proj")
    v_bias = has_bias(f"model.layers.{layer_idx}.self_attn.v_proj")
    o_bias = has_bias(f"model.layers.{layer_idx}.self_attn.o_proj")

    print(matmul_row("q_proj", "M", H, H,   "bias" if q_bias else ""))
    if q_bias:
        print(cpu_row("    bias_add (q)", f"[M, {H}] + [{H}]"))
    print(matmul_row("k_proj", "M", H, KVD, "bias" if k_bias else ""))
    if k_bias:
        print(cpu_row("    bias_add (k)", f"[M, {KVD}] + [{KVD}]"))
    print(matmul_row("v_proj", "M", H, KVD, "bias" if v_bias else ""))
    if v_bias:
        print(cpu_row("    bias_add (v)", f"[M, {KVD}] + [{KVD}]"))

    print(cpu_row("    RoPE (rotary_emb)", f"Q[M,{NH},{HD}], K[M,{NKV},{HD}]  → 旋转位置编码"))
    print(cpu_row("    GQA expand K/V", f"KV heads {NKV} → {NH}  (repeat_kv)"))
    print(cpu_row("    KV Cache append", f"K,V concat历史  [past+M, {NKV}, {HD}]"))

    # attention score: QK^T  K=HD=128 ✅
    ok1, t1 = align_check(HD, "HD")
    print(f"    {NPU}  MatMul  QK^T (per head)        [M×{HD}] @ [{HD}×seqlen]  →score{t1}")
    print(cpu_row("    scale + Softmax", f"score / sqrt({HD})  → attn_weights [M, seqlen]"))

    # score @ V  K=seqlen ⚠️ 需要seqlen是32的倍数
    print(f"    {NPU}  MatMul  score@V (per head)     [M×seqlen] @ [seqlen×{HD}]")
    print(f"          {WARN}score@V 的 K=seqlen，seqlen必须是32的倍数！")
    print(f"             prefill时需要pad序列长度；decode时KV cache长度需对齐")

    print(matmul_row("o_proj", "M", H, H, "bias" if o_bias else ""))

    # --- post_attention_layernorm ---
    print(cpu_row("  post_attn_layernorm (RMSNorm)", f"[M, {H}]  + 残差加法"))

    # --- MLP ---
    print(f"  │  [MLP  SwiGLU]")
    g_bias = has_bias(f"model.layers.{layer_idx}.mlp.gate_proj")
    u_bias = has_bias(f"model.layers.{layer_idx}.mlp.up_proj")
    d_bias = has_bias(f"model.layers.{layer_idx}.mlp.down_proj")
    print(matmul_row("gate_proj", "M", H, I, "bias" if g_bias else ""))
    print(matmul_row("up_proj",   "M", H, I, "bias" if u_bias else ""))
    print(cpu_row("    SiLU(gate) * up", f"element-wise  [M, {I}]"))
    print(matmul_row("down_proj", "M", I, H, "bias" if d_bias else ""))
    print(cpu_row("  residual add", f"[M, {H}]  + 残差"))

    if layer_idx < NL - 1:
        print(f"  │")
        print(f"  └─ → 残差输出传入下一层 hidden_state [M, {H}]")

print()
print("└─ 输出")
print(cpu_row("  final_norm (RMSNorm)", f"[M, {H}]"))
ok_lm, t_lm = align_check(H, "H")
print(f"    {NPU}  MatMul  lm_head (tied weights)  [M×{H}] @ [{H}×{VS}]{t_lm}")
print(cpu_row("  Softmax / Sampling", f"[M, {VS}] → next token"))

# ────────────────────────────────────────────
# 5. 汇总统计
# ────────────────────────────────────────────
print()
print("=" * 75)
print("  汇总统计")
print("=" * 75)

# 每层 matmul 数量
matmuls_per_layer = 7  # q/k/v/o/gate/up/down
# attention QK^T 和 score@V 每头一次，共 NH 次，但底层调用可以是 batched
attn_matmuls_per_layer = 2  # QK^T + score@V（逻辑上，可能batch处理）
total_proj_matmuls = matmuls_per_layer * NL
total_matmuls = total_proj_matmuls + NL * attn_matmuls_per_layer + 1  # +lm_head

print(f"  投影层 MatMul:  {matmuls_per_layer} × {NL}层 = {total_proj_matmuls} 次/token")
print(f"  Attention MatMul: (QK^T + score@V) × {NL}层 = {NL*attn_matmuls_per_layer} 次/token")
print(f"  lm_head MatMul: 1 次/token")
print(f"  合计 MatMul: {total_matmuls} 次/token（约占总计算量90%+）")
print()
print("  CPU 算子清单:")
print(f"    - RMSNorm × {NL*2+1} 次  (每层2次 + 最终1次)")
print(f"    - RoPE × {NL} 次")
print(f"    - GQA repeat_kv × {NL} 次")
print(f"    - KV Cache append/read × {NL} 次")
print(f"    - Softmax × {NL} 次")
print(f"    - SiLU × {NL} 次 (element-wise)")
print(f"    - Residual add × {NL*2} 次")
print(f"    - bias add × {NL*3} 次 (q/k/v_proj 有bias)")
print(f"    - Embedding lookup × 1 次")
print(f"    - Sampling × 1 次")
print()
print("  K维度32对齐检查:")
dims = [
    ("q/k/v/o proj K (hidden_size)", H),
    ("gate/up proj K (hidden_size)", H),
    ("down_proj K (intermediate_size)", I),
    ("QK^T K (head_dim)", HD),
    ("lm_head K (hidden_size)", H),
    ("score@V K (seqlen)", None),   # 动态
]
for name, val in dims:
    if val is None:
        print(f"    ⚠️  {name}: 动态！必须保证seqlen是32的倍数（padding）")
    else:
        ok = val % 32 == 0
        mark = "✅" if ok else "❌"
        print(f"    {mark} {name} = {val}  ({val}÷32={val//32})")

print()
print("  模型拆分建议 (按层均分到多个worker):")
layers_per_worker = NL // 4
print(f"    4节点方案: Embedding+L0~L{layers_per_worker-1} | L{layers_per_worker}~L{2*layers_per_worker-1} | "
      f"L{2*layers_per_worker}~L{3*layers_per_worker-1} | L{3*layers_per_worker}~L{NL-1}+lm_head")
print("=" * 75)

# 关闭文件并恢复 stdout
sys.stdout = sys.__stdout__
_file.close()
print(f"\n已保存到文件: model/{OUTPUT_FILE}")
