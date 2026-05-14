#!/usr/bin/env python3
"""
下载 Qwen2-1.5B-Instruct 模型
保存路径：models/Qwen2-1.5B-Instruct/（与 qwen1.5b/ 同级）

用法：
    python3 models/download_instruct.py

需要网络能访问 HuggingFace（或配置镜像）。
若访问不了，可在环境变量里设置镜像：
    export HF_ENDPOINT=https://hf-mirror.com
    python3 models/download_instruct.py
"""

import os
import sys

# ---- 配置 ----
MODEL_ID   = "Qwen/Qwen2-1.5B-Instruct"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))          # models/
SAVE_DIR   = os.path.join(SCRIPT_DIR, "Qwen2-1.5B-Instruct")    # models/Qwen2-1.5B-Instruct/

def check_deps():
    try:
        import huggingface_hub
    except ImportError:
        print("[错误] 缺少 huggingface_hub，请先安装：")
        print("    pip install huggingface_hub")
        sys.exit(1)

def download():
    from huggingface_hub import snapshot_download

    if os.path.isdir(SAVE_DIR) and os.listdir(SAVE_DIR):
        print(f"[跳过] 目录已存在且非空：{SAVE_DIR}")
        print("  若需重新下载，请手动删除该目录后再运行。")
        return

    print(f"[开始] 下载 {MODEL_ID}")
    print(f"[目标] {SAVE_DIR}")
    print()
    print("  提示：若速度慢，可设置镜像：")
    print("    export HF_ENDPOINT=https://hf-mirror.com")
    print()

    os.makedirs(SAVE_DIR, exist_ok=True)

    snapshot_download(
        repo_id   = MODEL_ID,
        local_dir = SAVE_DIR,
        # 只下载推理必需的文件，跳过 git lfs 指针、pytorch_model.bin 分片等大型冗余文件
        ignore_patterns = ["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
    )

    print()
    print(f"[完成] 模型已保存到：{SAVE_DIR}")
    print()
    print("下一步——在板子上测试（NPU 推理）：")
    print(f"  ./qwen2_demo {SAVE_DIR} <token_id1> [token_id2 ...]")
    print()
    print("或用 PyTorch 在本机验证（CPU）：")
    print("""  python3 -c "
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
tkz   = AutoTokenizer.from_pretrained('models/Qwen2-1.5B-Instruct')
model = AutoModelForCausalLM.from_pretrained('models/Qwen2-1.5B-Instruct',
            torch_dtype=torch.bfloat16, device_map='cpu')
msgs  = [{'role':'user','content':'1+1等于几'}]
text  = tkz.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
ids   = tkz.encode(text, return_tensors='pt')
out   = model.generate(ids, max_new_tokens=32, do_sample=False,
            pad_token_id=tkz.eos_token_id)
print(tkz.decode(out[0][ids.shape[1]:], skip_special_tokens=True))
  " """)

if __name__ == "__main__":
    check_deps()
    download()
