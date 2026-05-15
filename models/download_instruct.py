#!/usr/bin/env python3
"""
下载 Qwen2-1.5B-Instruct 模型
保存路径：models/qwen1.5b-instruct/Qwen2-1.5B-Instruct/

用法：
    # 自动选源
    python3 models/download_instruct.py

    # 强制指定来源
    python3 models/download_instruct.py --source modelscope
    python3 models/download_instruct.py --source hf-mirror
    python3 models/download_instruct.py --source huggingface
"""

import os
import sys
import argparse

# ---- 配置 ----
HF_MODEL_ID = "Qwen/Qwen2-1.5B-Instruct"
MS_MODEL_ID = "qwen/Qwen2-1.5B-Instruct"  # ModelScope ID（阿里云，国内最快）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(SCRIPT_DIR, "qwen1.5b-instruct/Qwen2-1.5B-Instruct")


# ============================================================
# 方式一：ModelScope（推荐，国内直连，Qwen 官方同步，有进度条）
# ============================================================
def download_modelscope():
    try:
        from modelscope.hub.snapshot_download import snapshot_download as ms_dl
    except ImportError:
        print("  [自动安装] modelscope ...")
        os.system(f"{sys.executable} -m pip install modelscope -q")
        from modelscope.hub.snapshot_download import snapshot_download as ms_dl

    print(f"  模型：{MS_MODEL_ID}")
    print(f"  目标：{SAVE_DIR}")
    ms_dl(
        model_id=MS_MODEL_ID,
        local_dir=SAVE_DIR,
    )


# ============================================================
# 方式二：HuggingFace（支持镜像站，内置 tqdm 进度条）
# ============================================================
def download_hf(endpoint=None):
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("  [自动安装] huggingface_hub ...")
        os.system(f"{sys.executable} -m pip install huggingface_hub -q")
        from huggingface_hub import snapshot_download

    # 安装 tqdm 确保进度条可用
    try:
        import tqdm  # noqa
    except ImportError:
        os.system(f"{sys.executable} -m pip install tqdm -q")

    if endpoint:
        os.environ["HF_ENDPOINT"] = endpoint
        print(f"  镜像：{endpoint}")
    print(f"  模型：{HF_MODEL_ID}")
    print(f"  目标：{SAVE_DIR}")

    snapshot_download(
        repo_id=HF_MODEL_ID,
        local_dir=SAVE_DIR,
        ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
    )


# ============================================================
# 主流程
# ============================================================
def already_downloaded():
    if not os.path.isdir(SAVE_DIR):
        return False
    return any(
        f.endswith(".safetensors") or f.startswith("pytorch_model")
        for f in os.listdir(SAVE_DIR)
    )


def main():
    parser = argparse.ArgumentParser(description="下载 Qwen2-1.5B-Instruct")
    parser.add_argument(
        "--source",
        choices=["auto", "modelscope", "hf-mirror", "huggingface"],
        default="auto",
        help="下载来源（默认 auto：依次尝试 modelscope → hf-mirror → huggingface）",
    )
    args = parser.parse_args()

    if already_downloaded():
        print(f"[跳过] 模型已存在：{SAVE_DIR}")
        print("  若需重新下载，请删除该目录后再运行。")
        return

    os.makedirs(SAVE_DIR, exist_ok=True)

    sources = [
        ("huggingface", lambda: download_hf(None)),
        ("modelscope", lambda: download_modelscope()),
        ("hf-mirror", lambda: download_hf("https://hf-mirror.com")),
    ]

    # 单一来源模式
    if args.source != "auto":
        sources = [(s, fn) for s, fn in sources if s == args.source]

    last_err = None
    for name, fn in sources:
        print(f"\n[来源] {name}")
        try:
            fn()
            last_err = None
            break
        except KeyboardInterrupt:
            print("\n[中断] 用户取消")
            sys.exit(0)
        except Exception as e:
            last_err = e
            print(f"  [失败] {e}")
            if (name, fn) != sources[-1]:
                print("  → 切换到下一个来源...")

    if last_err is not None:
        print("\n[错误] 所有来源均失败，请检查网络或手动下载后放到：")
        print(f"  {SAVE_DIR}")
        sys.exit(1)

    print(f"\n[完成] 模型已保存到：{SAVE_DIR}")
    print("\n--- 验证（本机 PyTorch CPU）---")
    print('python3 -c "')
    print("import torch")
    print("from transformers import AutoTokenizer, AutoModelForCausalLM")
    print(f"t = AutoTokenizer.from_pretrained('{SAVE_DIR}')")
    print(
        f"m = AutoModelForCausalLM.from_pretrained('{SAVE_DIR}', torch_dtype=torch.bfloat16, device_map='cpu')"
    )
    print(
        "ids = t.encode(t.apply_chat_template([{'role':'user','content':'1+1等于几'}], tokenize=False, add_generation_prompt=True), return_tensors='pt')"
    )
    print(
        "print(t.decode(m.generate(ids, max_new_tokens=16)[0][ids.shape[1]:], skip_special_tokens=True))"
    )
    print('"')


if __name__ == "__main__":
    main()
