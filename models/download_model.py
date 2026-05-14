#!/usr/bin/env python3
"""
下载 Qwen2-1.5B 模型到 models/qwen1.5b/Qwen1.5B 目录

用法:
    python models/download_qwen1.5b.py
    python models/download_qwen1.5b.py --model Qwen/Qwen2-1.5B
    python models/download_qwen1.5b.py --output /custom/path
"""

import argparse
import os
import sys

# ============================================================
# 常量 — 拉别的模型时改这里就行
# ============================================================
DEFAULT_MODEL_ID = "Qwen/Qwen2-1.5B"      # HuggingFace 模型 ID
DEFAULT_LOCAL_DIR = "qwen1.5b/Qwen1.5B"   # 相对于本脚本所在目录 (models/)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_output = os.path.join(script_dir, DEFAULT_LOCAL_DIR)

    parser = argparse.ArgumentParser(description=f"下载模型 {DEFAULT_MODEL_ID}")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_ID,
        help=f"HuggingFace 模型 ID (默认: {DEFAULT_MODEL_ID})",
    )
    parser.add_argument(
        "--output",
        default=default_output,
        help=f"输出目录 (默认: {default_output})",
    )
    args = parser.parse_args()

    print(f"模型: {args.model}")
    print(f"输出: {args.output}")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("缺少依赖，正在安装 huggingface_hub ...")
        os.system(f"{sys.executable} -m pip install huggingface_hub")
        from huggingface_hub import snapshot_download

    print("开始下载 ...")
    path = snapshot_download(
        repo_id=args.model,
        local_dir=args.output,
    )
    print(f"下载完成: {path}")


if __name__ == "__main__":
    main()
