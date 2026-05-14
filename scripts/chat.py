#!/usr/bin/env python3
"""
qwen2_chat Python 包装器 —— 真正「可打字、可看见文字回复」的单轮对话入口

架构：
    用户输入 (str)
       ↓  AutoTokenizer.apply_chat_template + encode
    token ids (list[int])
       ↓  subprocess stdin
    qwen2_chat (C++，常驻，模型只加载一次)
       ↓  subprocess stdout
    token ids (list[int])
       ↓  tokenizer.decode
    文字回复 (str)

用法（本地，模型 + 二进制都在同机）：
    python3 chat.py \\
        --binary  /path/to/qwen2_chat \\
        --model   /path/to/Qwen1.5B \\
        --tokenizer /path/to/Qwen1.5B \\
        --max-new-tokens 128

说明：
    - 每条输入当作一次独立对话（C++ 端自动 reset KV Cache）
    - 后续如果要做多轮对话，只需在 Python 侧维护对话历史再 apply_chat_template
    - 本脚本不假设运行在板子上还是 WSL；它只关心 qwen2_chat 二进制的可执行路径
"""

import argparse
import os
import subprocess
import sys
import time


def build_argparser():
    p = argparse.ArgumentParser(description="Qwen2 NPU Chat (Python wrapper)")
    p.add_argument("--binary", required=True,
                   help="qwen2_chat 可执行文件路径")
    p.add_argument("--model", required=True,
                   help="模型目录（含 model.safetensors），传给 qwen2_chat")
    p.add_argument("--tokenizer", default=None,
                   help="tokenizer 目录，默认与 --model 相同")
    p.add_argument("--max-new-tokens", type=int, default=128,
                   help="每轮最多生成 token 数（默认 128）")
    p.add_argument("--lib-dir", default=None,
                   help="librknnrt.so 所在目录（板子上一般需要指定，会写入 LD_LIBRARY_PATH）")
    return p


def load_tokenizer(path):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(path, trust_remote_code=True)


def spawn_worker(binary, model_dir, max_new_tokens, lib_dir=None):
    env = os.environ.copy()
    if lib_dir:
        prev = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{lib_dir}:{prev}" if prev else lib_dir

    proc = subprocess.Popen(
        [binary, model_dir, str(max_new_tokens)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=sys.stderr,       # 让 C++ 的日志直接打到终端
        env=env,
        text=True,
        bufsize=1,               # 行缓冲
    )

    # 等待 "READY"
    while True:
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError("qwen2_chat 未能启动（stdout 提前结束）")
        line = line.strip()
        if line == "READY":
            break
        # 其他输出直接打印，便于看加载进度
        print(line, file=sys.stderr)
    return proc


def send_request(proc, token_ids):
    line = " ".join(map(str, token_ids))
    proc.stdin.write(line + "\n")
    proc.stdin.flush()

    resp = proc.stdout.readline().strip()
    if not resp:
        raise RuntimeError("qwen2_chat 返回空行（进程可能已退出）")
    if resp.startswith("ERR"):
        raise RuntimeError(f"qwen2_chat error: {resp}")
    if not resp.startswith("OK"):
        raise RuntimeError(f"unexpected response: {resp}")

    parts = resp.split()
    return list(map(int, parts[1:]))


def main():
    args = build_argparser().parse_args()
    tok_path = args.tokenizer or args.model

    print(f"[wrapper] 加载 tokenizer: {tok_path}", file=sys.stderr)
    tokenizer = load_tokenizer(tok_path)

    print(f"[wrapper] 启动 qwen2_chat: {args.binary}", file=sys.stderr)
    proc = spawn_worker(args.binary, args.model, args.max_new_tokens, args.lib_dir)
    print("[wrapper] 就绪。输入内容即可对话（exit 退出）", file=sys.stderr)

    try:
        while True:
            try:
                user = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not user or user.lower() == "exit":
                break

            messages = [{"role": "user", "content": user}]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            input_ids = tokenizer.encode(text)

            t0 = time.time()
            gen_ids = send_request(proc, input_ids)
            dt = time.time() - t0

            reply = tokenizer.decode(gen_ids, skip_special_tokens=True)
            print(f"Qwen: {reply}")
            print(f"      [生成 {len(gen_ids)} tokens, 总计 {dt:.1f}s,"
                  f" {len(gen_ids)/dt:.2f} tok/s]", file=sys.stderr)
    finally:
        try:
            proc.stdin.write("EXIT\n")
            proc.stdin.flush()
        except Exception:
            pass
        try:
            proc.wait(timeout=3)
        except Exception:
            proc.kill()
        print("[wrapper] 已退出", file=sys.stderr)


if __name__ == "__main__":
    main()
