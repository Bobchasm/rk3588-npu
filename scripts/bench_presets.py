#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PRESETS = ("accurate", "balanced", "fast")
PRESET_LABELS = {
    "accurate": "FP16 accurate",
    "balanced": "A8 balanced",
    "fast": "A8 fast",
}


def run(cmd, *, cwd=ROOT, check=True):
    return subprocess.run(cmd, cwd=cwd, check=check, text=True)


def parse_log(log_path, out_path):
    text = Path(log_path).read_text(errors="replace")
    rows = []
    pat = re.compile(
        r"prefill=(\d+) tok ([0-9.]+) ms, decode=(\d+) tok "
        r"([0-9.]+) ms \(([0-9.]+) tok/s\)\s+stop=(\S+)"
    )
    for line in text.splitlines():
        m = pat.search(line)
        if not m:
            continue
        rows.append(
            {
                "prefill_tokens": int(m.group(1)),
                "prefill_ms": float(m.group(2)),
                "decode_tokens": int(m.group(3)),
                "decode_ms": float(m.group(4)),
                "decode_tps_line": float(m.group(5)),
                "stop": m.group(6),
            }
        )

    oks = [
        line
        for line in Path(out_path).read_text(errors="replace").splitlines()
        if line.startswith("OK")
    ]
    if not rows:
        return {
            "rows": 0,
            "avg_prefill_ms": 0.0,
            "decode_tokens": 0,
            "decode_ms": 0.0,
            "decode_tps": 0.0,
            "fallback_fp16": text.count("fallback FP16"),
            "stops": {},
            "outputs": oks,
        }

    prefill_total = sum(r["prefill_ms"] for r in rows)
    decode_tokens = sum(r["decode_tokens"] for r in rows)
    decode_ms = sum(r["decode_ms"] for r in rows)
    stops = {}
    for r in rows:
        stops[r["stop"]] = stops.get(r["stop"], 0) + 1

    return {
        "rows": len(rows),
        "avg_prefill_ms": prefill_total / len(rows),
        "prefill_total_ms": prefill_total,
        "decode_tokens": decode_tokens,
        "decode_ms": decode_ms,
        "decode_tps": decode_tokens / (decode_ms / 1000.0) if decode_ms else 0.0,
        "fallback_fp16": text.count("fallback FP16"),
        "stops": stops,
        "outputs": oks,
    }


def compare_outputs(baseline, item):
    base = baseline.get("outputs", [])
    cur = item.get("outputs", [])
    n = min(len(base), len(cur))
    exact = 0
    first = 0
    for i in range(n):
        if base[i] == cur[i]:
            exact += 1
        b = base[i].split()
        c = cur[i].split()
        if len(b) > 1 and len(c) > 1 and b[1] == c[1]:
            first += 1
    item["compare_rows"] = n
    item["exact_match"] = exact
    item["first_token_match"] = first


def preset_paths(log_dir, preset):
    return (
        Path(log_dir) / f"log_{preset}.txt",
        Path(log_dir) / f"out_{preset}.txt",
    )


def run_remote(args):
    remote_dir = args.remote_dir.rstrip("/")
    local_dir = Path(args.log_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    for preset in PRESETS:
        remote_log = f"log_{preset}.txt"
        remote_out = f"out_{preset}.txt"
        command = (
            f"cd {remote_dir} && "
            "ulimit -n 4096 && "
            "chmod +x qwen2_chat && "
            r"export LD_LIBRARY_PATH=/root/matmul/worker_test:\$LD_LIBRARY_PATH && "
            f"RKLLM_PRESET={preset} ./qwen2_chat {args.model_dir} "
            f"{args.max_new_tokens} < {args.cases} > {remote_out} 2> {remote_log}"
        )
        run(
            [
                "timeout",
                str(args.timeout),
                "sshpass",
                "-p",
                args.password,
                "ssh",
                "-o",
                "StrictHostKeyChecking=no",
                args.host,
                f'chroot {args.chroot} /bin/bash -lc "{command}"',
            ]
        )
        log_path, out_path = preset_paths(local_dir, preset)
        run(
            [
                "timeout",
                "60",
                "sshpass",
                "-p",
                args.password,
                "scp",
                "-o",
                "StrictHostKeyChecking=no",
                f"{args.host}:{args.chroot}{remote_dir}/{remote_log}",
                str(log_path),
            ]
        )
        run(
            [
                "timeout",
                "60",
                "sshpass",
                "-p",
                args.password,
                "scp",
                "-o",
                "StrictHostKeyChecking=no",
                f"{args.host}:{args.chroot}{remote_dir}/{remote_out}",
                str(out_path),
            ]
        )


def load_results(log_dir):
    results = {}
    for preset in PRESETS:
        log_path, out_path = preset_paths(log_dir, preset)
        if not log_path.exists() or not out_path.exists():
            continue
        results[preset] = parse_log(log_path, out_path)
    if "accurate" in results:
        baseline = results["accurate"]
        for item in results.values():
            compare_outputs(baseline, item)
    return results


def print_markdown(results):
    print("| preset | avg prefill ms | decode tok/s | exact | first | stops | fallback |")
    print("|---|---:|---:|---:|---:|---|---:|")
    for preset in PRESETS:
        if preset not in results:
            continue
        item = results[preset]
        n = item.get("compare_rows", 0)
        exact = f"{item.get('exact_match', 0)}/{n}" if n else "-"
        first = f"{item.get('first_token_match', 0)}/{n}" if n else "-"
        stops = ",".join(f"{k}:{v}" for k, v in sorted(item.get("stops", {}).items()))
        print(
            f"| {PRESET_LABELS[preset]} | "
            f"{item['avg_prefill_ms']:.1f} | "
            f"{item['decode_tps']:.3f} | "
            f"{exact} | {first} | {stops} | {item['fallback_fp16']} |"
        )


def main():
    ap = argparse.ArgumentParser(description="Run or parse qwen2_chat preset benchmarks.")
    ap.add_argument("--log-dir", default="/tmp/rk3588_preset_bench")
    ap.add_argument("--summary-json", default="")
    ap.add_argument("--run-remote", action="store_true")
    ap.add_argument("--host", default="rk3588")
    ap.add_argument("--password", default="root")
    ap.add_argument("--chroot", default="/root/22_04_rootfs")
    ap.add_argument("--remote-dir", default="/root/matmul/worker_test")
    ap.add_argument("--model-dir", default="Qwen1.5B")
    ap.add_argument("--cases", default="cases.txt")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--timeout", type=int, default=2400)
    args = ap.parse_args()

    if args.run_remote:
        run_remote(args)

    results = load_results(args.log_dir)
    print_markdown(results)

    if args.summary_json:
        Path(args.summary_json).write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
