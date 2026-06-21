#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
import sys
import warnings

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "bindings", "python"))

import ray  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--actor-name", required=True)
    parser.add_argument("--namespace", default="rk3588-npu")
    parser.add_argument("--ray-address", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=10)
    parser.add_argument("--repetition-window", type=int, default=6)
    parser.add_argument("--pipeline-mode", default=None, choices=["centralized", "p2p"])
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--request-id", default=None)
    parser.add_argument("--no-reset-kv", action="store_true")
    parser.add_argument("token_ids", nargs="+", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    warnings.filterwarnings("ignore", category=FutureWarning, module=r"ray\._private\.worker")
    python_executable = os.environ.get("SCHEDULER_RAY_PYTHON")
    if python_executable:
        os.environ["RAY_PYTHON_EXECUTABLE"] = python_executable
    ray.init(
        address=args.ray_address,
        namespace=args.namespace,
        log_to_driver=False,
        logging_level=logging.ERROR,
        configure_logging=False,
    )
    try:
        actor = ray.get_actor(args.actor_name, namespace=args.namespace)
        result = ray.get(
            actor.generate.remote(
                input_ids=args.token_ids,
                max_new_tokens=args.max_new_tokens,
                repetition_window=args.repetition_window,
                pipeline_mode=args.pipeline_mode,
                reset_kv=not args.no_reset_kv,
                request_id=args.request_id,
                session_id=args.session_id,
            )
        )
    except Exception as exc:
        print("STATUS ERR")
        print(f"ERROR {exc}")
        return 1
    finally:
        ray.shutdown()

    print("STATUS OK")
    print("OUTPUT_IDS " + " ".join(str(x) for x in result.get("output_ids", [])))
    print(f"PREFILL_TOKENS {result.get('prefill_tokens', 0)}")
    print(f"DECODE_TOKENS {result.get('decode_tokens', 0)}")
    print(f"PREFILL_MS {result.get('prefill_ms', 0.0)}")
    print(f"DECODE_MS {result.get('decode_ms', 0.0)}")
    print(f"HIT_STOP {1 if result.get('hit_stop', False) else 0}")
    print(f"HIT_REPETITION {1 if result.get('hit_repetition', False) else 0}")
    print(f"REQUEST_COUNT {result.get('request_count', 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
