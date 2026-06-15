from __future__ import annotations

import argparse
import json
import time

import ray

from ray_common import add_runtime_args, init_ray


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("token_ids", nargs="+", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=10)
    parser.add_argument("--repetition-window", type=int, default=6)
    parser.add_argument("--request-id", default=None)
    parser.add_argument("--no-reset-kv", action="store_true")
    parser.add_argument("--stop-token", dest="stop_tokens", action="append", type=int, default=[])
    add_runtime_args(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    init_ray(args.ray_address or "auto", args.ray_namespace, args.object_store_memory_mb)
    actor = ray.get_actor(args.actor_name, namespace=args.ray_namespace)
    started = time.time()
    result = ray.get(
        actor.generate.remote(
            args.token_ids,
            max_new_tokens=args.max_new_tokens,
            repetition_window=args.repetition_window,
            stop_tokens=args.stop_tokens,
            reset_kv=not args.no_reset_kv,
            request_id=args.request_id,
        )
    )
    result["client_elapsed_ms"] = (time.time() - started) * 1000.0
    print(json.dumps(result, ensure_ascii=False, indent=2))
    ray.shutdown()


if __name__ == "__main__":
    main()
