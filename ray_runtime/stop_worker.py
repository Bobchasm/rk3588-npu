from __future__ import annotations

import argparse

import ray

from ray_common import add_runtime_args, init_ray


def parse_args():
    parser = argparse.ArgumentParser()
    add_runtime_args(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    init_ray(args.ray_address or "auto", args.ray_namespace, args.object_store_memory_mb)
    actor = ray.get_actor(args.actor_name, namespace=args.ray_namespace)
    ray.get(actor.shutdown.remote())
    ray.kill(actor, no_restart=True)
    ray.shutdown()
    print(f"[ray/stop_worker] stopped actor={args.actor_name}", flush=True)


if __name__ == "__main__":
    main()
