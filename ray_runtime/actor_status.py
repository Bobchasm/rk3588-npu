from __future__ import annotations

import argparse
import json

import ray

from ray_common import add_runtime_args, init_ray


def parse_args():
    parser = argparse.ArgumentParser()
    add_runtime_args(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    init_ray(args.ray_address or "auto", args.ray_namespace)
    actor = ray.get_actor(args.actor_name, namespace=args.ray_namespace)
    metadata = ray.get(actor.metadata.remote())
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    ray.shutdown()


if __name__ == "__main__":
    main()
