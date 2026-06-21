from __future__ import annotations

import argparse

import ray

from actors import FullModelWorkerActor
from ray_common import add_runtime_args, build_actor_options, init_ray


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("--target", default="pc", choices=["pc", "rk3588"])
    parser.add_argument("--device", default="auto",
                        help="pc target: cpu/gpu/auto; rk3588 target: cpu/npu/auto/single/sharded")
    parser.add_argument("--max-new-tokens", type=int, default=10)
    parser.add_argument("token_ids", nargs="+", type=int)
    add_runtime_args(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    init_ray(args.ray_address, args.ray_namespace)

    actor_options = build_actor_options(args.target, args.device, detached=False)
    actor_options["name"] = args.actor_name
    actor = FullModelWorkerActor.options(**actor_options).remote(
        args.target,
        args.model_dir,
        args.device,
        args.actor_name,
    )
    result = ray.get(
        actor.generate.remote(
            args.token_ids,
            max_new_tokens=args.max_new_tokens,
        )
    )
    print(result)
    ray.get(actor.shutdown.remote())
    ray.shutdown()


if __name__ == "__main__":
    main()
