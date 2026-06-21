from __future__ import annotations

import argparse
import signal
import time

import ray
from ray.exceptions import RayError

from actors import DistributedPipelineActor, FullModelWorkerActor, HeadWorkerActor, StageWorkerActor, TailWorkerActor
from ray_common import add_runtime_args, build_actor_options, init_ray, with_custom_resource


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("--target", default="pc", choices=["pc", "rk3588"])
    parser.add_argument("--device", default="auto",
                        help="pc target: cpu/gpu/auto; rk3588 target: cpu/npu/auto/single/sharded")
    parser.add_argument("--mode", default="full", choices=["full", "distributed"])
    parser.add_argument("--pipeline-mode", default="centralized", choices=["centralized", "p2p"])
    parser.add_argument("--num-stages", type=int, default=2)
    parser.add_argument("--detach-only", action="store_true",
                        help="Create detached actor and exit without keeping the driver alive")
    add_runtime_args(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    init_ray(args.ray_address, args.ray_namespace, args.object_store_memory_mb)

    actor_options = build_actor_options(args.target, args.device, detached=True, gpu_fraction=args.gpu_fraction)

    if args.mode == "full":
        full_actor_options = dict(actor_options)
        full_actor_options["name"] = args.actor_name
        actor = FullModelWorkerActor.options(**full_actor_options).remote(
            args.target,
            args.model_dir,
            args.device,
            args.actor_name,
        )
        metadata = ray.get(actor.metadata.remote())
        print(f"[ray/serve_worker] actor ready: {metadata}", flush=True)
    else:
        if args.num_stages == 1:
            stage_ranges = [(4, 12)]
            tail_begin = 12
        elif args.num_stages == 2:
            stage_ranges = [(4, 8), (8, 12)]
            tail_begin = 12
        else:
            raise ValueError("currently only --num-stages 1 or 2 is supported")

        head_actor_options = with_custom_resource(actor_options, args.head_resource)
        head_actor_options["name"] = args.actor_name + "-head"
        head = HeadWorkerActor.options(**head_actor_options).remote(
            args.target,
            args.model_dir,
            args.device,
            args.actor_name + "-head",
            0,
            4,
        )
        ray.get(head.metadata.remote())
        stages = [
            None
            for _ in range(args.num_stages)
        ]
        for i in range(args.num_stages):
            stage_actor_options = with_custom_resource(actor_options, args.stage_resource)
            stage_actor_options["name"] = f"{args.actor_name}-stage-{i}"
            stages[i] = StageWorkerActor.options(**stage_actor_options).remote(
                args.target,
                args.model_dir,
                args.device,
                f"{args.actor_name}-stage-{i}",
                stage_ranges[i][0],
                stage_ranges[i][1],
            )
            ray.get(stages[i].metadata.remote())
        tail_actor_options = with_custom_resource(actor_options, args.tail_resource)
        tail_actor_options["name"] = args.actor_name + "-tail"
        tail = TailWorkerActor.options(**tail_actor_options).remote(
            args.target,
            args.model_dir,
            args.device,
            args.actor_name + "-tail",
            tail_begin,
            28,
        )
        ray.get(tail.metadata.remote())
        pipeline_actor_options = with_custom_resource({
            "name": args.actor_name,
            "max_restarts": 0,
        }, args.pipeline_resource)
        if actor_options.get("lifetime") == "detached":
            pipeline_actor_options["lifetime"] = "detached"
        actor = DistributedPipelineActor.options(**pipeline_actor_options).remote(
            head,
            stages,
            tail,
            args.actor_name,
            args.pipeline_mode,
        )
        metadata = ray.get(actor.metadata.remote())
        print(f"[ray/serve_worker] distributed actor ready: {metadata}", flush=True)

    if args.detach_only:
        print("[ray/serve_worker] detach-only mode, exiting launcher", flush=True)
        ray.shutdown()
        return

    stop = {"value": False}

    def handle_signal(_signum, _frame):
        stop["value"] = True

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    print("[ray/serve_worker] service running, press Ctrl+C to stop", flush=True)
    try:
        while not stop["value"]:
            time.sleep(5.0)
            try:
                ray.get(actor.metadata.remote())
            except RayError:
                print("[ray/serve_worker] actor is no longer alive, exiting", flush=True)
                stop["value"] = True
    finally:
        try:
            ray.get(actor.shutdown.remote())
        except RayError:
            pass
        finally:
            try:
                ray.kill(actor, no_restart=True)
            except ValueError:
                pass
            ray.shutdown()
            print("[ray/serve_worker] service stopped", flush=True)


if __name__ == "__main__":
    main()
