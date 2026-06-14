from __future__ import annotations

import argparse
import signal
import time

import ray
from ray.exceptions import RayError

from actors import FullModelWorkerActor
from ray_common import add_runtime_args, build_actor_options, init_ray


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("--target", default="pc", choices=["pc"])
    parser.add_argument("--device", default="auto", choices=["cpu", "gpu", "auto"])
    parser.add_argument("--detach-only", action="store_true",
                        help="Create detached actor and exit without keeping the driver alive")
    add_runtime_args(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    init_ray(args.ray_address, args.ray_namespace)

    actor_options = build_actor_options(args.device, detached=True)
    actor_options["name"] = args.actor_name
    actor = FullModelWorkerActor.options(**actor_options).remote(
        args.target,
        args.model_dir,
        args.device,
        args.actor_name,
    )
    metadata = ray.get(actor.metadata.remote())
    print(f"[ray/serve_worker] actor ready: {metadata}", flush=True)

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
