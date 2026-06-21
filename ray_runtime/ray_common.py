from __future__ import annotations

import argparse
import os

import ray


DEFAULT_NAMESPACE = "rk3588-npu"
DEFAULT_ACTOR_NAME = "full-model-worker"
DEFAULT_OBJECT_STORE_MEMORY_MB = 128


def add_runtime_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--ray-address", default=None, help="Ray address, e.g. auto or ray://host:10001")
    parser.add_argument("--ray-namespace", default=DEFAULT_NAMESPACE)
    parser.add_argument("--actor-name", default=DEFAULT_ACTOR_NAME)
    parser.add_argument(
        "--object-store-memory-mb",
        type=int,
        default=DEFAULT_OBJECT_STORE_MEMORY_MB,
        help="Only used when starting a local Ray instance; reduces Ray plasma reservation for low-memory machines.",
    )
    parser.add_argument(
        "--gpu-fraction",
        type=float,
        default=1.0,
        help="GPU resource fraction requested by each GPU-backed Ray actor.",
    )
    parser.add_argument(
        "--head-resource",
        default=None,
        help="Custom Ray resource key required by the head actor, e.g. role_head.",
    )
    parser.add_argument(
        "--stage-resource",
        default=None,
        help="Custom Ray resource key required by each stage actor, e.g. role_stage.",
    )
    parser.add_argument(
        "--tail-resource",
        default=None,
        help="Custom Ray resource key required by the tail actor, e.g. role_tail.",
    )
    parser.add_argument(
        "--pipeline-resource",
        default=None,
        help="Custom Ray resource key required by the pipeline actor.",
    )
    return parser


def init_ray(address: str | None, namespace: str, object_store_memory_mb: int = DEFAULT_OBJECT_STORE_MEMORY_MB) -> None:
    runtime_env: dict = {}
    env_vars: dict[str, str] = {}
    for key in ("PYTHONPATH", "LD_LIBRARY_PATH", "PATH"):
        value = os.environ.get(key)
        if value:
            env_vars[key] = value
    if env_vars:
        runtime_env["env_vars"] = env_vars

    kwargs = {
        "namespace": namespace,
        "log_to_driver": True,
        "runtime_env": runtime_env,
    }
    if address:
        kwargs["address"] = address
    else:
        kwargs["include_dashboard"] = False
        kwargs["object_store_memory"] = object_store_memory_mb * 1024 * 1024
    ray.init(**kwargs)


def build_actor_options(target: str, device: str, detached: bool = True, gpu_fraction: float = 1.0) -> dict:
    options: dict = {
        "name": DEFAULT_ACTOR_NAME,
        "max_restarts": 0,
    }
    if detached:
        options["lifetime"] = "detached"
    # Only the pc target currently participates in Ray GPU resource scheduling.
    # rk3588 workers use the board-local NPU/CPU backend selection inside the engine
    # and should not request Ray GPU slots.
    if target == "pc" and device in {"gpu", "auto"}:
        options["num_gpus"] = gpu_fraction
    return options


def with_custom_resource(options: dict, resource_key: str | None) -> dict:
    updated = dict(options)
    if resource_key:
        updated["resources"] = {resource_key: 0.001}
    return updated
