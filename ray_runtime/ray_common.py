from __future__ import annotations

import argparse

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
    return parser


def init_ray(address: str | None, namespace: str, object_store_memory_mb: int = DEFAULT_OBJECT_STORE_MEMORY_MB) -> None:
    kwargs = {
        "namespace": namespace,
        "log_to_driver": True,
    }
    if address:
        kwargs["address"] = address
    else:
        kwargs["include_dashboard"] = False
        kwargs["object_store_memory"] = object_store_memory_mb * 1024 * 1024
    ray.init(**kwargs)


def build_actor_options(device: str, detached: bool = True, gpu_fraction: float = 1.0) -> dict:
    options: dict = {
        "name": DEFAULT_ACTOR_NAME,
        "max_restarts": 0,
    }
    if detached:
        options["lifetime"] = "detached"
    if device in {"gpu", "auto"}:
        options["num_gpus"] = gpu_fraction
    return options
