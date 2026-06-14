from __future__ import annotations

import argparse

import ray


DEFAULT_NAMESPACE = "rk3588-npu"
DEFAULT_ACTOR_NAME = "full-model-worker"


def add_runtime_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--ray-address", default=None, help="Ray address, e.g. auto or ray://host:10001")
    parser.add_argument("--ray-namespace", default=DEFAULT_NAMESPACE)
    parser.add_argument("--actor-name", default=DEFAULT_ACTOR_NAME)
    return parser


def init_ray(address: str | None, namespace: str) -> None:
    kwargs = {
        "namespace": namespace,
        "log_to_driver": True,
    }
    if address:
        kwargs["address"] = address
    else:
        kwargs["include_dashboard"] = False
    ray.init(**kwargs)


def build_actor_options(device: str, detached: bool = True) -> dict:
    options: dict = {
        "name": DEFAULT_ACTOR_NAME,
        "max_restarts": 0,
    }
    if detached:
        options["lifetime"] = "detached"
    if device in {"gpu", "auto"}:
        options["num_gpus"] = 1
    return options
