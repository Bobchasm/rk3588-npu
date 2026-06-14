from __future__ import annotations

import os
import time

import ray

from runtime import create_engine


@ray.remote(max_concurrency=1)
class FullModelWorkerActor:
    def __init__(
        self,
        target: str,
        model_dir: str,
        device: str = "cpu",
        actor_name: str = "full-model-worker",
    ) -> None:
        self._target = target
        self._model_dir = model_dir
        self._device = device
        self._actor_name = actor_name
        self._pid = os.getpid()
        self._request_count = 0
        self._load_started_at = time.time()
        print(
            f"[ray/FullModelWorkerActor] init begin pid={self._pid} "
            f"name={self._actor_name} "
            f"target={target} device={device} model_dir={model_dir}",
            flush=True,
        )
        print(
            "[ray/FullModelWorkerActor] env "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
            f"NVIDIA_VISIBLE_DEVICES={os.environ.get('NVIDIA_VISIBLE_DEVICES')}",
            flush=True,
        )
        self._engine = create_engine(target)
        self._engine.load(model_dir, device)
        self._loaded_at = time.time()
        print(
            f"[ray/FullModelWorkerActor] init done pid={self._pid} "
            f"name={self._actor_name} "
            f"target={self._target} device={self._device}",
            flush=True,
        )
        print("[ray/FullModelWorkerActor] ---------- ready for requests ----------", flush=True)

    def metadata(self) -> dict:
        return {
            "actor_name": self._actor_name,
            "pid": self._pid,
            "target": self._target,
            "device": self._device,
            "model_dir": self._model_dir,
            "request_count": self._request_count,
            "load_started_at": self._load_started_at,
            "loaded_at": getattr(self, "_loaded_at", None),
        }

    def generate(
        self,
        input_ids: list[int],
        max_new_tokens: int = 10,
        repetition_window: int = 6,
        stop_tokens: list[int] | None = None,
        reset_kv: bool = True,
        request_id: str | None = None,
    ) -> dict:
        started = time.time()
        self._request_count += 1
        rid = request_id or f"req-{self._pid}-{self._request_count}"
        print(
            f"[ray/FullModelWorkerActor] generate begin pid={self._pid} "
            f"name={self._actor_name} request_id={rid} input_tokens={len(input_ids)} "
            f"max_new_tokens={max_new_tokens} reset_kv={reset_kv}",
            flush=True,
        )
        if reset_kv:
            self._engine.reset()
        result = self._engine.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            repetition_window=repetition_window,
            stop_tokens=stop_tokens or [],
        )
        payload = {
            "request_id": rid,
            "output_ids": result.output_ids,
            "prefill_tokens": result.prefill_tokens,
            "decode_tokens": result.decode_tokens,
            "prefill_ms": result.prefill_ms,
            "decode_ms": result.decode_ms,
            "hit_stop": result.hit_stop,
            "hit_repetition": result.hit_repetition,
            "target": self._target,
            "device": self._device,
            "actor_name": self._actor_name,
            "elapsed_ms": (time.time() - started) * 1000.0,
            "request_count": self._request_count,
        }
        print(
            f"[ray/FullModelWorkerActor] generate done pid={self._pid} "
            f"name={self._actor_name} request_id={rid} "
            f"output_tokens={payload['decode_tokens']}",
            flush=True,
        )
        return payload

    def reset(self) -> None:
        self._engine.reset()

    def shutdown(self) -> None:
        print(
            f"[ray/FullModelWorkerActor] shutdown pid={self._pid} "
            f"name={self._actor_name} requests={self._request_count}",
            flush=True,
        )
        self._engine.destroy()
