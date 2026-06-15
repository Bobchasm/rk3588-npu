from __future__ import annotations

import os
import time

import ray

from runtime import create_engine


def _default_request_id(pid: int, count: int) -> str:
    return f"req-{pid}-{count}"


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
        self._engine = create_engine(target)
        self._engine.load(model_dir, device)
        print(
            f"[ray/FullModelWorkerActor] ready pid={self._pid} "
            f"name={self._actor_name} target={target} device={device}",
            flush=True,
        )

    def metadata(self) -> dict:
        return {
            "actor_name": self._actor_name,
            "pid": self._pid,
            "target": self._target,
            "device": self._device,
            "model_dir": self._model_dir,
            "request_count": self._request_count,
        }

    def generate(
        self,
        input_ids: list[int],
        max_new_tokens: int = 64,
        repetition_window: int = 6,
        stop_tokens: list[int] | None = None,
        reset_kv: bool = True,
        request_id: str | None = None,
    ) -> dict:
        started = time.time()
        self._request_count += 1
        rid = request_id or _default_request_id(self._pid, self._request_count)
        if reset_kv:
            self._engine.reset()
        result = self._engine.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            repetition_window=repetition_window,
            stop_tokens=stop_tokens or [],
        )
        return {
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

    def reset(self) -> None:
        self._engine.reset()

    def shutdown(self) -> None:
        self._engine.destroy()


@ray.remote(max_concurrency=1)
class HeadWorkerActor:
    def __init__(
        self,
        target: str,
        model_dir: str,
        device: str,
        actor_name: str,
        layer_begin: int = 0,
        layer_end: int = 4,
    ) -> None:
        self._target = target
        self._model_dir = model_dir
        self._device = device
        self._actor_name = actor_name
        self._pid = os.getpid()
        self._engine = create_engine(target)
        self._engine.load(model_dir, device, layer_begin, layer_end, True, False)
        print(f"[ray/HeadWorkerActor] ready pid={self._pid} name={actor_name}", flush=True)

    def tokens_to_hidden(self, input_ids: list[int]) -> list[int]:
        return self._engine.tokens_to_hidden(list(input_ids))

    def reset(self) -> None:
        self._engine.reset()

    def metadata(self) -> dict:
        return {"actor_name": self._actor_name, "pid": self._pid, "role": "head"}

    def shutdown(self) -> None:
        self._engine.destroy()


@ray.remote(max_concurrency=1)
class StageWorkerActor:
    def __init__(
        self,
        target: str,
        model_dir: str,
        device: str,
        actor_name: str,
        layer_begin: int = 4,
        layer_end: int = 8,
    ) -> None:
        self._target = target
        self._model_dir = model_dir
        self._device = device
        self._actor_name = actor_name
        self._pid = os.getpid()
        self._engine = create_engine(target)
        self._engine.load(model_dir, device, layer_begin, layer_end, False, False)
        print(f"[ray/StageWorkerActor] ready pid={self._pid} name={actor_name}", flush=True)

    def hidden_forward(self, input_f16: list[int], seq: int, pos_base: int) -> list[int]:
        return self._engine.hidden_forward(list(input_f16), seq, pos_base)

    def reset(self) -> None:
        self._engine.reset()

    def metadata(self) -> dict:
        return {"actor_name": self._actor_name, "pid": self._pid, "role": "stage"}

    def shutdown(self) -> None:
        self._engine.destroy()


@ray.remote(max_concurrency=1)
class TailWorkerActor:
    def __init__(
        self,
        target: str,
        model_dir: str,
        device: str,
        actor_name: str,
        layer_begin: int = 12,
        layer_end: int = 28,
    ) -> None:
        self._target = target
        self._model_dir = model_dir
        self._device = device
        self._actor_name = actor_name
        self._pid = os.getpid()
        self._engine = create_engine(target)
        self._engine.load(model_dir, device, layer_begin, layer_end, False, True)
        print(f"[ray/TailWorkerActor] ready pid={self._pid} name={actor_name}", flush=True)

    def hidden_to_token(self, input_f16: list[int], seq: int, pos_base: int) -> int:
        return self._engine.hidden_to_token(list(input_f16), seq, pos_base)

    def reset(self) -> None:
        self._engine.reset()

    def metadata(self) -> dict:
        return {"actor_name": self._actor_name, "pid": self._pid, "role": "tail"}

    def shutdown(self) -> None:
        self._engine.destroy()


@ray.remote(max_concurrency=1)
class DistributedPipelineActor:
    def __init__(self, head, stages: list, tail, actor_name: str = "distributed-pipeline") -> None:
        self._head = head
        self._stages = stages
        self._tail = tail
        self._actor_name = actor_name
        self._pid = os.getpid()
        self._request_count = 0
        print(f"[ray/DistributedPipelineActor] ready pid={self._pid} name={actor_name}", flush=True)

    def metadata(self) -> dict:
        return {
            "actor_name": self._actor_name,
            "pid": self._pid,
            "request_count": self._request_count,
            "num_stages": len(self._stages),
        }

    def reset(self) -> None:
        ray.get(self._head.reset.remote())
        for stage in self._stages:
            ray.get(stage.reset.remote())
        ray.get(self._tail.reset.remote())

    def generate(
        self,
        input_ids: list[int],
        max_new_tokens: int = 64,
        repetition_window: int = 6,
        stop_tokens: list[int] | None = None,
        reset_kv: bool = True,
        request_id: str | None = None,
    ) -> dict:
        started = time.time()
        self._request_count += 1
        rid = request_id or _default_request_id(self._pid, self._request_count)
        stop_set = set(stop_tokens or [151645, 151643])
        if reset_kv:
            self.reset()

        generated_ids: list[int] = []
        current_input = list(input_ids)
        prompt_len = len(input_ids)

        for _step in range(max_new_tokens):
            pos_base = prompt_len + len(generated_ids) - len(current_input)
            hidden = ray.get(self._head.tokens_to_hidden.remote(current_input))
            seq = len(current_input)
            for stage in self._stages:
                hidden = ray.get(stage.hidden_forward.remote(hidden, seq, pos_base))
            token_id = ray.get(self._tail.hidden_to_token.remote(hidden, seq, pos_base))
            generated_ids.append(token_id)
            if token_id in stop_set:
                break
            current_input = [token_id]

        return {
            "request_id": rid,
            "output_ids": generated_ids,
            "prefill_tokens": len(input_ids),
            "decode_tokens": len(generated_ids),
            "prefill_ms": 0.0,
            "decode_ms": 0.0,
            "hit_stop": bool(generated_ids and generated_ids[-1] in stop_set),
            "hit_repetition": False,
            "actor_name": self._actor_name,
            "elapsed_ms": (time.time() - started) * 1000.0,
            "request_count": self._request_count,
        }

    def shutdown(self) -> None:
        ray.get(self._head.shutdown.remote())
        for stage in self._stages:
            ray.get(stage.shutdown.remote())
        ray.get(self._tail.shutdown.remote())
