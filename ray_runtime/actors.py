from __future__ import annotations

import os
import time

import ray

from runtime import create_engine


def _default_request_id(pid: int, count: int) -> str:
    return f"req-{pid}-{count}"


def _common_prefix_len(lhs: list[int], rhs: list[int]) -> int:
    limit = min(len(lhs), len(rhs))
    index = 0
    while index < limit and lhs[index] == rhs[index]:
        index += 1
    return index


def _run_pipeline_step_local(head, stages: list, tail, current_input: list[int], pos_base: int) -> int:
    hidden = ray.get(head.tokens_to_hidden.remote(current_input))
    seq = len(current_input)
    for stage in stages:
        hidden = ray.get(stage.hidden_forward.remote(hidden, seq, pos_base))
    return ray.get(tail.hidden_to_token.remote(hidden, seq, pos_base))


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
        self._active_session_id: str | None = None
        self._cached_tokens: list[int] = []
        self._session_caches: dict[str, dict] = {}
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
            "active_session_id": self._active_session_id,
            "cached_tokens": len(self._cached_tokens),
            "cached_sessions": len(self._session_caches),
        }

    def _reset_session_state(self) -> None:
        self._engine.reset()
        self._active_session_id = None
        self._cached_tokens = []

    def _prepare_prompt(self,
                        session_id: str | None,
                        input_ids: list[int],
                        reset_kv: bool) -> tuple[list[int], int, bool]:
        if reset_kv or not session_id:
            self._reset_session_state()
            if session_id:
                self._active_session_id = session_id
            return list(input_ids), 0, False

        if self._active_session_id != session_id:
            cached = self._session_caches.get(session_id)
            if cached is not None:
                self._engine.restore_kv_state(cached["kv_state"])
                self._active_session_id = session_id
                self._cached_tokens = list(cached["cached_tokens"])
            else:
                self._reset_session_state()
                self._active_session_id = session_id
                self._cached_tokens = []

        prefix_len = _common_prefix_len(self._cached_tokens, input_ids)
        if prefix_len == len(self._cached_tokens) and prefix_len < len(input_ids):
            delta = list(input_ids[prefix_len:])
            if delta:
                return delta, prefix_len, True

        self._reset_session_state()
        self._active_session_id = session_id
        return list(input_ids), 0, False

    def generate(
        self,
        input_ids: list[int],
        max_new_tokens: int = 64,
        repetition_window: int = 6,
        stop_tokens: list[int] | None = None,
        reset_kv: bool = True,
        request_id: str | None = None,
        session_id: str | None = None,
    ) -> dict:
        started = time.time()
        self._request_count += 1
        rid = request_id or _default_request_id(self._pid, self._request_count)
        effective_input_ids, reused_tokens, cache_reused = self._prepare_prompt(
            session_id,
            list(input_ids),
            reset_kv,
        )
        result = self._engine.generate(
            input_ids=effective_input_ids,
            max_new_tokens=max_new_tokens,
            repetition_window=repetition_window,
            stop_tokens=stop_tokens or [],
        )
        if session_id:
            self._active_session_id = session_id
            self._cached_tokens = list(input_ids) + list(result.output_ids)
            self._session_caches[session_id] = {
                "kv_state": self._engine.snapshot_kv_state(),
                "cached_tokens": list(self._cached_tokens),
            }
        elif reset_kv:
            self._cached_tokens = []
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
            "session_id": session_id or "",
            "cache_reused": cache_reused,
            "reused_tokens": reused_tokens,
            "effective_prefill_tokens": len(effective_input_ids),
        }

    def reset(self) -> None:
        self._reset_session_state()

    def clear_session(self, session_id: str | None = None) -> None:
        if session_id is None:
            self._session_caches.clear()
            self._reset_session_state()
        elif session_id == self._active_session_id:
            self._session_caches.pop(session_id, None)
            self._reset_session_state()
        else:
            self._session_caches.pop(session_id, None)

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
        self._active_session_id: str | None = None
        self._session_caches: dict[str, object] = {}
        print(f"[ray/HeadWorkerActor] ready pid={self._pid} name={actor_name}", flush=True)

    def tokens_to_hidden(self, input_ids: list[int]) -> list[int]:
        return self._engine.tokens_to_hidden(list(input_ids))

    def reset(self) -> None:
        self._engine.reset()
        self._active_session_id = None

    def activate_session(self, session_id: str) -> bool:
        if self._active_session_id == session_id:
            return True
        state = self._session_caches.get(session_id)
        if state is None:
            self._engine.reset()
            self._active_session_id = session_id
            return False
        self._engine.restore_kv_state(state)
        self._active_session_id = session_id
        return True

    def save_active_session(self, session_id: str) -> None:
        self._session_caches[session_id] = self._engine.snapshot_kv_state()
        self._active_session_id = session_id

    def clear_session(self, session_id: str | None = None) -> None:
        if session_id is None:
            self._session_caches.clear()
            self.reset()
            return
        self._session_caches.pop(session_id, None)
        if session_id == self._active_session_id:
            self.reset()

    def metadata(self) -> dict:
        return {
            "actor_name": self._actor_name,
            "pid": self._pid,
            "role": "head",
            "active_session_id": self._active_session_id,
            "cached_sessions": len(self._session_caches),
        }

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
        self._active_session_id: str | None = None
        self._session_caches: dict[str, object] = {}
        print(f"[ray/StageWorkerActor] ready pid={self._pid} name={actor_name}", flush=True)

    def hidden_forward(self, input_f16: list[int], seq: int, pos_base: int) -> list[int]:
        return self._engine.hidden_forward(list(input_f16), seq, pos_base)

    def reset(self) -> None:
        self._engine.reset()
        self._active_session_id = None

    def activate_session(self, session_id: str) -> bool:
        if self._active_session_id == session_id:
            return True
        state = self._session_caches.get(session_id)
        if state is None:
            self._engine.reset()
            self._active_session_id = session_id
            return False
        self._engine.restore_kv_state(state)
        self._active_session_id = session_id
        return True

    def save_active_session(self, session_id: str) -> None:
        self._session_caches[session_id] = self._engine.snapshot_kv_state()
        self._active_session_id = session_id

    def clear_session(self, session_id: str | None = None) -> None:
        if session_id is None:
            self._session_caches.clear()
            self.reset()
            return
        self._session_caches.pop(session_id, None)
        if session_id == self._active_session_id:
            self.reset()

    def metadata(self) -> dict:
        return {
            "actor_name": self._actor_name,
            "pid": self._pid,
            "role": "stage",
            "active_session_id": self._active_session_id,
            "cached_sessions": len(self._session_caches),
        }

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
        self._active_session_id: str | None = None
        self._session_caches: dict[str, object] = {}
        print(f"[ray/TailWorkerActor] ready pid={self._pid} name={actor_name}", flush=True)

    def hidden_to_token(self, input_f16: list[int], seq: int, pos_base: int) -> int:
        return self._engine.hidden_to_token(list(input_f16), seq, pos_base)

    def reset(self) -> None:
        self._engine.reset()
        self._active_session_id = None

    def activate_session(self, session_id: str) -> bool:
        if self._active_session_id == session_id:
            return True
        state = self._session_caches.get(session_id)
        if state is None:
            self._engine.reset()
            self._active_session_id = session_id
            return False
        self._engine.restore_kv_state(state)
        self._active_session_id = session_id
        return True

    def save_active_session(self, session_id: str) -> None:
        self._session_caches[session_id] = self._engine.snapshot_kv_state()
        self._active_session_id = session_id

    def clear_session(self, session_id: str | None = None) -> None:
        if session_id is None:
            self._session_caches.clear()
            self.reset()
            return
        self._session_caches.pop(session_id, None)
        if session_id == self._active_session_id:
            self.reset()

    def metadata(self) -> dict:
        return {
            "actor_name": self._actor_name,
            "pid": self._pid,
            "role": "tail",
            "active_session_id": self._active_session_id,
            "cached_sessions": len(self._session_caches),
        }

    def shutdown(self) -> None:
        self._engine.destroy()


@ray.remote(max_concurrency=1)
class DistributedPipelineActor:
    def __init__(
        self,
        head,
        stages: list,
        tail,
        actor_name: str = "distributed-pipeline",
        pipeline_mode: str = "centralized",
    ) -> None:
        self._head = head
        self._stages = stages
        self._tail = tail
        self._actor_name = actor_name
        self._pipeline_mode = pipeline_mode
        self._pid = os.getpid()
        self._request_count = 0
        self._active_session_id: str | None = None
        self._cached_tokens: list[int] = []
        self._session_caches: dict[str, dict] = {}
        print(
            f"[ray/DistributedPipelineActor] ready pid={self._pid} "
            f"name={actor_name} mode={pipeline_mode}",
            flush=True,
        )

    def metadata(self) -> dict:
        return {
            "actor_name": self._actor_name,
            "pid": self._pid,
            "request_count": self._request_count,
            "num_stages": len(self._stages),
            "pipeline_mode": self._pipeline_mode,
            "active_session_id": self._active_session_id,
            "cached_tokens": len(self._cached_tokens),
            "cached_sessions": len(self._session_caches),
        }

    def _reset_pipeline_state(self) -> None:
        ray.get(self._head.reset.remote())
        for stage in self._stages:
            ray.get(stage.reset.remote())
        ray.get(self._tail.reset.remote())
        self._active_session_id = None
        self._cached_tokens = []

    def reset(self) -> None:
        self._reset_pipeline_state()

    def clear_session(self, session_id: str | None = None) -> None:
        if session_id is None:
            self._session_caches.clear()
            self._reset_pipeline_state()
        elif session_id == self._active_session_id:
            self._session_caches.pop(session_id, None)
            self._reset_pipeline_state()
        else:
            self._session_caches.pop(session_id, None)

    def _prepare_prompt(self,
                        session_id: str | None,
                        input_ids: list[int],
                        reset_kv: bool) -> tuple[list[int], int, bool]:
        if reset_kv or not session_id:
            self._reset_pipeline_state()
            if session_id:
                self._active_session_id = session_id
            return list(input_ids), 0, False

        if self._active_session_id != session_id:
            head_hit, *stage_hits, tail_hit = ray.get(
                [
                    self._head.activate_session.remote(session_id),
                    *[stage.activate_session.remote(session_id) for stage in self._stages],
                    self._tail.activate_session.remote(session_id),
                ]
            )
            cached = self._session_caches.get(session_id)
            self._active_session_id = session_id
            if cached is not None and head_hit and tail_hit and all(stage_hits):
                self._cached_tokens = list(cached["cached_tokens"])
            else:
                self._cached_tokens = []

        prefix_len = _common_prefix_len(self._cached_tokens, input_ids)
        if prefix_len == len(self._cached_tokens) and prefix_len < len(input_ids):
            delta = list(input_ids[prefix_len:])
            if delta:
                return delta, prefix_len, True

        self._reset_pipeline_state()
        self._active_session_id = session_id
        return list(input_ids), 0, False

    def _generate_centralized(
        self,
        input_ids: list[int],
        max_new_tokens: int,
        stop_set: set[int],
    ) -> list[int]:
        generated_ids: list[int] = []
        current_input = list(input_ids)
        prompt_len = len(input_ids)

        for _step in range(max_new_tokens):
            pos_base = prompt_len + len(generated_ids) - len(current_input)
            token_id = _run_pipeline_step_local(
                self._head,
                self._stages,
                self._tail,
                current_input,
                pos_base,
            )
            generated_ids.append(token_id)
            if token_id in stop_set:
                break
            current_input = [token_id]
        return generated_ids

    def _generate_p2p(
        self,
        input_ids: list[int],
        max_new_tokens: int,
        stop_set: set[int],
    ) -> list[int]:
        # Current P2P mode keeps the same actor graph but switches to stage-to-stage chaining
        # semantics in the control layer. The physical transport is still handled by Ray.
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
        return generated_ids

    def generate(
        self,
        input_ids: list[int],
        max_new_tokens: int = 64,
        repetition_window: int = 6,
        stop_tokens: list[int] | None = None,
        reset_kv: bool = True,
        request_id: str | None = None,
        pipeline_mode: str | None = None,
        session_id: str | None = None,
    ) -> dict:
        started = time.time()
        self._request_count += 1
        rid = request_id or _default_request_id(self._pid, self._request_count)
        stop_set = set(stop_tokens or [151645, 151643])
        effective_input_ids, reused_tokens, cache_reused = self._prepare_prompt(
            session_id,
            list(input_ids),
            reset_kv,
        )
        active_mode = pipeline_mode or self._pipeline_mode
        if active_mode == "centralized":
            generated_ids = self._generate_centralized(effective_input_ids, max_new_tokens, stop_set)
        elif active_mode == "p2p":
            generated_ids = self._generate_p2p(effective_input_ids, max_new_tokens, stop_set)
        else:
            raise ValueError(f"unsupported pipeline_mode: {active_mode}")

        if session_id:
            self._active_session_id = session_id
            self._cached_tokens = list(input_ids) + list(generated_ids)
            ray.get(self._head.save_active_session.remote(session_id))
            for stage in self._stages:
                ray.get(stage.save_active_session.remote(session_id))
            ray.get(self._tail.save_active_session.remote(session_id))
            self._session_caches[session_id] = {
                "cached_tokens": list(self._cached_tokens),
            }
        elif reset_kv:
            self._cached_tokens = []

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
            "pipeline_mode": active_mode,
            "elapsed_ms": (time.time() - started) * 1000.0,
            "request_count": self._request_count,
            "session_id": session_id or "",
            "cache_reused": cache_reused,
            "reused_tokens": reused_tokens,
            "effective_prefill_tokens": len(effective_input_ids),
        }

    def shutdown(self) -> None:
        ray.get(self._head.shutdown.remote())
        for stage in self._stages:
            ray.get(stage.shutdown.remote())
        ray.get(self._tail.shutdown.remote())
