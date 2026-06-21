from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Sequence


@dataclass
class GenerateResult:
    output_ids: list[int]
    prefill_tokens: int
    decode_tokens: int
    prefill_ms: float
    decode_ms: float
    hit_stop: bool
    hit_repetition: bool


class _ModuleEngineAdapter:
    module_name: str = ""

    def __init__(self) -> None:
        module = importlib.import_module(self.module_name, package=__package__)
        self._engine = module.Engine()

    def load(
        self,
        model_dir: str,
        device: str = "cpu",
        layer_begin: int = 0,
        layer_end: int = -1,
        include_embedding: bool = True,
        include_final_norm_and_head: bool = True,
    ) -> None:
        self._engine.load(
            model_dir,
            device,
            layer_begin,
            layer_end,
            include_embedding,
            include_final_norm_and_head,
        )

    def reset(self) -> None:
        self._engine.reset()

    def destroy(self) -> None:
        self._engine.destroy()

    def generate(
        self,
        input_ids: Sequence[int],
        max_new_tokens: int = 10,
        repetition_window: int = 6,
        stop_tokens: Sequence[int] | None = None,
    ) -> GenerateResult:
        raw = self._engine.generate(
            list(input_ids),
            max_new_tokens,
            repetition_window,
            list(stop_tokens or []),
        )
        return GenerateResult(
            output_ids=list(raw.output_ids),
            prefill_tokens=raw.prefill_tokens,
            decode_tokens=raw.decode_tokens,
            prefill_ms=raw.prefill_ms,
            decode_ms=raw.decode_ms,
            hit_stop=raw.hit_stop,
            hit_repetition=raw.hit_repetition,
        )

    def tokens_to_hidden(self, input_ids: Sequence[int]):
        return list(self._engine.tokens_to_hidden(list(input_ids)))

    def hidden_forward(self, input_f16: Sequence[int], seq: int, pos_base: int):
        return list(self._engine.hidden_forward(list(input_f16), seq, pos_base))

    def hidden_to_token(self, input_f16: Sequence[int], seq: int, pos_base: int):
        return int(self._engine.hidden_to_token(list(input_f16), seq, pos_base))

    def supports_partition_pipeline(self) -> bool:
        return all(
            hasattr(self._engine, name)
            for name in ("tokens_to_hidden", "hidden_forward", "hidden_to_token")
        )


class PcEngineAdapter(_ModuleEngineAdapter):
    module_name = ".pc_engine"


class Rk3588EngineAdapter(_ModuleEngineAdapter):
    module_name = ".rk3588_engine"


def create_engine(target: str = "pc"):
    normalized = target.strip().lower()
    if normalized == "pc":
        return PcEngineAdapter()
    if normalized in {"rk3588", "rk3588-board"}:
        return Rk3588EngineAdapter()
    raise ValueError(f"unsupported engine target: {target}")
