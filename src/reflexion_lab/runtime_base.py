from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .schemas import JudgeResult, QAExample, ReflectionEntry


@dataclass
class RuntimeCall:
    content: str
    total_tokens: int
    latency_ms: int
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_prompt_tokens: int = 0
    cost_usd: float = 0.0


class RuntimeAdapter(Protocol):
    mode_name: str

    def actor(
        self,
        example: QAExample,
        attempt_id: int,
        reflection_memory: list[str],
        trajectory: list[str],
    ) -> RuntimeCall:
        ...

    def evaluator(
        self,
        example: QAExample,
        answer: str,
        attempt_id: int,
        reflection_memory: list[str],
        trajectory: list[str],
    ) -> tuple[JudgeResult, RuntimeCall]:
        ...

    def reflector(
        self,
        example: QAExample,
        answer: str,
        judge: JudgeResult,
        attempt_id: int,
        reflection_memory: list[str],
        trajectory: list[str],
    ) -> tuple[ReflectionEntry, RuntimeCall]:
        ...
