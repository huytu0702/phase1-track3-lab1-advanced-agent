from __future__ import annotations

import json

from .pricing import estimate_cost_usd
from .runtime_base import RuntimeAdapter, RuntimeCall
from .schemas import FailureMode, JudgeResult, QAExample, ReflectionEntry
from .utils import normalize_answer

FIRST_ATTEMPT_WRONG = {
    "hp2": "London",
    "hp4": "Atlantic Ocean",
    "hp6": "Red Sea",
    "hp8": "Andes",
}

FAILURE_MODE_BY_QID: dict[str, FailureMode] = {
    "hp2": "incomplete_multi_hop",
    "hp4": "wrong_final_answer",
    "hp6": "entity_drift",
    "hp8": "entity_drift",
}


class MockRuntime(RuntimeAdapter):
    mode_name = "mock"
    model_name = "gpt-5.4-mini"

    def _usage(self, attempt_id: int, kind: str) -> RuntimeCall:
        if kind == "actor":
            prompt_tokens = 120 + (attempt_id * 12)
            completion_tokens = 70 + (attempt_id * 19)
            cached_prompt_tokens = 20
        elif kind == "evaluator":
            prompt_tokens = 95 + (attempt_id * 8)
            completion_tokens = 45 + (attempt_id * 5)
            cached_prompt_tokens = 0
        else:
            prompt_tokens = 80 + (attempt_id * 7)
            completion_tokens = 40 + (attempt_id * 4)
            cached_prompt_tokens = 0

        total_tokens = prompt_tokens + completion_tokens
        latency_ms = (
            70 + (attempt_id * 7)
            if kind == "actor"
            else 50 + (attempt_id * 5)
            if kind == "evaluator"
            else 45 + (attempt_id * 5)
        )
        return RuntimeCall(
            content="",
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            model=self.model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_prompt_tokens=cached_prompt_tokens,
            cost_usd=estimate_cost_usd(
                model=self.model_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cached_prompt_tokens=cached_prompt_tokens,
            ),
        )

    def actor(
        self,
        example: QAExample,
        attempt_id: int,
        reflection_memory: list[str],
        trajectory: list[str],
    ) -> RuntimeCall:
        if example.qid not in FIRST_ATTEMPT_WRONG:
            answer = example.gold_answer
        elif attempt_id == 1 and not reflection_memory:
            answer = FIRST_ATTEMPT_WRONG[example.qid]
        else:
            answer = example.gold_answer
        usage = self._usage(attempt_id, "actor")
        usage.content = answer
        return usage

    def evaluator(
        self,
        example: QAExample,
        answer: str,
        attempt_id: int,
        reflection_memory: list[str],
        trajectory: list[str],
    ) -> tuple[JudgeResult, RuntimeCall]:
        if normalize_answer(example.gold_answer) == normalize_answer(answer):
            result = JudgeResult(
                score=1,
                reason="Final answer matches the gold answer after normalization.",
                missing_evidence=[],
                spurious_claims=[],
                failure_mode="none",
            )
        elif normalize_answer(answer) == "london":
            result = JudgeResult(
                score=0,
                reason="Stopped at the birthplace city and missed the second hop.",
                missing_evidence=["Need to identify the river flowing through London."],
                spurious_claims=[],
                failure_mode="incomplete_multi_hop",
            )
        else:
            result = JudgeResult(
                score=0,
                reason="Selected an incorrect second-hop entity.",
                missing_evidence=["Need to verify the second paragraph entity."],
                spurious_claims=[answer],
                failure_mode=FAILURE_MODE_BY_QID.get(example.qid, "wrong_final_answer"),
            )

        usage = self._usage(attempt_id, "evaluator")
        usage.content = result.model_dump_json()
        return result, usage

    def reflector(
        self,
        example: QAExample,
        answer: str,
        judge: JudgeResult,
        attempt_id: int,
        reflection_memory: list[str],
        trajectory: list[str],
    ) -> tuple[ReflectionEntry, RuntimeCall]:
        strategy = (
            "Explicitly resolve hop-1 entity then map to hop-2 in context."
            if judge.failure_mode == "incomplete_multi_hop"
            else "Ground final entity in the second context chunk before answering."
        )
        reflection = ReflectionEntry(
            attempt_id=attempt_id,
            failure_reason=judge.reason,
            lesson="Complete all hops and verify final entity against supporting evidence.",
            next_strategy=strategy,
        )
        usage = self._usage(attempt_id, "reflector")
        usage.content = json.dumps(reflection.model_dump(), ensure_ascii=True)
        return reflection, usage
