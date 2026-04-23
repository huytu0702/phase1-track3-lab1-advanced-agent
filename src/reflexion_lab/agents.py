from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .runtime_base import RuntimeAdapter
from .schemas import AttemptTrace, FailureMode, QAExample, ReflectionEntry, RunRecord


@dataclass
class BaseAgent:
    runtime: RuntimeAdapter
    agent_type: Literal["react", "reflexion"]
    max_attempts: int = 1
    adaptive_max_attempts: bool = False

    def _is_unsalvageable(self, reason: str, failure_mode: FailureMode) -> bool:
        reason_l = reason.lower()
        return "unsalvageable" in reason_l or failure_mode == "insufficient_context"

    def run(self, example: QAExample) -> RunRecord:
        reflection_memory: list[str] = []
        reflections: list[ReflectionEntry] = []
        traces: list[AttemptTrace] = []
        trajectory: list[str] = []

        final_answer = ""
        final_score = 0
        final_failure_mode: FailureMode = "none"
        same_failure_streak = 0
        prev_failure_mode: FailureMode = "none"

        for attempt_id in range(1, self.max_attempts + 1):
            actor_call = self.runtime.actor(
                example=example,
                attempt_id=attempt_id,
                reflection_memory=reflection_memory,
                trajectory=trajectory,
            )
            answer = actor_call.content.strip()
            final_answer = answer

            judge, eval_call = self.runtime.evaluator(
                example=example,
                answer=answer,
                attempt_id=attempt_id,
                reflection_memory=reflection_memory,
                trajectory=trajectory,
            )

            trace = AttemptTrace(
                attempt_id=attempt_id,
                answer=answer,
                score=judge.score,
                reason=judge.reason,
                failure_mode=judge.failure_mode,
                token_estimate=actor_call.total_tokens + eval_call.total_tokens,
                latency_ms=actor_call.latency_ms + eval_call.latency_ms,
                cost_usd=actor_call.cost_usd + eval_call.cost_usd,
            )
            trajectory.append(
                f"attempt={attempt_id} answer={answer} score={judge.score} failure_mode={judge.failure_mode}"
            )

            final_score = judge.score
            final_failure_mode = judge.failure_mode

            if judge.score == 1:
                traces.append(trace)
                break

            if judge.failure_mode == prev_failure_mode and judge.failure_mode != "none":
                same_failure_streak += 1
            else:
                same_failure_streak = 1
            prev_failure_mode = judge.failure_mode

            if self.adaptive_max_attempts and self._is_unsalvageable(judge.reason, judge.failure_mode):
                traces.append(trace)
                break
            if self.adaptive_max_attempts and same_failure_streak >= 2:
                traces.append(trace)
                break

            if self.agent_type != "reflexion" or attempt_id >= self.max_attempts:
                traces.append(trace)
                break

            reflection, reflect_call = self.runtime.reflector(
                example=example,
                answer=answer,
                judge=judge,
                attempt_id=attempt_id,
                reflection_memory=reflection_memory,
                trajectory=trajectory,
            )
            reflection_text = f"lesson={reflection.lesson} | strategy={reflection.next_strategy}"

            trace.reflection = reflection
            trace.token_estimate += reflect_call.total_tokens
            trace.latency_ms += reflect_call.latency_ms
            trace.cost_usd += reflect_call.cost_usd
            reflections.append(reflection)

            if self.adaptive_max_attempts and reflection_memory and reflection_text == reflection_memory[-1]:
                traces.append(trace)
                break

            reflection_memory.append(reflection_text)
            traces.append(trace)

        total_tokens = sum(t.token_estimate for t in traces)
        total_latency = sum(t.latency_ms for t in traces)
        total_cost = round(sum(t.cost_usd for t in traces), 10)
        if final_score == 1:
            final_failure_mode = "none"

        return RunRecord(
            qid=example.qid,
            question=example.question,
            gold_answer=example.gold_answer,
            agent_type=self.agent_type,
            predicted_answer=final_answer,
            is_correct=bool(final_score),
            attempts=len(traces),
            token_estimate=total_tokens,
            latency_ms=total_latency,
            cost_usd=total_cost,
            failure_mode=final_failure_mode,
            reflections=reflections,
            traces=traces,
        )


class ReActAgent(BaseAgent):
    def __init__(self, runtime: RuntimeAdapter) -> None:
        super().__init__(
            runtime=runtime,
            agent_type="react",
            max_attempts=1,
            adaptive_max_attempts=False,
        )


class ReflexionAgent(BaseAgent):
    def __init__(self, runtime: RuntimeAdapter, max_attempts: int = 3, adaptive_max_attempts: bool = True) -> None:
        super().__init__(
            runtime=runtime,
            agent_type="reflexion",
            max_attempts=max_attempts,
            adaptive_max_attempts=adaptive_max_attempts,
        )
