from __future__ import annotations

from src.reflexion_lab.agents import ReActAgent, ReflexionAgent
from src.reflexion_lab.runtime_base import RuntimeCall
from src.reflexion_lab.schemas import JudgeResult, QAExample, ReflectionEntry


def _example() -> QAExample:
    return QAExample.model_validate(
        {
            "qid": "hpx",
            "difficulty": "medium",
            "question": "Q?",
            "gold_answer": "A2",
            "context": [{"title": "t", "text": "ctx"}],
        }
    )


class ScriptedRuntime:
    mode_name = "mock"

    def __init__(self, actors, judges, reflections=None):
        self.actors = actors
        self.judges = judges
        self.reflections = reflections or []
        self.actor_i = 0
        self.eval_i = 0
        self.refl_i = 0

    def actor(self, example, attempt_id, reflection_memory, trajectory):
        value = self.actors[self.actor_i]
        self.actor_i += 1
        return RuntimeCall(content=value, total_tokens=10, latency_ms=5)

    def evaluator(self, example, answer, attempt_id, reflection_memory, trajectory):
        judge = self.judges[self.eval_i]
        self.eval_i += 1
        return judge, RuntimeCall(content=judge.model_dump_json(), total_tokens=7, latency_ms=3)

    def reflector(self, example, answer, judge, attempt_id, reflection_memory, trajectory):
        reflection = self.reflections[self.refl_i]
        self.refl_i += 1
        return reflection, RuntimeCall(content=reflection.model_dump_json(), total_tokens=6, latency_ms=2)


def test_react_runs_single_attempt():
    runtime = ScriptedRuntime(
        actors=["A2"],
        judges=[
            JudgeResult(
                score=1,
                reason="ok",
                missing_evidence=[],
                spurious_claims=[],
                failure_mode="none",
            )
        ],
    )
    record = ReActAgent(runtime=runtime).run(_example())
    assert record.attempts == 1
    assert record.is_correct is True
    assert len(record.reflections) == 0


def test_reflexion_retries_with_reflection_memory():
    runtime = ScriptedRuntime(
        actors=["A1", "A2"],
        judges=[
            JudgeResult(
                score=0,
                reason="need hop 2",
                missing_evidence=["hop2"],
                spurious_claims=["A1"],
                failure_mode="incomplete_multi_hop",
            ),
            JudgeResult(
                score=1,
                reason="ok",
                missing_evidence=[],
                spurious_claims=[],
                failure_mode="none",
            ),
        ],
        reflections=[
            ReflectionEntry(
                attempt_id=1,
                failure_reason="need hop 2",
                lesson="Complete both hops",
                next_strategy="resolve bridge entity first",
            )
        ],
    )
    agent = ReflexionAgent(runtime=runtime, max_attempts=3, adaptive_max_attempts=True)
    record = agent.run(_example())
    assert record.attempts == 2
    assert record.is_correct is True
    assert len(record.reflections) == 1


def test_adaptive_stops_after_two_same_failure_modes():
    runtime = ScriptedRuntime(
        actors=["A1", "A1b", "A2"],
        judges=[
            JudgeResult(
                score=0,
                reason="bad entity",
                missing_evidence=[],
                spurious_claims=["A1"],
                failure_mode="entity_drift",
            ),
            JudgeResult(
                score=0,
                reason="still bad entity",
                missing_evidence=[],
                spurious_claims=["A1b"],
                failure_mode="entity_drift",
            ),
            JudgeResult(
                score=1,
                reason="ok",
                missing_evidence=[],
                spurious_claims=[],
                failure_mode="none",
            ),
        ],
        reflections=[
            ReflectionEntry(
                attempt_id=1,
                failure_reason="bad entity",
                lesson="ground entity",
                next_strategy="validate second chunk",
            )
        ],
    )
    agent = ReflexionAgent(runtime=runtime, max_attempts=4, adaptive_max_attempts=True)
    record = agent.run(_example())
    assert record.attempts == 2
    assert record.is_correct is False
    assert len(record.reflections) == 1


def test_adaptive_stops_on_unsalvageable_signal():
    runtime = ScriptedRuntime(
        actors=["UNKNOWN", "A2"],
        judges=[
            JudgeResult(
                score=0,
                reason="Low confidence and unsalvageable due to missing context.",
                missing_evidence=["context missing"],
                spurious_claims=[],
                failure_mode="insufficient_context",
            ),
            JudgeResult(
                score=1,
                reason="ok",
                missing_evidence=[],
                spurious_claims=[],
                failure_mode="none",
            ),
        ],
        reflections=[],
    )
    agent = ReflexionAgent(runtime=runtime, max_attempts=3, adaptive_max_attempts=True)
    record = agent.run(_example())
    assert record.attempts == 1
    assert record.is_correct is False
