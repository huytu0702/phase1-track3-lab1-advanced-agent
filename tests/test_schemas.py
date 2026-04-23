from src.reflexion_lab.schemas import JudgeResult, ReflectionEntry


def test_judge_result_schema_validation():
    payload = {
        "score": 1,
        "reason": "Exact match.",
        "missing_evidence": [],
        "spurious_claims": [],
        "failure_mode": "none",
    }
    result = JudgeResult.model_validate(payload)
    assert result.score == 1
    assert result.failure_mode == "none"


def test_reflection_entry_schema_validation():
    payload = {
        "attempt_id": 2,
        "failure_reason": "Second hop missing.",
        "lesson": "Complete both hops.",
        "next_strategy": "Resolve the bridge entity first.",
    }
    result = ReflectionEntry.model_validate(payload)
    assert result.attempt_id == 2
    assert "Resolve" in result.next_strategy
