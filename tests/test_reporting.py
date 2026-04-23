from src.reflexion_lab.reporting import build_report
from src.reflexion_lab.schemas import RunRecord


def _record(agent_type: str, idx: int, ok: bool, failure_mode: str) -> RunRecord:
    return RunRecord.model_validate(
        {
            "qid": f"q{idx}",
            "question": "Q",
            "gold_answer": "A",
            "agent_type": agent_type,
            "predicted_answer": "A" if ok else "B",
            "is_correct": ok,
            "attempts": 1 if agent_type == "react" else 2,
            "token_estimate": 100 + idx,
            "latency_ms": 50 + idx,
            "failure_mode": "none" if ok else failure_mode,
            "reflections": [],
            "traces": [],
        }
    )


def test_report_contract_has_required_shape():
    records = []
    for i in range(20):
        records.append(_record("react", i, ok=(i % 2 == 0), failure_mode="wrong_final_answer"))
        records.append(
            _record("reflexion", i + 100, ok=(i % 3 != 0), failure_mode="incomplete_multi_hop")
        )
    report = build_report(
        records=records,
        dataset_name="hotpot_100.json",
        mode="openai-compatible",
        extensions=["structured_evaluator", "reflection_memory", "adaptive_max_attempts"],
    )
    payload = report.model_dump()
    assert set(payload.keys()) == {"meta", "summary", "failure_modes", "examples", "extensions", "discussion"}
    assert len(report.examples) >= 20
    assert len(report.failure_modes) >= 3
    assert len(report.discussion) > 250
