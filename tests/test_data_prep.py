from src.reflexion_lab.data_prep import build_hotpot_subset, convert_hotpot_record
from src.reflexion_lab.schemas import QAExample


def _raw_record(i: int) -> dict:
    return {
        "_id": f"id-{i}",
        "question": f"Question {i}?",
        "answer": f"Answer {i}",
        "level": "medium" if i % 2 == 0 else "hard",
        "supporting_facts": [["Title A", 0], ["Title B", 0]],
        "context": [
            ["Title A", [f"A sentence {i}."]],
            ["Title B", [f"B sentence {i}."]],
            ["Title C", [f"C sentence {i}."]],
        ],
    }


def test_convert_hotpot_record_matches_schema():
    item = convert_hotpot_record(_raw_record(1), context_limit=2)
    validated = QAExample.model_validate(item)
    assert validated.qid == "id-1"
    assert len(validated.context) == 2


def test_build_hotpot_subset_returns_exact_100():
    payload = [_raw_record(i) for i in range(130)]
    subset = build_hotpot_subset(payload, sample_size=100, seed=7, context_limit=3)
    assert len(subset) == 100
    assert len({item["qid"] for item in subset}) == 100
