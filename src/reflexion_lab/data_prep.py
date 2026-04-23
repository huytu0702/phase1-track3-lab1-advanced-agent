from __future__ import annotations

import random
from typing import Any

from .schemas import QAExample


def _difficulty(level: str) -> str:
    level_l = level.lower().strip()
    if level_l in {"easy", "medium", "hard"}:
        return level_l
    return "medium"


def _join_sentences(value: Any) -> str:
    if isinstance(value, list):
        return " ".join(str(item).strip() for item in value if str(item).strip())
    return str(value).strip()


def _extract_supporting_titles(record: dict[str, Any]) -> set[str]:
    supporting = record.get("supporting_facts", [])
    if isinstance(supporting, dict):
        titles = supporting.get("title", [])
        return {str(item).strip() for item in list(titles) if str(item).strip()}
    return {
        str(item[0]).strip()
        for item in supporting
        if isinstance(item, list) and item
    }


def _extract_context_pairs(raw_context: Any) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    if isinstance(raw_context, dict):
        titles = list(raw_context.get("title", []))
        sentences = list(raw_context.get("sentences", []))
        for title, sent_list in zip(titles, sentences):
            text = _join_sentences(list(sent_list))
            title_s = str(title).strip()
            if title_s and text:
                pairs.append((title_s, text))
        return pairs

    if isinstance(raw_context, list):
        for item in raw_context:
            if not isinstance(item, list) or len(item) < 2:
                continue
            title = str(item[0]).strip()
            text = _join_sentences(item[1])
            if title and text:
                pairs.append((title, text))
    return pairs


def convert_hotpot_record(record: dict[str, Any], context_limit: int = 6) -> dict[str, Any]:
    question = str(record.get("question", "")).strip()
    answer = str(record.get("answer", "")).strip()
    qid = str(record.get("_id", "") or record.get("id", "")).strip()
    if not qid:
        raise ValueError("Hotpot record is missing '_id'.")
    if not question or not answer:
        raise ValueError(f"Hotpot record {qid} has empty question/answer.")

    raw_context = record.get("context", [])
    supporting_titles = _extract_supporting_titles(record)
    context_pairs = _extract_context_pairs(raw_context)

    prioritized: list[tuple[str, str]] = []
    remainder: list[tuple[str, str]] = []
    for pair in context_pairs:
        title = pair[0]
        if title in supporting_titles:
            prioritized.append(pair)
        else:
            remainder.append(pair)

    merged = prioritized + remainder
    if not merged:
        raise ValueError(f"Hotpot record {qid} has no usable context chunks.")
    selected = merged[:context_limit]
    return {
        "qid": qid,
        "difficulty": _difficulty(str(record.get("level", "medium"))),
        "question": question,
        "gold_answer": answer,
        "context": [{"title": title, "text": text} for title, text in selected],
    }


def build_hotpot_subset(
    records: list[dict[str, Any]],
    sample_size: int = 100,
    seed: int = 42,
    context_limit: int = 6,
) -> list[dict[str, Any]]:
    if len(records) < sample_size:
        raise ValueError(f"Need at least {sample_size} records, got {len(records)}.")
    rng = random.Random(seed)
    sampled = rng.sample(records, k=sample_size)
    converted = [convert_hotpot_record(item, context_limit=context_limit) for item in sampled]
    for item in converted:
        QAExample.model_validate(item)
    return converted
