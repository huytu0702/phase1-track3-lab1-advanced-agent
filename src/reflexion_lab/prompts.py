from __future__ import annotations

from .schemas import JudgeResult, QAExample

ACTOR_SYSTEM = """
You are a multi-hop QA assistant.
Rules:
1) Use only the provided context paragraphs.
2) Complete all hops before deciding the final entity.
3) Return only the final short answer as plain text, no explanation.
4) If context is insufficient, return exactly: UNKNOWN.
""".strip()

EVALUATOR_SYSTEM = """
You are a strict QA evaluator.
Given question, gold answer, predicted answer, and context, return one JSON object only.
Required schema:
{
  "score": 0 or 1,
  "reason": "short rationale",
  "missing_evidence": ["..."],
  "spurious_claims": ["..."],
  "failure_mode": "none|entity_drift|incomplete_multi_hop|wrong_final_answer|looping|reflection_overfit|insufficient_context"
}
Rules:
- score = 1 only when predicted answer matches gold answer semantically.
- failure_mode must be "none" when score = 1.
- Keep reason concise and factual.
""".strip()

REFLECTOR_SYSTEM = """
You are a reflection module that improves the next attempt.
Return one JSON object only with this schema:
{
  "attempt_id": integer,
  "failure_reason": "short diagnosis",
  "lesson": "portable lesson under 30 words",
  "next_strategy": "concrete strategy for the next attempt"
}
Rules:
- Focus on why the previous answer failed and how to fix it.
- Do not repeat full chain-of-thought.
- Keep output concise and actionable.
""".strip()


def format_context(example: QAExample) -> str:
    lines: list[str] = []
    for idx, chunk in enumerate(example.context, start=1):
        lines.append(f"[{idx}] Title: {chunk.title}")
        lines.append(f"[{idx}] Text: {chunk.text}")
    return "\n".join(lines)


def build_actor_user_prompt(
    example: QAExample,
    reflection_memory: list[str],
    trajectory: list[str],
) -> str:
    reflection_section = "\n".join(f"- {item}" for item in reflection_memory) or "- (none)"
    trajectory_section = "\n".join(f"- {item}" for item in trajectory[-3:]) or "- (none)"
    return f"""Question:
{example.question}

Context:
{format_context(example)}

Recent trajectory:
{trajectory_section}

Reflection memory:
{reflection_section}

Return final answer only.""".strip()


def build_evaluator_user_prompt(example: QAExample, predicted_answer: str) -> str:
    return f"""Question:
{example.question}

Gold answer:
{example.gold_answer}

Predicted answer:
{predicted_answer}

Context:
{format_context(example)}
""".strip()


def build_reflector_user_prompt(
    example: QAExample,
    attempt_id: int,
    predicted_answer: str,
    judge: JudgeResult,
    reflection_memory: list[str],
) -> str:
    memory_section = "\n".join(f"- {item}" for item in reflection_memory[-5:]) or "- (none)"
    return f"""Attempt id: {attempt_id}
Question: {example.question}
Predicted answer: {predicted_answer}
Judge score: {judge.score}
Judge reason: {judge.reason}
Judge missing evidence: {judge.missing_evidence}
Judge spurious claims: {judge.spurious_claims}
Judge failure mode: {judge.failure_mode}

Existing reflection memory:
{memory_section}
""".strip()
