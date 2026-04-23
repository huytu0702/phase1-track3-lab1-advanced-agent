from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from .schemas import ReportPayload, RunRecord


def summarize(records: list[RunRecord]) -> dict:
    grouped: dict[str, list[RunRecord]] = defaultdict(list)
    for record in records:
        grouped[record.agent_type].append(record)

    summary: dict[str, dict] = {}
    for agent_type, rows in grouped.items():
        summary[agent_type] = {
            "count": len(rows),
            "em": round(mean(1.0 if r.is_correct else 0.0 for r in rows), 4),
            "avg_attempts": round(mean(r.attempts for r in rows), 4),
            "avg_token_estimate": round(mean(r.token_estimate for r in rows), 2),
            "avg_latency_ms": round(mean(r.latency_ms for r in rows), 2),
            "avg_cost_usd": round(mean(r.cost_usd for r in rows), 8),
        }

    if "react" in summary and "reflexion" in summary:
        summary["delta_reflexion_minus_react"] = {
            "em_abs": round(summary["reflexion"]["em"] - summary["react"]["em"], 4),
            "attempts_abs": round(
                summary["reflexion"]["avg_attempts"] - summary["react"]["avg_attempts"],
                4,
            ),
            "tokens_abs": round(
                summary["reflexion"]["avg_token_estimate"] - summary["react"]["avg_token_estimate"],
                2,
            ),
            "latency_abs": round(
                summary["reflexion"]["avg_latency_ms"] - summary["react"]["avg_latency_ms"],
                2,
            ),
            "cost_abs_usd": round(
                summary["reflexion"]["avg_cost_usd"] - summary["react"]["avg_cost_usd"],
                8,
            ),
        }
    return summary


def failure_breakdown(records: list[RunRecord]) -> dict:
    by_agent: dict[str, Counter[str]] = defaultdict(Counter)
    overall: Counter[str] = Counter()
    for record in records:
        by_agent[record.agent_type][record.failure_mode] += 1
        overall[record.failure_mode] += 1

    return {
        "global": dict(overall),
        "react": dict(by_agent.get("react", Counter())),
        "reflexion": dict(by_agent.get("reflexion", Counter())),
    }


def _build_discussion(summary: dict, failure_modes: dict, mode: str) -> str:
    react = summary.get("react", {})
    reflexion = summary.get("reflexion", {})
    delta = summary.get("delta_reflexion_minus_react", {})
    global_failures = failure_modes.get("global", {})

    ranked = sorted(
        ((k, v) for k, v in global_failures.items() if k != "none"),
        key=lambda item: item[1],
        reverse=True,
    )
    top_failures = ", ".join(f"{name}={count}" for name, count in ranked[:3]) or "none"

    return (
        f"Mode '{mode}' compares ReAct and Reflexion under the same evaluation policy and dataset. "
        f"Reflexion EM={reflexion.get('em', 0)} while ReAct EM={react.get('em', 0)}, delta={delta.get('em_abs', 0)}. "
        f"The gain is accompanied by higher attempts ({reflexion.get('avg_attempts', 0)} vs {react.get('avg_attempts', 0)}), "
        f"token usage ({reflexion.get('avg_token_estimate', 0)} vs {react.get('avg_token_estimate', 0)}), latency "
        f"({reflexion.get('avg_latency_ms', 0)}ms vs {react.get('avg_latency_ms', 0)}ms), and average cost "
        f"({reflexion.get('avg_cost_usd', 0)} USD vs {react.get('avg_cost_usd', 0)} USD per example). "
        f"Residual failure modes are concentrated in {top_failures}. Reflection memory is useful when the first attempt "
        f"misses a bridge entity or drifts after the first hop. Adaptive stopping limits wasted retries in repeated or "
        f"unsalvageable cases, but hard entity-disambiguation errors still remain and require stronger grounding/evaluator signals."
    )


def build_report(
    records: list[RunRecord],
    dataset_name: str,
    mode: str = "mock",
    extensions: list[str] | None = None,
) -> ReportPayload:
    examples = [
        {
            "qid": r.qid,
            "agent_type": r.agent_type,
            "gold_answer": r.gold_answer,
            "predicted_answer": r.predicted_answer,
            "is_correct": r.is_correct,
            "attempts": r.attempts,
            "failure_mode": r.failure_mode,
            "cost_usd": r.cost_usd,
            "reflection_count": len(r.reflections),
        }
        for r in records
    ]
    summary = summarize(records)
    failure_modes = failure_breakdown(records)
    if extensions is None:
        extensions = ["structured_evaluator", "reflection_memory", "benchmark_report_json"]
        if mode == "mock":
            extensions.append("mock_mode_for_autograding")
    return ReportPayload(
        meta={
            "dataset": dataset_name,
            "mode": mode,
            "num_records": len(records),
            "agents": sorted({r.agent_type for r in records}),
        },
        summary=summary,
        failure_modes=failure_modes,
        examples=examples,
        extensions=extensions,
        discussion=_build_discussion(summary, failure_modes, mode),
    )


def save_report(report: ReportPayload, out_dir: str | Path) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "report.json"
    md_path = out_dir / "report.md"

    json_path.write_text(json.dumps(report.model_dump(), indent=2), encoding="utf-8")
    s = report.summary
    react = s.get("react", {})
    reflexion = s.get("reflexion", {})
    delta = s.get("delta_reflexion_minus_react", {})
    ext_lines = "\n".join(f"- {item}" for item in report.extensions)
    md = f"""# Lab 16 Benchmark Report

## Metadata
- Dataset: {report.meta['dataset']}
- Mode: {report.meta['mode']}
- Records: {report.meta['num_records']}
- Agents: {', '.join(report.meta['agents'])}

## Summary
| Metric | ReAct | Reflexion | Delta |
|---|---:|---:|---:|
| EM | {react.get('em', 0)} | {reflexion.get('em', 0)} | {delta.get('em_abs', 0)} |
| Avg attempts | {react.get('avg_attempts', 0)} | {reflexion.get('avg_attempts', 0)} | {delta.get('attempts_abs', 0)} |
| Avg tokens | {react.get('avg_token_estimate', 0)} | {reflexion.get('avg_token_estimate', 0)} | {delta.get('tokens_abs', 0)} |
| Avg latency (ms) | {react.get('avg_latency_ms', 0)} | {reflexion.get('avg_latency_ms', 0)} | {delta.get('latency_abs', 0)} |
| Avg cost (USD) | {react.get('avg_cost_usd', 0)} | {reflexion.get('avg_cost_usd', 0)} | {delta.get('cost_abs_usd', 0)} |

## Failure modes
```json
{json.dumps(report.failure_modes, indent=2)}
```

## Extensions implemented
{ext_lines}

## Discussion
{report.discussion}
"""
    md_path.write_text(md, encoding="utf-8")
    return json_path, md_path
