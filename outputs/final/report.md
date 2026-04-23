# Lab 16 Benchmark Report

## Metadata
- Dataset: hotpot_100.json
- Mode: openai-compatible
- Records: 200
- Agents: react, reflexion

## Summary
| Metric | ReAct | Reflexion | Delta |
|---|---:|---:|---:|
| EM | 0.9 | 0.95 | 0.05 |
| Avg attempts | 1 | 1.1 | 0.1 |
| Avg tokens | 2421.29 | 2848.4 | 427.11 |
| Avg latency (ms) | 6507.68 | 11338.25 | 4830.57 |
| Avg cost (USD) | 0.00084271 | 0.00102539 | 0.00018268 |

## Failure modes
```json
{
  "global": {
    "none": 185,
    "insufficient_context": 2,
    "wrong_final_answer": 13
  },
  "react": {
    "none": 90,
    "insufficient_context": 1,
    "wrong_final_answer": 9
  },
  "reflexion": {
    "none": 95,
    "insufficient_context": 1,
    "wrong_final_answer": 4
  }
}
```

## Extensions implemented
- structured_evaluator
- reflection_memory
- benchmark_report_json
- adaptive_max_attempts

## Discussion
Mode 'openai-compatible' compares ReAct and Reflexion under the same evaluation policy and dataset. Reflexion EM=0.95 while ReAct EM=0.9, delta=0.05. The gain is accompanied by higher attempts (1.1 vs 1), token usage (2848.4 vs 2421.29), latency (11338.25ms vs 6507.68ms), and average cost (0.00102539 USD vs 0.00084271 USD per example). Residual failure modes are concentrated in wrong_final_answer=13, insufficient_context=2. Reflection memory is useful when the first attempt misses a bridge entity or drifts after the first hop. Adaptive stopping limits wasted retries in repeated or unsalvageable cases, but hard entity-disambiguation errors still remain and require stronger grounding/evaluator signals.
