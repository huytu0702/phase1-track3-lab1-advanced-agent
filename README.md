# Reflexion Agent Lab (OpenAI-Compatible)

This repo benchmarks `react` vs `reflexion` on HotpotQA-style multi-hop QA with:
- strict structured evaluator JSON,
- reflection memory across attempts,
- adaptive early stopping.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Environment contract for real mode

Required:
- `DEFAULT_MODEL`
- `DEFAULT_BASE_URL`
- `DEFAULT_API_KEY`
- `JUDGE_MODEL`

Optional (fallback to `DEFAULT_*` if missing):
- `JUDGE_BASE_URL`
- `JUDGE_API_KEY`

## Build 100-sample dataset from a real HotpotQA dump

```bash
python scripts/prepare_hotpot_100.py --input-path path\to\hotpot_dev_distractor_v1.json --output-path data\hotpot_100.json --sample-size 100 --seed 42
```

## Run benchmark

Real mode (default):

```bash
python run_benchmark.py --dataset data/hotpot_100.json --out-dir outputs/final --mode real --reflexion-attempts 3
```

Warm-up real calls on 1-2 samples before full run:

```bash
python run_benchmark.py --dataset data/hotpot_100.json --out-dir outputs/warmup --mode real --sample-limit 2
```

Mock mode smoke test:

```bash
python run_benchmark.py --dataset data/hotpot_mini.json --out-dir outputs/sample_run --mode mock --reflexion-attempts 3
```

Outputs:
- `react_runs.jsonl`
- `reflexion_runs.jsonl`
- `report.json`
- `report.md`

## Autograde

```bash
python autograde.py --report-path outputs/final/report.json
```

## Notes

- Real mode fails fast if provider response does not contain `usage.total_tokens`.
- Cost is tracked from real usage with model pricing support for `gpt-5.4-mini`:
  - input: `0.75 USD / 1M tokens`
  - cached input: `0.075 USD / 1M tokens`
  - output: `1.25 USD / 1M tokens`
- Final report should be generated from real mode (not mock mode) for submission.
