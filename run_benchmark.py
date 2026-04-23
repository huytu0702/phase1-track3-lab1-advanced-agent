from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Literal

import typer
from rich import print

from src.reflexion_lab.agents import ReActAgent, ReflexionAgent
from src.reflexion_lab.config import LLMConfig
from src.reflexion_lab.mock_runtime import MockRuntime
from src.reflexion_lab.openai_compatible_runtime import OpenAICompatibleRuntime
from src.reflexion_lab.reporting import build_report, save_report
from src.reflexion_lab.runtime_base import RuntimeAdapter
from src.reflexion_lab.utils import load_dataset, load_jsonl_records, save_jsonl

app = typer.Typer(add_completion=False)


def _build_runtime(mode: Literal["real", "mock"]) -> RuntimeAdapter:
    if mode == "mock":
        return MockRuntime()
    config = LLMConfig.from_env(strict=True)
    return OpenAICompatibleRuntime(
        config=config,
        strict_usage=True,
        max_retries=5,
        backoff_base_seconds=1.0,
    )


def _batched(items: list, batch_size: int):
    for start in range(0, len(items), batch_size):
        yield (start // batch_size) + 1, items[start : start + batch_size]


def _run_batch_with_retry(
    react: ReActAgent,
    reflexion: ReflexionAgent,
    batch: list,
    retries: int,
) -> tuple[list, list]:
    attempt = 0
    while True:
        try:
            react_batch = [react.run(example) for example in batch]
            reflexion_batch = [reflexion.run(example) for example in batch]
            return react_batch, reflexion_batch
        except Exception:
            if attempt >= retries:
                raise
            sleep_s = 2**attempt
            print(f"[yellow]Batch failed. Retry {attempt + 1}/{retries} in {sleep_s}s[/yellow]")
            time.sleep(sleep_s)
            attempt += 1


@app.command()
def main(
    dataset: str = "data/hotpot_100.json",
    out_dir: str = "outputs/final",
    mode: Literal["real", "mock"] = "real",
    reflexion_attempts: int = 3,
    adaptive_max_attempts: bool = True,
    sample_limit: int = 0,
    batch_size: int = 10,
    batch_retries: int = 5,
    resume: bool = True,
) -> None:
    examples = load_dataset(dataset)
    if sample_limit > 0:
        examples = examples[:sample_limit]
    if batch_size <= 0:
        raise typer.BadParameter("--batch-size must be > 0")
    runtime = _build_runtime(mode=mode)

    react = ReActAgent(runtime=runtime)
    reflexion = ReflexionAgent(
        runtime=runtime,
        max_attempts=reflexion_attempts,
        adaptive_max_attempts=adaptive_max_attempts,
    )

    out_path = Path(out_dir)
    react_path = out_path / "react_runs.jsonl"
    reflexion_path = out_path / "reflexion_runs.jsonl"

    if resume:
        react_records = load_jsonl_records(react_path)
        reflexion_records = load_jsonl_records(reflexion_path)
        processed = min(len(react_records), len(reflexion_records))
        if processed > 0:
            print(f"[cyan]Resuming from {processed} processed examples[/cyan]")
        if len(react_records) != len(reflexion_records):
            print("[yellow]Uneven resume state detected, using common prefix only[/yellow]")
            react_records = react_records[:processed]
            reflexion_records = reflexion_records[:processed]
    else:
        react_records = []
        reflexion_records = []
        processed = 0

    remaining_examples = examples[processed:]
    total_batches = (len(remaining_examples) + batch_size - 1) // batch_size if remaining_examples else 0

    for batch_idx, batch in _batched(remaining_examples, batch_size):
        print(f"[cyan]Batch {batch_idx}/{total_batches}[/cyan] size={len(batch)}")
        react_batch, reflexion_batch = _run_batch_with_retry(
            react=react,
            reflexion=reflexion,
            batch=batch,
            retries=batch_retries,
        )
        react_records.extend(react_batch)
        reflexion_records.extend(reflexion_batch)
        save_jsonl(react_path, react_records)
        save_jsonl(reflexion_path, reflexion_records)

    all_records = react_records + reflexion_records

    extensions = ["structured_evaluator", "reflection_memory", "benchmark_report_json"]
    if adaptive_max_attempts:
        extensions.append("adaptive_max_attempts")
    if mode == "mock":
        extensions.append("mock_mode_for_autograding")

    report = build_report(
        all_records,
        dataset_name=Path(dataset).name,
        mode=runtime.mode_name,
        extensions=extensions,
    )
    json_path, md_path = save_report(report, out_path)
    print(f"[green]Saved[/green] {json_path}")
    print(f"[green]Saved[/green] {md_path}")
    print(json.dumps(report.summary, indent=2))


if __name__ == "__main__":
    app()
