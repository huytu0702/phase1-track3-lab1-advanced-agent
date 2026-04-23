from __future__ import annotations

import json
from pathlib import Path

import typer

from src.reflexion_lab.data_prep import build_hotpot_subset

app = typer.Typer(add_completion=False)


@app.command()
def main(
    input_path: str = typer.Option(..., "--input-path"),
    output_path: str = typer.Option("data/hotpot_100.json", "--output-path"),
    sample_size: int = typer.Option(100, "--sample-size"),
    seed: int = typer.Option(42, "--seed"),
    context_limit: int = typer.Option(6, "--context-limit"),
) -> None:
    source = Path(input_path)
    if not source.exists():
        raise typer.BadParameter(f"Input file not found: {source}")
    raw_payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw_payload, list):
        raise typer.BadParameter("Hotpot dump must be a JSON list.")

    subset = build_hotpot_subset(
        records=raw_payload,
        sample_size=sample_size,
        seed=seed,
        context_limit=context_limit,
    )
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(subset, ensure_ascii=False, indent=2), encoding="utf-8")
    typer.echo(f"Wrote {len(subset)} examples to {target}")


if __name__ == "__main__":
    app()
