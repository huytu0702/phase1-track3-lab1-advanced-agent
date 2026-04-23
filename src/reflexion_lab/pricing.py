from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPricing:
    input_per_million: float
    cached_input_per_million: float
    output_per_million: float


MODEL_PRICING_USD_PER_MILLION: dict[str, ModelPricing] = {
    "gpt-5.4-mini": ModelPricing(
        input_per_million=0.75,
        cached_input_per_million=0.075,
        output_per_million=1.25,
    )
}


def estimate_cost_usd(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    cached_prompt_tokens: int = 0,
) -> float:
    pricing = MODEL_PRICING_USD_PER_MILLION.get(model)
    if pricing is None:
        return 0.0
    prompt_tokens = max(0, int(prompt_tokens))
    completion_tokens = max(0, int(completion_tokens))
    cached_prompt_tokens = max(0, min(int(cached_prompt_tokens), prompt_tokens))
    non_cached_prompt_tokens = max(0, prompt_tokens - cached_prompt_tokens)
    input_cost = (non_cached_prompt_tokens / 1_000_000) * pricing.input_per_million
    cached_cost = (cached_prompt_tokens / 1_000_000) * pricing.cached_input_per_million
    output_cost = (completion_tokens / 1_000_000) * pricing.output_per_million
    return round(input_cost + cached_cost + output_cost, 10)
