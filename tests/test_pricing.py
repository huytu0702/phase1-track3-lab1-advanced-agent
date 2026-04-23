from src.reflexion_lab.pricing import estimate_cost_usd


def test_gpt54mini_cost_with_cached_tokens():
    cost = estimate_cost_usd(
        model="gpt-5.4-mini",
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
        cached_prompt_tokens=100_000,
    )
    # 900k uncached input * 0.75 + 100k cached input * 0.075 + 1M output * 1.25
    assert round(cost, 6) == round(1.9325, 6)


def test_unknown_model_cost_is_zero():
    assert estimate_cost_usd("unknown-model", 1000, 1000, 0) == 0.0
