from __future__ import annotations

import json

import pytest

from src.reflexion_lab.config import LLMConfig
from src.reflexion_lab.openai_compatible_runtime import OpenAICompatibleRuntime
from src.reflexion_lab.schemas import QAExample


class _FakeUsage:
    def __init__(self, total_tokens):
        self.total_tokens = total_tokens
        self.prompt_tokens = 80
        self.completion_tokens = 43
        self.prompt_tokens_details = {"cached_tokens": 20}


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content: str, total_tokens: int | None):
        self.choices = [_FakeChoice(content)]
        self.usage = _FakeUsage(total_tokens)


class _FakeCompletions:
    def __init__(self, with_usage: bool = True, failures_before_success: int = 0, error_type=Exception):
        self.with_usage = with_usage
        self.failures_before_success = failures_before_success
        self.error_type = error_type
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        if self.calls <= self.failures_before_success:
            raise self.error_type("temporary failure")
        system_prompt = kwargs["messages"][0]["content"]
        if "strict QA evaluator" in system_prompt:
            content = json.dumps(
                {
                    "score": 1,
                    "reason": "match",
                    "missing_evidence": [],
                    "spurious_claims": [],
                    "failure_mode": "none",
                }
            )
        elif "reflection module" in system_prompt:
            content = json.dumps(
                {
                    "attempt_id": 1,
                    "failure_reason": "missing second hop",
                    "lesson": "Do both hops",
                    "next_strategy": "Check second paragraph",
                }
            )
        else:
            content = "River Thames"
        total_tokens = 123 if self.with_usage else None
        return _FakeResponse(content=content, total_tokens=total_tokens)


class _FakeChat:
    def __init__(self, with_usage: bool = True, failures_before_success: int = 0, error_type=Exception):
        self.completions = _FakeCompletions(
            with_usage=with_usage,
            failures_before_success=failures_before_success,
            error_type=error_type,
        )


class _FakeOpenAI:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        with_usage: bool = True,
        failures_before_success: int = 0,
        error_type=Exception,
    ):
        self.chat = _FakeChat(
            with_usage=with_usage,
            failures_before_success=failures_before_success,
            error_type=error_type,
        )


def _sample_example() -> QAExample:
    return QAExample.model_validate(
        {
            "qid": "q1",
            "difficulty": "medium",
            "question": "What river flows through London?",
            "gold_answer": "River Thames",
            "context": [{"title": "London", "text": "London is crossed by the River Thames."}],
        }
    )


def test_runtime_parses_usage_and_json(monkeypatch):
    monkeypatch.setattr(
        "src.reflexion_lab.openai_compatible_runtime.OpenAI",
        lambda base_url, api_key: _FakeOpenAI(base_url, api_key, with_usage=True),
    )
    cfg = LLMConfig(
        default_model="gpt-5.4-mini",
        default_base_url="http://localhost:8000/v1",
        default_api_key="key",
        judge_model="gpt-5.4-mini",
        judge_base_url="http://localhost:8000/v1",
        judge_api_key="key",
    )
    runtime = OpenAICompatibleRuntime(cfg, strict_usage=True)
    example = _sample_example()

    actor = runtime.actor(example, attempt_id=1, reflection_memory=[], trajectory=[])
    judge, eval_call = runtime.evaluator(
        example, answer=actor.content, attempt_id=1, reflection_memory=[], trajectory=[]
    )

    assert actor.total_tokens == 123
    assert eval_call.total_tokens == 123
    assert actor.cost_usd > 0
    assert judge.score == 1


def test_runtime_requires_usage_in_real_mode(monkeypatch):
    monkeypatch.setattr(
        "src.reflexion_lab.openai_compatible_runtime.OpenAI",
        lambda base_url, api_key: _FakeOpenAI(base_url, api_key, with_usage=False),
    )
    cfg = LLMConfig(
        default_model="model-a",
        default_base_url="http://localhost:8000/v1",
        default_api_key="key",
        judge_model="model-j",
        judge_base_url="http://localhost:8000/v1",
        judge_api_key="key",
    )
    runtime = OpenAICompatibleRuntime(cfg, strict_usage=True)
    with pytest.raises(RuntimeError, match="usage.total_tokens"):
        runtime.actor(_sample_example(), attempt_id=1, reflection_memory=[], trajectory=[])


class _TransientError(Exception):
    pass


def test_runtime_retries_with_exponential_backoff(monkeypatch):
    fake_client = _FakeOpenAI(
        base_url="http://localhost:8000/v1",
        api_key="key",
        with_usage=True,
        failures_before_success=2,
        error_type=_TransientError,
    )
    monkeypatch.setattr(
        "src.reflexion_lab.openai_compatible_runtime.OpenAI",
        lambda base_url, api_key: fake_client,
    )
    sleeps: list[float] = []
    monkeypatch.setattr("src.reflexion_lab.openai_compatible_runtime.time.sleep", lambda s: sleeps.append(s))

    cfg = LLMConfig(
        default_model="gpt-5.4-mini",
        default_base_url="http://localhost:8000/v1",
        default_api_key="key",
        judge_model="gpt-5.4-mini",
        judge_base_url="http://localhost:8000/v1",
        judge_api_key="key",
    )
    runtime = OpenAICompatibleRuntime(cfg, strict_usage=True, max_retries=5, backoff_base_seconds=1.0)
    monkeypatch.setattr(runtime, "_is_retryable_error", lambda exc: isinstance(exc, _TransientError))

    call = runtime.actor(_sample_example(), attempt_id=1, reflection_memory=[], trajectory=[])
    assert call.total_tokens == 123
    assert fake_client.chat.completions.calls == 3
    assert len(sleeps) == 2
