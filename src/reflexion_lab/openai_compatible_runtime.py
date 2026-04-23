from __future__ import annotations

import json
import random
import re
import time
from time import perf_counter
from typing import Any

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, RateLimitError

from .config import LLMConfig
from .pricing import estimate_cost_usd
from .prompts import (
    ACTOR_SYSTEM,
    EVALUATOR_SYSTEM,
    REFLECTOR_SYSTEM,
    build_actor_user_prompt,
    build_evaluator_user_prompt,
    build_reflector_user_prompt,
)
from .runtime_base import RuntimeAdapter, RuntimeCall
from .schemas import JudgeResult, QAExample, ReflectionEntry


class OpenAICompatibleRuntime(RuntimeAdapter):
    mode_name = "openai-compatible"

    def __init__(
        self,
        config: LLMConfig,
        strict_usage: bool = True,
        max_retries: int = 5,
        backoff_base_seconds: float = 1.0,
    ) -> None:
        self.config = config
        self.strict_usage = strict_usage
        self.max_retries = max_retries
        self.backoff_base_seconds = backoff_base_seconds
        self._clients: dict[tuple[str, str], OpenAI] = {}

    def _client(self, base_url: str, api_key: str) -> OpenAI:
        key = (base_url, api_key)
        if key not in self._clients:
            self._clients[key] = OpenAI(base_url=base_url, api_key=api_key)
        return self._clients[key]

    @staticmethod
    def _extract_content(response: Any) -> str:
        message = response.choices[0].message
        content = message.content
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))
                elif hasattr(item, "text"):
                    parts.append(getattr(item, "text", ""))
            return "\n".join(part for part in parts if part).strip()
        return str(content).strip()

    @staticmethod
    def _extract_cached_prompt_tokens(usage: Any) -> int:
        details = getattr(usage, "prompt_tokens_details", None)
        if details is None:
            return 0
        if isinstance(details, dict):
            return int(details.get("cached_tokens", 0) or 0)
        return int(getattr(details, "cached_tokens", 0) or 0)

    @staticmethod
    def _parse_json_content(content: str) -> dict[str, Any]:
        text = content.strip()
        if text.startswith("```"):
            fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
            if fence_match:
                text = fence_match.group(1).strip()
        if "{" in text and "}" in text and (not text.startswith("{") or not text.endswith("}")):
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                text = text[start : end + 1]
        return json.loads(text)

    def _chat(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: str,
        base_url: str,
        api_key: str,
        json_mode: bool = False,
    ) -> RuntimeCall:
        client = self._client(base_url, api_key)
        started = perf_counter()
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = self._create_with_retry(client=client, kwargs=kwargs)
        elapsed_ms = int((perf_counter() - started) * 1000)
        usage = getattr(response, "usage", None)
        total_tokens = getattr(usage, "total_tokens", None)
        if total_tokens is None:
            if self.strict_usage:
                raise RuntimeError("Real mode requires response.usage.total_tokens from provider.")
            total_tokens = 0
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        cached_prompt_tokens = self._extract_cached_prompt_tokens(usage) if usage is not None else 0
        cost_usd = estimate_cost_usd(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_prompt_tokens=cached_prompt_tokens,
        )
        return RuntimeCall(
            content=self._extract_content(response),
            total_tokens=int(total_tokens),
            latency_ms=elapsed_ms,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_prompt_tokens=cached_prompt_tokens,
            cost_usd=cost_usd,
        )

    def _is_retryable_error(self, exc: Exception) -> bool:
        if isinstance(exc, (RateLimitError, APIConnectionError, APITimeoutError)):
            return True
        if isinstance(exc, APIStatusError):
            return exc.status_code in {408, 409, 429, 500, 502, 503, 504}
        return False

    def _create_with_retry(self, client: OpenAI, kwargs: dict[str, Any]) -> Any:
        attempt = 0
        while True:
            try:
                return client.chat.completions.create(**kwargs)
            except Exception as exc:
                if (not self._is_retryable_error(exc)) or attempt >= self.max_retries:
                    raise
                backoff = self.backoff_base_seconds * (2**attempt)
                jitter = random.uniform(0.0, 0.25)
                time.sleep(backoff + jitter)
                attempt += 1

    def actor(
        self,
        example: QAExample,
        attempt_id: int,
        reflection_memory: list[str],
        trajectory: list[str],
    ) -> RuntimeCall:
        return self._chat(
            system_prompt=ACTOR_SYSTEM,
            user_prompt=build_actor_user_prompt(example, reflection_memory, trajectory),
            model=self.config.default_model,
            base_url=self.config.default_base_url,
            api_key=self.config.default_api_key,
        )

    def evaluator(
        self,
        example: QAExample,
        answer: str,
        attempt_id: int,
        reflection_memory: list[str],
        trajectory: list[str],
    ) -> tuple[JudgeResult, RuntimeCall]:
        call = self._chat(
            system_prompt=EVALUATOR_SYSTEM,
            user_prompt=build_evaluator_user_prompt(example, answer),
            model=self.config.judge_model,
            base_url=self.config.judge_base_url,
            api_key=self.config.judge_api_key,
            json_mode=True,
        )
        try:
            parsed_json = self._parse_json_content(call.content)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Evaluator returned non-JSON content: {call.content!r}") from exc
        return JudgeResult.model_validate(parsed_json), call

    def reflector(
        self,
        example: QAExample,
        answer: str,
        judge: JudgeResult,
        attempt_id: int,
        reflection_memory: list[str],
        trajectory: list[str],
    ) -> tuple[ReflectionEntry, RuntimeCall]:
        call = self._chat(
            system_prompt=REFLECTOR_SYSTEM,
            user_prompt=build_reflector_user_prompt(
                example=example,
                attempt_id=attempt_id,
                predicted_answer=answer,
                judge=judge,
                reflection_memory=reflection_memory,
            ),
            model=self.config.default_model,
            base_url=self.config.default_base_url,
            api_key=self.config.default_api_key,
            json_mode=True,
        )
        try:
            parsed_json = self._parse_json_content(call.content)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Reflector returned non-JSON content: {call.content!r}") from exc
        return ReflectionEntry.model_validate(parsed_json), call
