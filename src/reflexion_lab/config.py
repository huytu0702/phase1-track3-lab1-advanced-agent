from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class LLMConfig:
    default_model: str
    default_base_url: str
    default_api_key: str
    judge_model: str
    judge_base_url: str
    judge_api_key: str

    @classmethod
    def from_env(cls, strict: bool = True) -> "LLMConfig":
        load_dotenv()
        required = {
            "DEFAULT_MODEL": os.getenv("DEFAULT_MODEL", "").strip(),
            "DEFAULT_BASE_URL": os.getenv("DEFAULT_BASE_URL", "").strip(),
            "DEFAULT_API_KEY": os.getenv("DEFAULT_API_KEY", "").strip(),
            "JUDGE_MODEL": os.getenv("JUDGE_MODEL", "").strip(),
        }
        missing = [key for key, value in required.items() if not value]
        if strict and missing:
            raise ValueError(f"Missing required env vars for real mode: {', '.join(missing)}")

        default_base_url = required["DEFAULT_BASE_URL"]
        default_api_key = required["DEFAULT_API_KEY"]
        return cls(
            default_model=required["DEFAULT_MODEL"],
            default_base_url=default_base_url,
            default_api_key=default_api_key,
            judge_model=required["JUDGE_MODEL"],
            judge_base_url=os.getenv("JUDGE_BASE_URL", "").strip() or default_base_url,
            judge_api_key=os.getenv("JUDGE_API_KEY", "").strip() or default_api_key,
        )
