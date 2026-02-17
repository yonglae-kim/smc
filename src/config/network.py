from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, field_validator


class NetworkCfg(BaseModel):
    timeout_sec: float = 12
    max_retries: int = 4
    backoff_base_sec: float = 0.6
    jitter_sec: float = 0.35
    rate_limit_per_sec: float = 4.0
    cache_mode: Literal["use", "refresh", "snapshot"] = "use"
    cache_ttl_sec: float = 0.0
    cache_snapshot_id: str = ""

    @field_validator("timeout_sec", "backoff_base_sec", "jitter_sec", "rate_limit_per_sec", "cache_ttl_sec")
    @classmethod
    def _non_negative_float(cls, value: float) -> float:
        if value < 0:
            raise ValueError("must be >= 0")
        return value

    @field_validator("max_retries")
    @classmethod
    def _non_negative_int(cls, value: int) -> int:
        if value < 0:
            raise ValueError("must be >= 0")
        return value
