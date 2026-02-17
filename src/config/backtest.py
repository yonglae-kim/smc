from __future__ import annotations

from typing import Any, Dict

from pydantic import BaseModel, Field, field_validator


class TpCfg(BaseModel):
    rr_target: float = 2.0
    partial_rr: float = 1.0
    partial_size: float = 0.5
    move_stop_to_entry: bool = True

    @field_validator("rr_target", "partial_rr")
    @classmethod
    def _non_negative_float(cls, value: float) -> float:
        if value < 0:
            raise ValueError("must be >= 0")
        return value

    @field_validator("partial_size")
    @classmethod
    def _fraction(cls, value: float) -> float:
        if not 0 <= value <= 1:
            raise ValueError("must be between 0 and 1")
        return value


class BacktestCfg(BaseModel):
    start: str = "2022-01-01"
    end: str = "2025-12-31"
    fill_price: str = "next_open"
    fee_bps: float = 8
    slippage_bps: float = 5
    max_positions: int = 10
    risk_per_trade: float = 0.01
    stop_grace_days: int = 15
    allow_non_stop_exits_during_stop_grace: bool = True
    tp: TpCfg = TpCfg()
    strategy: str = "ob_pullback"
    strategy_params: Dict[str, Any] = Field(default_factory=dict)
    symbols: Any = "TOP500"
    warmup_bars: int = 260
    max_fetch_count: int = 6000

    @field_validator("fee_bps", "slippage_bps", "risk_per_trade")
    @classmethod
    def _non_negative_float(cls, value: float) -> float:
        if value < 0:
            raise ValueError("must be >= 0")
        return value

    @field_validator("max_positions", "stop_grace_days", "warmup_bars", "max_fetch_count")
    @classmethod
    def _positive_int(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("must be > 0")
        return value
