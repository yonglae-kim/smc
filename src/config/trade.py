from __future__ import annotations

from typing import Dict, Literal

from pydantic import BaseModel, Field, field_validator


class TradeCfg(BaseModel):
    execution_delay_days: int = 1
    entry_price_mode: str = "next_open"
    force_top_k: int = 0
    min_score: float = 4.0
    min_expected_return: float = 0.02
    min_rr: float = 1.5
    min_risk_ratio: float = 0.001
    stop_atr_mult: float = 1.5
    tp_rr_target: float = 2.0
    tp_partial_rr: float = 1.0
    tp_partial_size: float = 0.5
    move_stop_to_entry: bool = True
    max_hold_days: int = 20
    score_exit_threshold: float = 2.0
    exit_on_structure_break: bool = True
    structure_break_quality_min: float = 0.0
    exit_on_score_drop: bool = True
    tp_sl_conflict: Literal["conservative", "optimistic"] = "conservative"
    trail_atr_mult: float = 0.0
    early_exit_rsi_macd_enabled: bool = True
    early_exit_rsi_macd_days: int = 3
    early_exit_rsi_threshold: float = 45.0
    early_exit_macd_hist_threshold: float = 0.0
    early_exit_bear_trend_enabled: bool = True
    early_exit_ma20_slope_atr_threshold: float = 0.0
    tp1_risk_reduction_enabled: bool = True
    tp1_stop_atr_buffer: float = 0.25
    tp1_trail_atr_mult: float = 0.0
    min_score_regime_non_tailwind_add: float = 0.0
    min_score_regime_headwind_add: float = 0.0
    entry_type_score_add: Dict[str, float] = Field(default_factory=dict)
    enable_trend_filter: bool = False
    trend_ma_stack: bool = True
    trend_slope_atr_min: float = 0.0
    enable_volatility_filter: bool = False
    max_atr_pct: float = 0.08
    max_atr_ratio: float = 2.2
    enable_volume_confirm: bool = False
    min_volume_ratio: float = 1.2
    enable_rs_rank: bool = False
    rs_rank_min_pct: float = 0.5
    enable_bb_squeeze_breakout: bool = False
    bb_squeeze_max_width: float = 0.12
    min_confirmations: int = 0

    @field_validator("execution_delay_days", "force_top_k", "max_hold_days", "early_exit_rsi_macd_days", "min_confirmations")
    @classmethod
    def _non_negative_int(cls, value: int) -> int:
        if value < 0:
            raise ValueError("must be >= 0")
        return value

    @field_validator(
        "min_expected_return",
        "min_rr",
        "min_risk_ratio",
        "stop_atr_mult",
        "tp_rr_target",
        "tp_partial_rr",
        "trail_atr_mult",
        "tp1_stop_atr_buffer",
        "tp1_trail_atr_mult",
        "min_score_regime_non_tailwind_add",
        "min_score_regime_headwind_add",
        "trend_slope_atr_min",
        "max_atr_pct",
        "max_atr_ratio",
        "min_volume_ratio",
        "rs_rank_min_pct",
        "bb_squeeze_max_width",
    )
    @classmethod
    def _non_negative_float(cls, value: float) -> float:
        if value < 0:
            raise ValueError("must be >= 0")
        return value

    @field_validator("tp_partial_size")
    @classmethod
    def _fraction(cls, value: float) -> float:
        if not 0 <= value <= 1:
            raise ValueError("must be between 0 and 1")
        return value
