from __future__ import annotations

from pydantic import BaseModel, field_validator


class AnalysisCfg(BaseModel):
    fractal_n: int = 4
    atr_period: int = 14
    rsi_period: int = 14
    ma_fast: int = 20
    ma_slow: int = 200
    bos_buffer_atr: float = 0.2
    fvg_min_width_atr: float = 0.3
    ob_min_push_atr: float = 1.5
    max_zone_age_bars: int = 120

    @field_validator("fractal_n", "atr_period", "rsi_period", "ma_fast", "ma_slow", "max_zone_age_bars")
    @classmethod
    def _positive_int(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("must be > 0")
        return value

    @field_validator("bos_buffer_atr", "fvg_min_width_atr", "ob_min_push_atr")
    @classmethod
    def _non_negative_float(cls, value: float) -> float:
        if value < 0:
            raise ValueError("must be >= 0")
        return value
