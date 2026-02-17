from __future__ import annotations

from typing import Dict

from pydantic import BaseModel


class AppCfg(BaseModel):
    out_dir: str
    cache_dir: str
    log_level: str = "INFO"


class UniverseCfg(BaseModel):
    ohlcv_lookback_days: int = 250
    liquidity_window: int = 20
    top_liquidity: int = 0
    daily_recalc_top: int = 800
    weekly_full_scan_weekday: int = 0
    include_daily_value_rank_addon: int = 200


class ScoringCfg(BaseModel):
    weights: Dict[str, float]
    top_detail: int = 50


class SymbolRegimeCfg(BaseModel):
    index_lookback_days: int = 520
    min_regime_bars: int = 260
    rs_lookback_days: int = 60
    atr_spike_mult: float = 1.8


class ReportCfg(BaseModel):
    title: str
    max_table_rows: int = 500
    chart_lookback: int = 180
    include_sort_search_js: bool = True
    chart_image_mode: str = "inline_base64"
    mobile_light_mode: bool = True
    execution_guide: str = "이 리포트는 19:00 KST 종가 이후 계산되며, 다음 거래일 시가 기준 실행을 가정합니다."
    tp_sl_conflict_note: str = "OHLC만 있을 때 TP/SL 동시 터치 시 보수적(Stop 우선) 가정."
