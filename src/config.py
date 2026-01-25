from __future__ import annotations
from pydantic import BaseModel
from typing import Dict, Any
import yaml

class AppCfg(BaseModel):
    out_dir: str
    cache_dir: str
    log_level: str = "INFO"

class NetworkCfg(BaseModel):
    timeout_sec: float = 12
    max_retries: int = 4
    backoff_base_sec: float = 0.6
    jitter_sec: float = 0.35
    rate_limit_per_sec: float = 4.0

class UniverseCfg(BaseModel):
    ohlcv_lookback_days: int = 120
    liquidity_window: int = 20
    top_liquidity: int = 500
    daily_recalc_top: int = 800
    weekly_full_scan_weekday: int = 0
    include_daily_value_rank_addon: int = 200

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

class ScoringCfg(BaseModel):
    weights: Dict[str, float]
    top_detail: int = 50

class RegimeCfg(BaseModel):
    index_lookback_days: int = 260
    rs_lookback_days: int = 60
    atr_spike_mult: float = 1.8

class ReportCfg(BaseModel):
    title: str
    max_table_rows: int = 500
    chart_lookback: int = 180
    include_sort_search_js: bool = True


class BacktestCfg(BaseModel):
    start: str = "2022-01-01"
    end: str = "2025-12-31"
    fill_price: str = "next_open"  # next_open | close
    fee_bps: float = 8
    slippage_bps: float = 5
    max_positions: int = 10
    risk_per_trade: float = 0.01
    strategy: str = "ob_pullback"
    symbols: Any = "TOP500"  # "TOP500" or list[str]
    warmup_bars: int = 260
    max_fetch_count: int = 6000

class Config(BaseModel):
    app: AppCfg
    network: NetworkCfg
    universe: UniverseCfg
    analysis: AnalysisCfg
    scoring: ScoringCfg
    regime: RegimeCfg
    report: ReportCfg
    backtest: BacktestCfg

def load_config(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config.model_validate(raw)
