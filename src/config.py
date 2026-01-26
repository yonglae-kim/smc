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
    cache_mode: str = "use"  # use | refresh | snapshot
    cache_ttl_sec: float = 0.0
    cache_snapshot_id: str = ""

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
    execution_guide: str = "이 리포트는 19:00 KST 종가 이후 계산되며, 다음 거래일 시가 기준 실행을 가정합니다."
    tp_sl_conflict_note: str = "OHLC만 있을 때 TP/SL 동시 터치 시 보수적(Stop 우선) 가정."

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
    exit_on_score_drop: bool = True
    exit_on_momentum_fade: bool = True
    momentum_exit_days: int = 3
    momentum_exit_rsi_threshold: float = 45.0
    momentum_exit_macd_threshold: float = 0.0
    exit_on_ma20_trend_break: bool = True
    tp_sl_conflict: str = "conservative"  # conservative | optimistic
    trail_atr_mult: float = 0.0
    tp1_risk_reduction: bool = True
    tp1_trail_atr_mult: float = 0.8
    tp1_stop_atr_buffer: float = 0.25


class BacktestCfg(BaseModel):
    start: str = "2022-01-01"
    end: str = "2025-12-31"
    fill_price: str = "next_open"  # next_open | close
    fee_bps: float = 8
    slippage_bps: float = 5
    max_positions: int = 10
    risk_per_trade: float = 0.01
    stop_grace_days: int = 15
    class TpCfg(BaseModel):
        rr_target: float = 2.0
        partial_rr: float = 1.0
        partial_size: float = 0.5
        move_stop_to_entry: bool = True

    tp: TpCfg = TpCfg()
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
    trade: TradeCfg = TradeCfg()
    backtest: BacktestCfg

def load_config(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config.model_validate(raw)
