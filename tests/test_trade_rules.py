from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.backtest.engine import run_backtest
from src.config import (
    AppCfg,
    AnalysisCfg,
    BacktestCfg,
    Config,
    NetworkCfg,
    RegimeCfg,
    ReportCfg,
    ScoringCfg,
    TradeCfg,
    UniverseCfg,
)
from src.strategy.soft_score import SoftScoreStrategy
from src.trading.models import Position
from src.trading.rules import TradeRules


def make_cfg() -> Config:
    return Config(
        app=AppCfg(out_dir="./out", cache_dir="./cache", log_level="INFO"),
        network=NetworkCfg(),
        universe=UniverseCfg(),
        analysis=AnalysisCfg(),
        scoring=ScoringCfg(weights={}, top_detail=5),
        regime=RegimeCfg(),
        report=ReportCfg(title="Test Report"),
        trade=TradeCfg(
            min_score=0.0,
            min_expected_return=0.0,
            min_rr=0.0,
            score_exit_threshold=0.0,
            exit_on_structure_break=True,
            exit_on_score_drop=True,
            tp_sl_conflict="conservative",
        ),
        backtest=BacktestCfg(
            start="2024-01-01",
            end="2024-06-30",
            strategy_params={
                "threshold": 0.0,
                "require_tailwind": False,
                "require_above_ma200": False,
                "ma_slope_gate": {
                    "enabled": True,
                    "ma_fast": 20,
                    "ma_slow": 200,
                    "slope_window": 5,
                    "buy_slope_threshold": 0.015,
                    "sell_slope_threshold": -0.015,
                    "require_close_confirm_for_buy": True,
                    "require_close_confirm_for_sell": True,
                },
            },
            warmup_bars=60,
            max_fetch_count=500,
        ),
    )


def sample_ctx() -> dict:
    return {
        "symbol": "000000",
        "name": "Sample",
        "market": "KOSPI",
        "close": 105.0,
        "atr14": 4.0,
        "room_to_high_atr": 2.0,
        "dist_to_ob_atr": 0.2,
        "dist_to_fvg_atr": None,
        "ob": {"lower": 95.0, "upper": 100.0, "invalidation": 92.0, "kind": "BULL", "status": "active", "quality": 2.0, "age": 2},
        "fvg": None,
        "fvg_active": False,
        "regime": {"tag": "TAILWIND", "atr_spike": False},
        "rs": {"tag": "RS_STRONG"},
        "structure_bias": "BULL",
        "above_ma200": False,
        "above_ma20": True,
        "ma20_above_ma200": False,
        "ma20": 100.0,
        "ma200": 110.0,
        "ma_slope_fast": 100.0,
        "ma_slope_slow": 110.0,
        "ma_slope_pct": 0.02,
        "rsi14": 55.0,
        "macd_hist": 1.2,
        "macd_line": 1.4,
        "macd_signal": 1.0,
        "volume_ratio": 1.8,
        "momentum_20": 0.05,
        "momentum_60": 0.1,
        "ma20_slope_atr": 0.2,
        "atr_ratio": 1.0,
        "ob_quality": 2.5,
        "ob_age": 5,
        "fvg_age": 0,
        "tag_confluence_ob_fvg": True,
    }


def test_signal_valid_from():
    cfg = make_cfg()
    strategy = SoftScoreStrategy(cfg)
    rules = TradeRules(cfg, strategy=strategy)
    ctx = sample_ctx()
    cal = [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
    signal, _ = rules.build_signal("2024-01-02", ctx, cal, entry_price=105.0)
    assert signal.timestamp == "2024-01-02"
    assert signal.valid_from == "2024-01-03"


def test_buy_reasons_generated():
    cfg = make_cfg()
    rules = TradeRules(cfg, strategy=SoftScoreStrategy(cfg))
    ctx = sample_ctx()
    signal, _ = rules.build_signal("2024-01-02", ctx, [pd.Timestamp("2024-01-02")], entry_price=105.0)
    assert signal.reasons


def test_tp_sl_conflict_rule():
    cfg = make_cfg()
    rules = TradeRules(cfg, strategy=SoftScoreStrategy(cfg))
    pos = Position(
        symbol="000000",
        name="Sample",
        market="KOSPI",
        entry_time="2024-01-02",
        entry_price=100.0,
        size=1.0,
        remaining_size=1.0,
        stop_loss=90.0,
        take_profit=110.0,
        trail=None,
        exit_rules={},
    )
    bar = {"open": 100.0, "high": 112.0, "low": 88.0, "close": 100.0}
    decisions = rules.evaluate_exit(pos, bar, None, "2024-01-03")
    assert any(d.action == "EXIT" and "conservative" in d.reason for d in decisions)

    cfg.trade.tp_sl_conflict = "optimistic"
    rules = TradeRules(cfg, strategy=SoftScoreStrategy(cfg))
    decisions = rules.evaluate_exit(pos, bar, None, "2024-01-03")
    assert any(d.action == "EXIT" and "optimistic" in d.reason for d in decisions)


def test_end_to_end_smoke():
    cfg = make_cfg()
    strategy = SoftScoreStrategy(cfg)
    dates = pd.date_range("2023-08-01", periods=200, freq="D")
    prices = pd.Series(range(100, 100 + len(dates)))
    df = pd.DataFrame(
        {
            "date": dates,
            "open": prices,
            "high": prices + 1,
            "low": prices - 1,
            "close": prices + 0.5,
            "volume": 100000,
        }
    )
    idx_df = df.copy()
    symbols = [{"symbol": "000000", "name": "Sample", "market": "KOSPI"}]
    result = run_backtest(symbols, {"000000": df}, idx_df, idx_df, cfg, strategy)
    assert "equity_curve" in result
    assert result["start"] == "2024-01-01"
