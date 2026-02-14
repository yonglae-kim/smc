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
    ReportCfg,
    ScoringCfg,
    SymbolRegimeCfg,
    TradeCfg,
    UniverseCfg,
    load_config,
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
        symbol_regime=SymbolRegimeCfg(),
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
        "symbol_regime": {"tag": "TAILWIND", "atr_spike": False},
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




def _sample_position_for_exit_tests() -> Position:
    return Position(
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


def test_stop_grace_allows_take_profit_exit():
    cfg = make_cfg()
    rules = TradeRules(cfg, strategy=SoftScoreStrategy(cfg))
    pos = _sample_position_for_exit_tests()

    bar = {"open": 100.0, "high": 111.0, "low": 89.0, "close": 105.0}
    decisions = rules.evaluate_exit(
        pos,
        bar,
        None,
        "2024-01-03",
        {"stop_grace_active": True, "allow_non_stop_exits_during_stop_grace": True},
    )

    assert any(d.action == "EXIT" and "익절" in d.reason for d in decisions)
    assert all("손절" not in d.reason for d in decisions if d.action == "EXIT")


def test_stop_grace_blocks_stop_loss_exit():
    cfg = make_cfg()
    rules = TradeRules(cfg, strategy=SoftScoreStrategy(cfg))
    pos = _sample_position_for_exit_tests()

    bar = {"open": 100.0, "high": 108.0, "low": 89.0, "close": 104.0}
    decisions = rules.evaluate_exit(
        pos,
        bar,
        None,
        "2024-01-03",
        {"stop_grace_active": True, "allow_non_stop_exits_during_stop_grace": True},
    )

    assert any(d.action == "HOLD" for d in decisions)
    assert all("손절" not in d.reason for d in decisions if d.action == "EXIT")


def test_exits_behave_normally_after_stop_grace():
    cfg = make_cfg()
    rules = TradeRules(cfg, strategy=SoftScoreStrategy(cfg))
    pos = _sample_position_for_exit_tests()

    bar = {"open": 100.0, "high": 108.0, "low": 89.0, "close": 104.0}
    decisions = rules.evaluate_exit(
        pos,
        bar,
        None,
        "2024-01-03",
        {"stop_grace_active": False, "allow_non_stop_exits_during_stop_grace": True},
    )

    assert any(d.action == "EXIT" and "손절" in d.reason for d in decisions)

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
    symbols = [{"symbol": "000000", "name": "Sample", "market": "KOSPI"}]
    result = run_backtest(symbols, {"000000": df}, cfg, strategy)
    assert "equity_curve" in result
    assert result["start"] == "2024-01-01"


def test_run_backtest_next_open_fill_does_not_reference_close_branch_vars(monkeypatch):
    cfg = make_cfg()
    cfg.backtest.fill_price = "next_open"
    cfg.trade.entry_price_mode = "next_open"
    strategy = SoftScoreStrategy(cfg)

    dates = pd.date_range("2023-10-01", periods=120, freq="D")
    prices = pd.Series(range(100, 100 + len(dates)), dtype=float)
    df = pd.DataFrame(
        {
            "date": dates,
            "open": prices,
            "high": prices + 1,
            "low": prices - 2,
            "close": prices + 0.5,
            "volume": 100000,
        }
    )

    def fake_analyze_symbol(meta, df_slice, _cfg):
        ctx = sample_ctx().copy()
        ctx["symbol"] = meta["symbol"]
        ctx["name"] = meta.get("name", "")
        ctx["close"] = float(df_slice["close"].iloc[-1])
        return ctx

    monkeypatch.setattr("src.backtest.engine.analyze_symbol", fake_analyze_symbol)
    monkeypatch.setattr("src.backtest.engine.score_candidate", lambda ctx, _: ctx)

    symbols = [{"symbol": "000000", "name": "Sample", "market": "KOSPI"}]
    result = run_backtest(symbols, {"000000": df}, cfg, strategy)

    assert isinstance(result, dict)
    assert "equity_curve" in result


def test_load_config_reads_backtest_strategy_params(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        """
app:
  out_dir: ./out
  cache_dir: ./cache
network: {}
universe: {}
analysis: {}
scoring:
  weights: {}
symbol_regime: {}
report:
  title: Test Report
backtest:
  strategy_params:
    threshold: 0.77
""".strip(),
        encoding="utf-8",
    )

    cfg = load_config(str(cfg_path))

    assert cfg.backtest.strategy_params["threshold"] == 0.77


def test_soft_score_confluence_uses_tag_field_without_tags():
    cfg = make_cfg()
    strategy = SoftScoreStrategy(cfg)
    ctx = sample_ctx().copy()
    ctx.pop("tags", None)
    ctx["tag_confluence_ob_fvg"] = True

    result = strategy.evaluate(ctx)

    assert result["breakdown"]["confluence"] == strategy.w_confluence
    assert result["score"] >= strategy.w_confluence


def test_soft_score_confluence_keeps_tags_compatibility():
    cfg = make_cfg()
    strategy = SoftScoreStrategy(cfg)
    ctx = sample_ctx().copy()
    ctx["tag_confluence_ob_fvg"] = None
    ctx["tags"] = ["Confluence_OB_FVG"]

    result = strategy.evaluate(ctx)

    assert result["breakdown"]["confluence"] == strategy.w_confluence
