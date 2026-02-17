from __future__ import annotations

import pandas as pd

from src.backtest.engine import run_backtest
from src.config import load_config
from src.strategy.soft_score import SoftScoreStrategy

from tests.trading.conftest import make_cfg, sample_ctx


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
    df = pd.DataFrame({"date": dates, "open": prices, "high": prices + 1, "low": prices - 2, "close": prices + 0.5, "volume": 100000})

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
