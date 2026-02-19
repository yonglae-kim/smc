from __future__ import annotations

import math

from src.strategy.registry import StrategyMeta, build_default_registry
from src.strategy.soft_score import SoftScoreStrategy
from src.trading.policies.entry_policy import EntryPolicy
from src.trading.rules import TradeRules

from .conftest import make_cfg, sample_ctx


def test_strategy_registry_no_duplicates():
    registry = build_default_registry()
    assert registry.register(
        StrategyMeta(
            name="cross_sectional_momentum_trend",
            aliases=("xsmom",),
            indicators=("ret_252", "ret_21", "mom_252_21", "ma200", "atr14"),
            description="dup",
        )
    ) is False


def test_gate_behavior():
    cfg = make_cfg()
    cfg.backtest.strategy_params["cross_momentum_gate_enabled"] = True
    cfg.backtest.strategy_params["cross_momentum_gate_mode"] = "ma200"
    cfg.backtest.strategy_params["low_vol_penalty_enabled"] = True
    cfg.backtest.strategy_params["low_vol_penalty_lambda"] = 1.0

    rules = TradeRules(cfg, strategy=SoftScoreStrategy(cfg))
    ctx = sample_ctx()
    ctx["ma200"] = 100.0
    ctx["close"] = 110.0
    ctx["mom_252_21_z"] = 1.0
    ctx["vol_60_z"] = 1.0
    signal_pass, _ = rules.build_signal("2024-01-02", ctx, [], entry_price=110.0)
    assert signal_pass.gates["cross_momentum_trend"] is True

    ctx_fail = sample_ctx()
    ctx_fail["ma200"] = 120.0
    ctx_fail["close"] = 110.0
    ctx_fail["mom_252_21_z"] = 1.0
    ctx_fail["vol_60_z"] = 3.0
    signal_fail, _ = rules.build_signal("2024-01-02", ctx_fail, [], entry_price=110.0)
    assert signal_fail.gates["cross_momentum_trend"] is False
    assert signal_fail.score < signal_pass.score


def test_entry_stop_target_math():
    cfg = make_cfg()
    cfg.backtest.strategy_params["momentum_breakout_entry_enabled"] = True
    cfg.backtest.strategy_params["momentum_breakout_tick"] = 0.0
    cfg.backtest.strategy_params["momentum_stop_atr_mult"] = 2.5
    cfg.backtest.strategy_params["momentum_rr_target"] = 2.0
    policy = EntryPolicy(cfg)

    ctx = sample_ctx()
    ctx["recent_high_20"] = 100.0
    ctx["close"] = 99.0
    ctx["atr14"] = 4.0
    ctx["ob"] = None
    plan = policy.build_entry_plan(ctx, entry_price=99.0)

    assert math.isclose(plan.entry_price, 100.0)
    assert math.isclose(plan.stop_loss, 90.0)
    assert math.isclose(plan.take_profit, 120.0)
