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
    assert math.isclose(plan.take_profit, 125.5)


def test_entry_policy_prefers_higher_recovery_candidate_when_ob_and_fvg_coexist():
    cfg = make_cfg()
    policy = EntryPolicy(cfg)
    ctx = sample_ctx()
    ctx["close"] = 106.0
    ctx["ob"] = {"lower": 90.0, "upper": 95.0, "kind": "BULL", "quality": 1.0, "age": 5}
    ctx["fvg"] = {"lower": 102.0, "upper": 104.0, "kind": "BULL", "quality": 2.0, "age": 1}

    plan = policy.build_entry_plan(ctx, entry_price=106.0)

    assert plan.entry_type == "limit_pullback"
    assert math.isclose(plan.entry_price, 104.0)
    assert any("복수 진입 후보" in reason for reason in plan.rationale)


def test_entry_policy_prefers_in_zone_candidate_over_far_reclaim_candidate():
    cfg = make_cfg()
    policy = EntryPolicy(cfg)
    ctx = sample_ctx()
    ctx["close"] = 99.0
    ctx["ob"] = {"lower": 110.0, "upper": 114.0, "kind": "BULL", "quality": 1.0, "age": 3}
    ctx["fvg"] = {"lower": 98.0, "upper": 101.0, "kind": "BULL", "quality": 1.0, "age": 2}

    plan = policy.build_entry_plan(ctx, entry_price=99.0)

    assert plan.entry_type == "limit_in_zone"
    assert math.isclose(plan.entry_price, 99.5)


def test_breakout_candidate_not_forced_when_recovery_is_lower_than_zone_entry():
    cfg = make_cfg()
    cfg.backtest.strategy_params["momentum_breakout_entry_enabled"] = True
    cfg.backtest.strategy_params["momentum_breakout_tick"] = 0.0
    policy = EntryPolicy(cfg)

    ctx = sample_ctx()
    ctx["close"] = 100.0
    ctx["recent_high_20"] = 110.0
    ctx["ob"] = {"lower": 98.0, "upper": 102.0, "kind": "BULL", "quality": 1.0, "age": 1}
    ctx["fvg"] = None

    plan = policy.build_entry_plan(ctx, entry_price=100.0)

    assert plan.entry_type == "limit_in_zone"
    assert math.isclose(plan.entry_price, 100.0)


def test_breakout_candidate_selected_when_recovery_is_high():
    cfg = make_cfg()
    cfg.backtest.strategy_params["momentum_breakout_entry_enabled"] = True
    cfg.backtest.strategy_params["momentum_breakout_tick"] = 0.0
    policy = EntryPolicy(cfg)

    ctx = sample_ctx()
    ctx["close"] = 109.8
    ctx["recent_high_20"] = 110.0
    ctx["momentum_20"] = 0.08
    ctx["momentum_60"] = 0.2
    ctx["ob"] = {"lower": 95.0, "upper": 97.0, "kind": "BULL", "quality": 0.5, "age": 6}
    ctx["fvg"] = None

    plan = policy.build_entry_plan(ctx, entry_price=109.8)

    assert plan.entry_type == "breakout_20"
    assert math.isclose(plan.entry_price, 110.0)
