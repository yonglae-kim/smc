from __future__ import annotations

from src.strategy.soft_score import SoftScoreStrategy
from src.trading.models import Position
from src.trading.rules import TradeRules

from .conftest import make_cfg


def _sample_position() -> Position:
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


def test_tp_sl_conflict_rule():
    cfg = make_cfg()
    rules = TradeRules(cfg, strategy=SoftScoreStrategy(cfg))
    bar = {"open": 100.0, "high": 112.0, "low": 88.0, "close": 100.0}
    decisions = rules.evaluate_exit(_sample_position(), bar, None, "2024-01-03")
    assert any(d.action == "EXIT" and "conservative" in d.reason for d in decisions)

    cfg.trade.tp_sl_conflict = "optimistic"
    rules = TradeRules(cfg, strategy=SoftScoreStrategy(cfg))
    decisions = rules.evaluate_exit(_sample_position(), bar, None, "2024-01-03")
    assert any(d.action == "EXIT" and "optimistic" in d.reason for d in decisions)


def test_stop_grace_allows_take_profit_exit():
    rules = TradeRules(make_cfg(), strategy=SoftScoreStrategy(make_cfg()))
    decisions = rules.evaluate_exit(
        _sample_position(),
        {"open": 100.0, "high": 111.0, "low": 89.0, "close": 105.0},
        None,
        "2024-01-03",
        {"stop_grace_active": True, "allow_non_stop_exits_during_stop_grace": True},
    )
    assert any(d.action == "EXIT" and "익절" in d.reason for d in decisions)


def test_stop_grace_blocks_stop_loss_exit():
    rules = TradeRules(make_cfg(), strategy=SoftScoreStrategy(make_cfg()))
    decisions = rules.evaluate_exit(
        _sample_position(),
        {"open": 100.0, "high": 108.0, "low": 89.0, "close": 104.0},
        None,
        "2024-01-03",
        {"stop_grace_active": True, "allow_non_stop_exits_during_stop_grace": True},
    )
    assert any(d.action == "HOLD" for d in decisions)


def test_exits_behave_normally_after_stop_grace():
    rules = TradeRules(make_cfg(), strategy=SoftScoreStrategy(make_cfg()))
    decisions = rules.evaluate_exit(
        _sample_position(),
        {"open": 100.0, "high": 108.0, "low": 89.0, "close": 104.0},
        None,
        "2024-01-03",
        {"stop_grace_active": False, "allow_non_stop_exits_during_stop_grace": True},
    )
    assert any(d.action == "EXIT" and "손절" in d.reason for d in decisions)
