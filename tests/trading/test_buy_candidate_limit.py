from __future__ import annotations

from src.trading.models import EntryPlan, TradeSignal
from src.trading.rules import TradeRules

from .conftest import make_cfg


def _signal(symbol: str, score: float) -> tuple[TradeSignal, EntryPlan]:
    signal = TradeSignal(
        timestamp="2024-01-02",
        valid_from="2024-01-03",
        symbol=symbol,
        direction="BUY",
        score=score,
        confidence=1.0,
        reasons=[],
        gates={"gate": True},
    )
    plan = EntryPlan(
        entry_type="next_open",
        entry_price=100.0,
        stop_loss=95.0,
        take_profit=110.0,
        rr=2.0,
        expected_return=0.05,
        rationale=[],
        invalidation="",
    )
    return signal, plan


def test_select_buy_candidates_is_capped_at_five_by_default():
    cfg = make_cfg()
    rules = TradeRules(cfg)
    signals = [_signal(f"{i:06d}", 100 - i) for i in range(10)]

    selected = rules.select_buy_candidates(signals)

    assert len(selected) == 5
    assert [row[0].symbol for row in selected] == [f"{i:06d}" for i in range(5)]


def test_select_buy_candidates_respects_force_top_k_but_still_caps_total():
    cfg = make_cfg()
    cfg.trade.force_top_k = 10
    rules = TradeRules(cfg)
    signals = [_signal(f"{i:06d}", 100 - i) for i in range(10)]
    # Make top-ranked symbols fail gate so force_top_k path is used.
    for signal, _ in signals[:3]:
        signal.gates = {"gate": False}

    selected = rules.select_buy_candidates(signals)

    assert len(selected) == 5
