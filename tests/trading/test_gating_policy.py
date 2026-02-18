from __future__ import annotations

from src.strategy.soft_score import SoftScoreStrategy
from src.trading.rules import TradeRules

from .conftest import make_cfg, sample_ctx


def test_signal_has_gate_fields():
    cfg = make_cfg()
    rules = TradeRules(cfg, strategy=SoftScoreStrategy(cfg))
    signal, _ = rules.build_signal("2024-01-02", sample_ctx(), [], entry_price=105.0)
    assert "score_min" in signal.gates
    assert "min_rr" in signal.gates
    assert "min_expected_return" in signal.gates



def test_reclaim_entry_type_adds_min_score_gate():
    cfg = make_cfg()
    cfg.trade.entry_type_score_add = {"reclaim": 100.0}
    rules = TradeRules(cfg, strategy=SoftScoreStrategy(cfg))
    ctx = sample_ctx()
    ctx["close"] = 90.0  # below OB lower -> reclaim
    signal, entry_plan = rules.build_signal("2024-01-02", ctx, [], entry_price=90.0)
    assert entry_plan.entry_type == "reclaim"
    assert signal.gates["score_min"] is False


def test_headwind_regime_raises_min_score():
    cfg = make_cfg()
    cfg.trade.min_score_regime_headwind_add = 100.0
    rules = TradeRules(cfg, strategy=SoftScoreStrategy(cfg))
    ctx = sample_ctx()
    ctx["symbol_regime"] = {"tag": "HEADWIND", "atr_spike": False}
    signal, _ = rules.build_signal("2024-01-02", ctx, [], entry_price=105.0)
    assert signal.gates["score_min"] is False
