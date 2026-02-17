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
