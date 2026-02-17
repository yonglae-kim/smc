from __future__ import annotations

import pandas as pd

from src.strategy.soft_score import SoftScoreStrategy
from src.trading.rules import TradeRules

from .conftest import make_cfg, sample_ctx


def test_signal_valid_from():
    cfg = make_cfg()
    rules = TradeRules(cfg, strategy=SoftScoreStrategy(cfg))
    signal, _ = rules.build_signal("2024-01-02", sample_ctx(), [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")], entry_price=105.0)
    assert signal.timestamp == "2024-01-02"
    assert signal.valid_from == "2024-01-03"


def test_buy_reasons_generated():
    cfg = make_cfg()
    rules = TradeRules(cfg, strategy=SoftScoreStrategy(cfg))
    signal, _ = rules.build_signal("2024-01-02", sample_ctx(), [pd.Timestamp("2024-01-02")], entry_price=105.0)
    assert signal.reasons
