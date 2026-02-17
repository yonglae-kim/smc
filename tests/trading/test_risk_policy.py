from __future__ import annotations

import pandas as pd

from src.strategy.soft_score import SoftScoreStrategy
from src.trading.rules import TradeRules

from .conftest import make_cfg, sample_ctx


def test_build_position_populates_exit_rules():
    cfg = make_cfg()
    rules = TradeRules(cfg, strategy=SoftScoreStrategy(cfg))
    signal, plan = rules.build_signal("2024-01-02", sample_ctx(), [pd.Timestamp("2024-01-03")], entry_price=105.0)
    pos = rules.build_position(signal, plan, "2024-01-03", 105.0, 1.0, sample_ctx())
    assert pos.exit_rules["tp_sl_conflict"] == cfg.trade.tp_sl_conflict
    assert pos.entry_price == 105.0
