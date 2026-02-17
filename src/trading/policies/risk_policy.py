from __future__ import annotations

from ..models import AnalysisContext, EntryPlan, Position, TradeSignal
from .entry_policy import safe_float


class RiskPolicy:
    def __init__(self, cfg):
        trade = getattr(cfg, "trade", None)
        self.min_risk_ratio = float(getattr(trade, "min_risk_ratio", 0.001))
        self.rr_target = float(getattr(trade, "tp_rr_target", 2.0))
        self.partial_rr = float(getattr(trade, "tp_partial_rr", 1.0))
        self.partial_size = float(getattr(trade, "tp_partial_size", 0.0))
        self.max_hold_days = int(getattr(trade, "max_hold_days", 20))
        self.score_exit_threshold = float(getattr(trade, "score_exit_threshold", 0.0))
        self.exit_on_structure_break = bool(getattr(trade, "exit_on_structure_break", True))
        self.exit_on_score_drop = bool(getattr(trade, "exit_on_score_drop", True))
        self.tp_sl_conflict = str(getattr(trade, "tp_sl_conflict", "conservative"))
        self.trail_atr_mult = float(getattr(trade, "trail_atr_mult", 0.0))

    def build_position(
        self,
        signal: TradeSignal,
        entry_plan: EntryPlan,
        entry_date: str,
        entry_price: float,
        size: float,
        ctx: AnalysisContext,
    ) -> Position:
        stop_loss = entry_plan.stop_loss
        if stop_loss >= entry_price:
            stop_loss = entry_price * (1 - self.min_risk_ratio)
        take_profit = entry_plan.take_profit
        if take_profit <= entry_price:
            risk = max(1e-6, entry_price - stop_loss)
            take_profit = entry_price + self.rr_target * risk

        exit_rules = {
            "max_hold_days": self.max_hold_days,
            "score_exit_threshold": self.score_exit_threshold,
            "exit_on_structure_break": self.exit_on_structure_break,
            "exit_on_score_drop": self.exit_on_score_drop,
            "tp_sl_conflict": self.tp_sl_conflict,
        }
        tp1_price = None
        tp1_size = 0.0
        if self.partial_size > 0 and self.partial_rr > 0:
            risk_per_share = max(1e-6, entry_price - entry_plan.stop_loss)
            tp1_price = entry_price + self.partial_rr * risk_per_share
            tp1_size = size * self.partial_size

        atr = safe_float(ctx.atr14, 0.0)
        stop_distance_atr = (entry_price - stop_loss) / atr if atr > 0 else None
        return Position(
            symbol=signal.symbol,
            name=ctx.name,
            market=ctx.market,
            entry_time=entry_date,
            entry_price=entry_price,
            size=size,
            remaining_size=size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            trail=self.trail_atr_mult if self.trail_atr_mult > 0 else None,
            exit_rules=exit_rules,
            state="open",
            entry_score=signal.score,
            entry_breakdown=signal.score_breakdown,
            entry_stop_loss=stop_loss,
            entry_atr=atr if atr > 0 else None,
            entry_structure_bias=ctx.structure_bias,
            stop_distance_atr=stop_distance_atr,
            tp1_price=tp1_price,
            tp1_size=tp1_size,
        )
