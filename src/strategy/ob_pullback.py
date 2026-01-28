from __future__ import annotations
from typing import Dict, Any, Optional
from .base import Strategy, Signal

class OBPullbackStrategy(Strategy):
    """Example strategy for backtesting only.
    ENTER: demand OB exists, not fully mitigated, dist_to_ob_atr <= 0.3,
    positive momentum + trend strength, avoid high volatility spikes.
    EXIT: close < invalidation (demand) OR after hold_days (handled by engine) OR scaled TP (handled by engine).
    """
    def on_day(self, date: str, symbol: str, ctx: Dict[str,Any]) -> Optional[Signal]:
        ob = ctx.get("ob")
        if not ob:
            return None
        if ob.get("kind") != "DEMAND":
            return None
        if ob.get("status") == "fully":
            return None
        if ctx.get("dist_to_ob_atr") is None or ctx.get("dist_to_ob_atr") > 0.3:
            return None
        momentum_60 = ctx.get("momentum_60")
        if momentum_60 is None or momentum_60 <= 0:
            return None
        ma20_slope_atr = ctx.get("ma20_slope_atr")
        if ma20_slope_atr is None or ma20_slope_atr < 0.1:
            return None
        atr_ratio = ctx.get("atr_ratio")
        if atr_ratio is not None and atr_ratio >= 1.5:
            return None
        # ok
        return Signal(symbol=symbol, side="LONG", action="ENTER", reason="OBPullback: demand OB near + momentum/trend")
