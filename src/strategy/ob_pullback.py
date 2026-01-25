from __future__ import annotations
from typing import Dict, Any, Optional
from .base import Strategy, Signal

class OBPullbackStrategy(Strategy):
    """Example strategy for backtesting only.
    ENTER: demand OB exists, not fully mitigated, dist_to_ob_atr <= 0.3, regime tailwind, not atr spike.
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
        if ctx.get("regime",{}).get("tag") != "TAILWIND":
            return None
        if ctx.get("regime",{}).get("atr_spike"):
            return None
        # ok
        return Signal(symbol=symbol, side="LONG", action="ENTER", reason="OBPullback: demand OB near + tailwind")
