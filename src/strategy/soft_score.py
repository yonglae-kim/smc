from __future__ import annotations
from typing import Dict, Any, Optional, Tuple

from .base import Strategy

def _bucket_dist(dist: Optional[float], levels: list[tuple[float,float]]) -> float:
    if dist is None:
        return 0.0
    for mx, sc in levels:
        if dist <= mx:
            return float(sc)
    return 0.0

class SoftScoreStrategy(Strategy):
    def __init__(self, cfg):
        p = getattr(cfg.backtest, "strategy_params", {}) or {}
        self.threshold = float(p.get("threshold", 3.0))
        self.dist_ob_levels = p.get("dist_ob_levels", [(0.3, 4), (0.6, 2), (1.0, 1)])
        self.dist_fvg_levels = p.get("dist_fvg_levels", [(0.3, 2), (0.6, 1)])
        self.w_confluence = float(p.get("w_confluence", 2.0))
        self.w_regime_tailwind = float(p.get("w_regime_tailwind", 2.0))
        self.w_regime_headwind = float(p.get("w_regime_headwind", -2.0))
        self.w_rs_strong = float(p.get("w_rs_strong", 1.0))
        self.w_rs_weak = float(p.get("w_rs_weak", -1.0))
        self.w_atr_spike = float(p.get("w_atr_spike", -2.0))
        self.w_struct_bull = float(p.get("w_struct_bull", 1.0))
        self.w_struct_bear = float(p.get("w_struct_bear", -1.0))

    def rank(self, date: str, symbol: str, ctx: Dict[str,Any]) -> Optional[Tuple[float,str,Dict[str,Any]]]:
        # Hard Gate
        has_ob = bool(ctx.get("ob"))
        has_fvg = bool(ctx.get("fvg_active"))
        if not (has_ob or has_fvg):
            return None

        ob = ctx.get("ob") or {}
        invalidation = ob.get("invalidation")
        if invalidation is None and ctx.get("atr") is None:
            return None

        breakdown: Dict[str,Any] = {}
        score = 0.0

        s_ob = _bucket_dist(ctx.get("dist_to_ob_atr"), self.dist_ob_levels) if has_ob else 0.0
        s_fvg = _bucket_dist(ctx.get("dist_to_fvg_atr"), self.dist_fvg_levels) if has_fvg else 0.0
        score += s_ob + s_fvg
        breakdown["dist_ob"] = s_ob
        breakdown["dist_fvg"] = s_fvg

        tags = set(ctx.get("tags", []))
        if "Confluence_OB_FVG" in tags:
            score += self.w_confluence
            breakdown["confluence"] = self.w_confluence
        else:
            breakdown["confluence"] = 0.0

        regime = (ctx.get("regime") or {})
        rtag = regime.get("tag")
        if rtag == "TAILWIND":
            score += self.w_regime_tailwind
            breakdown["regime"] = self.w_regime_tailwind
        elif rtag == "HEADWIND":
            score += self.w_regime_headwind
            breakdown["regime"] = self.w_regime_headwind
        else:
            breakdown["regime"] = 0.0

        rs = (ctx.get("rs") or {})
        rst = rs.get("tag")
        if rst == "RS_STRONG":
            score += self.w_rs_strong
            breakdown["rs"] = self.w_rs_strong
        elif rst == "RS_WEAK":
            score += self.w_rs_weak
            breakdown["rs"] = self.w_rs_weak
        else:
            breakdown["rs"] = 0.0

        if regime.get("atr_spike") is True:
            score += self.w_atr_spike
            breakdown["atr_spike"] = self.w_atr_spike
        else:
            breakdown["atr_spike"] = 0.0

        struct = ctx.get("structure") or {}
        st = struct.get("bias")
        if st == "BULL":
            score += self.w_struct_bull
            breakdown["structure"] = self.w_struct_bull
        elif st == "BEAR":
            score += self.w_struct_bear
            breakdown["structure"] = self.w_struct_bear
        else:
            breakdown["structure"] = 0.0

        breakdown["total"] = score
        if score < self.threshold:
            return None

        reason = f"SoftScore>=th({self.threshold}): {score:.1f}"
        return score, reason, breakdown
