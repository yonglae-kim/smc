from __future__ import annotations
from typing import Dict, Any, List
import math

def score_candidate(ctx: Dict[str,Any], weights: Dict[str,float]) -> Dict[str,Any]:
    """Explainable additive score. ctx contains tags + distances + regime + rs."""
    s = 0.0
    contrib = []

    def add(key, cond=True, val=None):
        nonlocal s
        if not cond:
            return
        w = float(weights.get(key, 0.0))
        s += w
        contrib.append({"key": key, "w": w, "val": val})

    # market structure
    add("structure_bull", ctx.get("structure_bias")=="BULL")
    add("structure_bear", ctx.get("structure_bias")=="BEAR")
    add("bos_recent", ctx.get("bos",{}).get("direction") is not None)

    # proximity (ATR units)
    d_ob = ctx.get("dist_to_ob_atr")
    d_fvg = ctx.get("dist_to_fvg_atr")
    add("ob_near", d_ob is not None and d_ob <= 1.0, d_ob)
    add("fvg_near", d_fvg is not None and d_fvg <= 1.0, d_fvg)
    add("ob_fvg_confluence", ctx.get("tag_confluence_ob_fvg", False))

    # indicators
    add("above_ma200", ctx.get("above_ma200", False))
    rsi = ctx.get("rsi14")
    add("rsi_neutral", rsi is not None and 40 <= rsi <= 60, rsi)

    # RS + regime
    add("rs_strong", ctx.get("rs",{}).get("tag")=="RS_STRONG")
    reg_tag = ctx.get("regime",{}).get("tag")
    add("regime_tailwind", reg_tag=="TAILWIND")
    add("regime_headwind", reg_tag=="HEADWIND")
    add("atr_spike_risk", ctx.get("regime",{}).get("atr_spike", False))

    ctx["score"] = float(s)
    ctx["score_components"] = contrib
    return ctx
