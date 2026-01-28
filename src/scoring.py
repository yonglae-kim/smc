from __future__ import annotations
from typing import Dict, Any, List
import math

def score_candidate(ctx: Dict[str,Any], weights: Dict[str,float]) -> Dict[str,Any]:
    """Explainable additive score. ctx contains tags + distances + momentum + trend metrics."""
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
    add("above_ma20", ctx.get("above_ma20", False))
    add("ma20_above_ma200", ctx.get("ma20_above_ma200", False))
    rsi = ctx.get("rsi14")
    add("rsi_neutral", rsi is not None and 40 <= rsi <= 60, rsi)
    add("rsi_bullish", rsi is not None and 50 <= rsi <= 70, rsi)
    macd_line = ctx.get("macd_line")
    macd_signal = ctx.get("macd_signal")
    macd_hist = ctx.get("macd_hist")
    add("macd_bullish", macd_hist is not None and macd_hist > 0, macd_hist)
    add("macd_cross", macd_line is not None and macd_signal is not None and macd_line > macd_signal, macd_line)
    volume_ratio = ctx.get("volume_ratio")
    add("volume_surge", volume_ratio is not None and volume_ratio >= 1.3, volume_ratio)

    # momentum + trend strength
    momentum_20 = ctx.get("momentum_20")
    add("momentum_20_positive", momentum_20 is not None and momentum_20 > 0, momentum_20)
    momentum_60 = ctx.get("momentum_60")
    add("momentum_60_positive", momentum_60 is not None and momentum_60 > 0, momentum_60)
    ma20_slope_atr = ctx.get("ma20_slope_atr")
    add("ma20_slope_strong", ma20_slope_atr is not None and ma20_slope_atr >= 0.15, ma20_slope_atr)
    vol_adj_return_20 = ctx.get("vol_adj_return_20")
    add("vol_adj_return_20", vol_adj_return_20 is not None and vol_adj_return_20 >= 1.0, vol_adj_return_20)

    ctx["score"] = float(s)
    ctx["score_components"] = contrib
    return ctx
