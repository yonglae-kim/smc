from __future__ import annotations
import pandas as pd
from typing import Optional, Dict, Any
from .pivots import Pivot

def latest_significant_pivots(struct_points: list[dict], min_strength: float=1.2) -> dict:
    hs=[p for p in struct_points if p["kind"]=="H" and (p["strength"] or 0)>=min_strength]
    ls=[p for p in struct_points if p["kind"]=="L" and (p["strength"] or 0)>=min_strength]
    return {"lastH": hs[-1] if hs else None, "lastL": ls[-1] if ls else None}

def bos_choch(df: pd.DataFrame, struct_points: list[dict], atr: pd.Series, buffer_atr: float) -> Dict[str, Any]:
    """BOS hint: close breaks last significant pivot by buffer*ATR.
    Returns dict with {direction, level, quality, date, note}
    """
    sig = latest_significant_pivots(struct_points)
    lastH = sig["lastH"]
    lastL = sig["lastL"]
    if len(df) < 5 or atr is None:
        return {"direction": None}
    c = float(df["close"].iloc[-1])
    a = float(atr.iloc[-1])
    buf = buffer_atr * a
    out = {"direction": None}

    if lastH and (c > float(lastH["price"]) + buf):
        level = float(lastH["price"])
        quality = abs(c-level)/(a+1e-9)
        out = {"direction": "BULL", "level": level, "date": str(df["date"].iloc[-1].date()),
               "quality": float(quality), "ref_pivot": lastH}
    elif lastL and (c < float(lastL["price"]) - buf):
        level = float(lastL["price"])
        quality = abs(c-level)/(a+1e-9)
        out = {"direction": "BEAR", "level": level, "date": str(df["date"].iloc[-1].date()),
               "quality": float(quality), "ref_pivot": lastL}
    return out
