from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any

@dataclass
class OrderBlock:
    kind: str  # 'DEMAND' or 'SUPPLY'
    idx: int
    date: str
    lower: float
    upper: float
    invalidation: float
    age: int
    status: str  # unmitigated/partial/fully
    quality: float

def _is_bull(o,c): return c > o
def _is_bear(o,c): return c < o

def detect_ob_from_bos(df: pd.DataFrame, bos: Dict[str,Any], atr: pd.Series, min_push_atr: float, max_age: int) -> Optional[OrderBlock]:
    if not bos or not bos.get("direction"):
        return None
    direction = bos["direction"]
    # Find last opposite-color candle before BOS pivot index (ref pivot idx)
    ref = bos.get("ref_pivot")
    if not ref:
        return None
    pivot_idx = int(ref["idx"])
    o = df["open"].values
    c = df["close"].values
    h = df["high"].values
    l = df["low"].values
    a = float(atr.iloc[pivot_idx]) if atr is not None else 0.0
    if a <= 0:
        return None

    if direction == "BULL":
        # last bearish candle before pivot_idx
        idx = None
        for i in range(pivot_idx, max(-1, pivot_idx-15), -1):
            if _is_bear(o[i], c[i]):
                idx = i; break
        if idx is None:
            return None
        lower = float(l[idx]); upper = float(o[idx])  # conservative [low, open]
        invalidation = lower
        kind = "DEMAND"
        # push quality: next 3~5 bars net move
        j2 = min(len(df)-1, idx+5)
        push = float(df["close"].iloc[j2]) - float(df["close"].iloc[idx])
        quality = (push/(a+1e-9))
    else:
        idx = None
        for i in range(pivot_idx, max(-1, pivot_idx-15), -1):
            if _is_bull(o[i], c[i]):
                idx = i; break
        if idx is None:
            return None
        lower = float(o[idx]); upper = float(h[idx])  # conservative [open, high]
        invalidation = upper
        kind = "SUPPLY"
        j2 = min(len(df)-1, idx+5)
        push = float(df["close"].iloc[idx]) - float(df["close"].iloc[j2])
        quality = (push/(a+1e-9))

    # filter: require push >= min_push_atr
    if quality < float(min_push_atr):
        # keep but low quality; caller can score lower
        pass

    # mitigation status by scanning subsequent candles for entry into zone
    status = "unmitigated"
    partial = 0.0
    last_idx = min(len(df)-1, idx+max_age)
    for j in range(idx+1, last_idx+1):
        lo = float(l[j]); hi = float(h[j])
        if hi < lower or lo > upper:
            continue
        overlap_low = max(lo, lower)
        overlap_high = min(hi, upper)
        depth = max(0.0, overlap_high-overlap_low)
        partial = max(partial, depth/(upper-lower+1e-9))
        if lo <= lower and hi >= upper:
            status = "fully"; break
        else:
            status = "partial"
    age = last_idx - idx
    return OrderBlock(kind, idx, str(df["date"].iloc[idx].date()), lower, upper, invalidation, int(age), status, float(quality))
