from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
import numpy as np

@dataclass
class Pivot:
    idx: int
    date: pd.Timestamp
    kind: str  # 'H' or 'L'
    price: float
    strength: float  # heuristic score

def fractal_pivots(df: pd.DataFrame, n: int, atr: pd.Series) -> list[Pivot]:
    """n-bar fractal pivots with confirmation lag (last n bars unconfirmed)."""
    highs = df["high"].values
    lows = df["low"].values
    piv=[]
    last_confirm = len(df) - 1 - n
    for i in range(n, last_confirm):
        h = highs[i]
        l = lows[i]
        if h == max(highs[i-n:i+n+1]):
            amp = h - min(lows[i-n:i+n+1])
            a = float(atr.iloc[i]) if atr is not None else 0.0
            strength = (amp / (a+1e-9))
            piv.append(Pivot(i, df["date"].iloc[i], "H", float(h), float(strength)))
        if l == min(lows[i-n:i+n+1]):
            amp = max(highs[i-n:i+n+1]) - l
            a = float(atr.iloc[i]) if atr is not None else 0.0
            strength = (amp / (a+1e-9))
            piv.append(Pivot(i, df["date"].iloc[i], "L", float(l), float(strength)))
    piv.sort(key=lambda p: p.idx)
    return piv

def classify_structure(pivots: list[Pivot]) -> list[dict]:
    """Return list of swing points with HH/HL/LH/LL classification (based on last same-kind pivot)."""
    lastH = None
    lastL = None
    out=[]
    for p in pivots:
        if p.kind=="H":
            cls = None
            if lastH is not None:
                cls = "HH" if p.price > lastH.price else "LH"
            lastH = p
        else:
            cls = None
            if lastL is not None:
                cls = "HL" if p.price > lastL.price else "LL"
            lastL = p
        out.append({"idx": p.idx, "date": str(p.date.date()), "kind": p.kind, "price": p.price, "cls": cls, "strength": p.strength})
    return out
