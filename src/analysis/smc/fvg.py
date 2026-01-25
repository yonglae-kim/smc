from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
import numpy as np

@dataclass
class FVG:
    kind: str  # 'BULL' or 'BEAR'
    start_idx: int
    end_idx: int
    lower: float
    upper: float
    created_date: str
    age: int
    fill_ratio: float
    status: str  # unfilled/partial/filled

def detect_fvgs(df: pd.DataFrame, atr: pd.Series, min_width_atr: float, max_age: int) -> list[FVG]:
    highs = df["high"].values
    lows = df["low"].values
    fvgs=[]
    for i in range(1, len(df)-1):
        # i is middle candle
        a = float(atr.iloc[i]) if atr is not None else 0.0
        if a <= 0:
            continue
        # Bullish: low[i+1] > high[i-1]
        if lows[i+1] > highs[i-1]:
            lower = float(highs[i-1]); upper = float(lows[i+1])
            if (upper-lower) >= (min_width_atr*a):
                fvgs.append(FVG("BULL", i-1, i+1, lower, upper, str(df["date"].iloc[i].date()), 0, 0.0, "unfilled"))
        # Bearish: high[i+1] < low[i-1]
        if highs[i+1] < lows[i-1]:
            lower = float(highs[i+1]); upper = float(lows[i-1])
            if (upper-lower) >= (min_width_atr*a):
                fvgs.append(FVG("BEAR", i-1, i+1, lower, upper, str(df["date"].iloc[i].date()), 0, 0.0, "unfilled"))

    # Track fill status forward (simple): compare subsequent wicks crossing into zone.
    out=[]
    for z in fvgs:
        created = z.end_idx
        lower, upper = z.lower, z.upper
        filled = False
        partial = 0.0
        last_idx = min(len(df)-1, created+max_age)
        for j in range(created+1, last_idx+1):
            lo = float(df["low"].iloc[j]); hi = float(df["high"].iloc[j])
            # overlap depth
            if hi < lower or lo > upper:
                pass
            else:
                overlap_low = max(lo, lower)
                overlap_high = min(hi, upper)
                depth = max(0.0, overlap_high-overlap_low)
                partial = max(partial, depth/(upper-lower+1e-9))
                if lo <= lower and hi >= upper:
                    filled = True
                    created = created
                    age = j - z.end_idx
                    out.append(FVG(z.kind, z.start_idx, z.end_idx, lower, upper, z.created_date, age, 1.0, "filled"))
                    break
        if not filled:
            age = last_idx - z.end_idx
            status = "partial" if partial > 0 else "unfilled"
            out.append(FVG(z.kind, z.start_idx, z.end_idx, lower, upper, z.created_date, age, float(partial), status))
    return out
