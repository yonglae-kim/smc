from __future__ import annotations
from typing import Dict, Any
import pandas as pd
import numpy as np
from ..analysis.indicators import sma, rsi, atr

def compute_regime(index_df: pd.DataFrame, cfg) -> Dict[str, Any]:
    df = index_df.copy()
    df["ma200"] = sma(df["close"], 200)
    df["rsi14"] = rsi(df["close"], int(cfg.rsi_period))
    df["atr14"] = atr(df, int(cfg.atr_period))
    last = df.iloc[-1]
    ma200 = float(last["ma200"]) if not pd.isna(last["ma200"]) else None
    rsi14 = float(last["rsi14"]) if not pd.isna(last["rsi14"]) else None
    atr14 = float(last["atr14"]) if not pd.isna(last["atr14"]) else None

    above_ma200 = (ma200 is not None) and (float(last["close"]) >= ma200)
    rsi_ge_50 = (rsi14 is not None) and (rsi14 >= 50)

    # ATR spike vs recent median
    atr_med = float(df["atr14"].tail(60).median()) if df["atr14"].tail(60).notna().any() else None
    atr_spike = (atr14 is not None and atr_med is not None and atr14 >= float(cfg.atr_spike_mult) * atr_med)

    return {
        "asof": str(last["date"].date()),
        "close": float(last["close"]),
        "above_ma200": bool(above_ma200),
        "rsi_ge_50": bool(rsi_ge_50),
        "atr_spike": bool(atr_spike),
        "ma200": ma200, "rsi14": rsi14, "atr14": atr14,
        "atr_med60": atr_med,
        "tag": "TAILWIND" if (above_ma200 and rsi_ge_50) else ("HEADWIND" if ((not above_ma200) and (not rsi_ge_50)) else "MIXED")
    }

def relative_strength(symbol_df: pd.DataFrame, index_df: pd.DataFrame, lookback: int) -> Dict[str, Any]:
    s = symbol_df.set_index("date")["close"]
    i = index_df.set_index("date")["close"]
    common = s.index.intersection(i.index)
    if len(common) < lookback + 5:
        return {"tag": "RS_UNKNOWN"}
    s = s.loc[common].tail(lookback+1)
    i = i.loc[common].tail(lookback+1)
    sret = (s.iloc[-1]/s.iloc[0]) - 1.0
    iret = (i.iloc[-1]/i.iloc[0]) - 1.0
    diff = sret - iret
    tag = "RS_STRONG" if diff > 0.02 else ("RS_WEAK" if diff < -0.02 else "RS_NEUTRAL")
    return {"tag": tag, "symbol_ret": float(sret), "index_ret": float(iret), "diff": float(diff)}
