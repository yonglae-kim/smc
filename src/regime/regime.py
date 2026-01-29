from __future__ import annotations
from typing import Dict, Any
import pandas as pd
import numpy as np
from ..analysis.indicators import sma, rsi, atr

def compute_regime(
    index_df: pd.DataFrame,
    cfg,
    *,
    ma200: float | None = None,
    rsi14: float | None = None,
    atr14: float | None = None,
) -> Dict[str, Any]:
    df = index_df
    needs_columns = not {"ma200", "rsi14", "atr14"}.issubset(df.columns)
    if needs_columns:
        df = index_df.copy()
        if "ma200" not in df.columns:
            df["ma200"] = sma(df["close"], 200)
        if "rsi14" not in df.columns:
            df["rsi14"] = rsi(df["close"], int(cfg.analysis.rsi_period))
        if "atr14" not in df.columns:
            df["atr14"] = atr(df, int(cfg.analysis.atr_period))

    last = df.iloc[-1]
    ma200 = ma200 if ma200 is not None else (float(last["ma200"]) if not pd.isna(last["ma200"]) else None)
    rsi14 = rsi14 if rsi14 is not None else (float(last["rsi14"]) if not pd.isna(last["rsi14"]) else None)
    atr14 = atr14 if atr14 is not None else (float(last["atr14"]) if not pd.isna(last["atr14"]) else None)

    above_ma200 = (ma200 is not None) and (float(last["close"]) >= ma200)
    rsi_ge_50 = (rsi14 is not None) and (rsi14 >= 50)

    # ATR spike vs recent median
    atr_med = float(df["atr14"].tail(60).median()) if df["atr14"].tail(60).notna().any() else None
    atr_spike = (
        atr14 is not None
        and atr_med is not None
        and atr14 >= float(cfg.symbol_regime.atr_spike_mult) * atr_med
    )

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

def relative_strength(symbol_df: pd.DataFrame, lookback: int) -> Dict[str, Any]:
    s = symbol_df.set_index("date")["close"]
    if len(s) < lookback + 5:
        return {"tag": "RS_UNKNOWN"}
    s = s.tail(lookback + 1)
    sret = (s.iloc[-1] / s.iloc[0]) - 1.0
    tag = "RS_STRONG" if sret > 0.02 else ("RS_WEAK" if sret < -0.02 else "RS_NEUTRAL")
    return {"tag": tag, "symbol_ret": float(sret)}
