from __future__ import annotations
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

from .analysis.indicators import sma, rsi, atr, macd
from .analysis.smc.pivots import fractal_pivots, classify_structure
from .analysis.smc.structure import bos_choch
from .analysis.smc.fvg import detect_fvgs
from .analysis.smc.ob import detect_ob_from_bos
from .regime.regime import relative_strength
from .scoring import score_candidate

def analyze_symbol(symbol_meta: Dict[str,Any], df: pd.DataFrame, index_df: pd.DataFrame, cfg) -> Optional[Dict[str,Any]]:
    """Compute features + SMC context. Returns context dict suitable for scoring/report."""
    if df is None or len(df) < 60:
        return None
    df = df.sort_values("date").reset_index(drop=True).copy()

    # indicators
    df["ma20"] = sma(df["close"], int(cfg.analysis.ma_fast))
    df["ma200"] = sma(df["close"], int(cfg.analysis.ma_slow))
    df["rsi14"] = rsi(df["close"], int(cfg.analysis.rsi_period))
    df["atr14"] = atr(df, int(cfg.analysis.atr_period))
    macd_line, macd_signal, macd_hist = macd(df["close"])
    df["macd_line"] = macd_line
    df["macd_signal"] = macd_signal
    df["macd_hist"] = macd_hist
    if "volume" in df.columns:
        df["vol_sma20"] = df["volume"].rolling(20, min_periods=20).mean()

    last = df.iloc[-1]
    atr_last = float(last["atr14"]) if not pd.isna(last["atr14"]) else None

    piv = fractal_pivots(df, int(cfg.analysis.fractal_n), df["atr14"])
    struct_pts = classify_structure(piv)
    bos = bos_choch(df, struct_pts, df["atr14"], float(cfg.analysis.bos_buffer_atr))
    ob = detect_ob_from_bos(df, bos, df["atr14"], float(cfg.analysis.ob_min_push_atr), int(cfg.analysis.max_zone_age_bars))
    fvgs = detect_fvgs(df, df["atr14"], float(cfg.analysis.fvg_min_width_atr), int(cfg.analysis.max_zone_age_bars))

    # pick most relevant zones (latest unfilled/partial)
    fvg_pick = None
    for z in reversed(fvgs):
        if z.status in ("unfilled","partial"):
            fvg_pick = z; break

    close = float(last["close"])
    ma20 = float(last["ma20"]) if not pd.isna(last["ma20"]) else None
    ma200 = float(last["ma200"]) if not pd.isna(last["ma200"]) else None
    above_ma200 = (ma200 is not None) and (close >= ma200)
    above_ma20 = (ma20 is not None) and (close >= ma20)
    ma20_above_ma200 = (ma20 is not None and ma200 is not None) and (ma20 >= ma200)
    rsi_last = float(last["rsi14"]) if not pd.isna(last["rsi14"]) else None
    macd_line_last = float(last["macd_line"]) if not pd.isna(last["macd_line"]) else None
    macd_signal_last = float(last["macd_signal"]) if not pd.isna(last["macd_signal"]) else None
    macd_hist_last = float(last["macd_hist"]) if not pd.isna(last["macd_hist"]) else None
    volume_last = float(last["volume"]) if "volume" in last and not pd.isna(last["volume"]) else None
    vol_sma20_last = float(last["vol_sma20"]) if "vol_sma20" in last and not pd.isna(last["vol_sma20"]) else None
    volume_ratio = None
    if volume_last is not None and vol_sma20_last:
        volume_ratio = float(volume_last / (vol_sma20_last + 1e-9))

    # structure bias heuristic from last classified pivots
    bias = "NEUTRAL"
    last_cls = [p["cls"] for p in struct_pts if p["cls"]][-6:]
    if last_cls.count("HH") + last_cls.count("HL") >= 3:
        bias = "BULL"
    elif last_cls.count("LL") + last_cls.count("LH") >= 3:
        bias = "BEAR"

    # distances in ATR
    dist_to_ob = None
    invalidation = None
    ob_status = None
    if ob and atr_last and atr_last>0:
        # distance to nearest edge of zone
        if close < ob.lower:
            dist = (ob.lower - close)
        elif close > ob.upper:
            dist = (close - ob.upper)
        else:
            dist = 0.0
        dist_to_ob = float(dist/(atr_last+1e-9))
        invalidation = float(ob.invalidation)
        ob_status = ob.status

    dist_to_fvg = None
    fvg_status = None
    if fvg_pick and atr_last and atr_last>0:
        lower, upper = fvg_pick.lower, fvg_pick.upper
        if close < lower:
            dist = (lower - close)
        elif close > upper:
            dist = (close - upper)
        else:
            dist = 0.0
        dist_to_fvg = float(dist/(atr_last+1e-9))
        fvg_status = fvg_pick.status

    # Confluence: OB and FVG overlap/near
    confluence = False
    if ob and fvg_pick:
        a1, b1 = ob.lower, ob.upper
        a2, b2 = fvg_pick.lower, fvg_pick.upper
        overlap = not (b1 < a2 or b2 < a1)
        near = (abs(a1-b2) <= (atr_last or 0)*0.5) or (abs(a2-b1) <= (atr_last or 0)*0.5)
        confluence = bool(overlap or near)


    if index_df is None or len(index_df)==0:

        rs = {"tag": "UNKNOWN", "rs": None, "sym_ret": None, "idx_ret": None}

    else:

        rs = relative_strength(df, index_df, int(cfg.regime.rs_lookback_days))

    ctx = {
        "symbol": symbol_meta["symbol"],
        "name": symbol_meta.get("name",""),
        "market": symbol_meta.get("market",""),
        "asof": str(last["date"].date()),
        "close": close,
        "atr14": atr_last,
        "ma20": ma20,
        "ma200": ma200,
        "above_ma200": above_ma200,
        "above_ma20": above_ma20,
        "ma20_above_ma200": ma20_above_ma200,
        "rsi14": rsi_last,
        "macd_line": macd_line_last,
        "macd_signal": macd_signal_last,
        "macd_hist": macd_hist_last,
        "volume": volume_last,
        "volume_sma20": vol_sma20_last,
        "volume_ratio": volume_ratio,
        "structure_bias": bias,
        "bos": bos,
        "ob": None if ob is None else {
            "kind": ob.kind, "date": ob.date, "lower": ob.lower, "upper": ob.upper,
            "invalidation": ob.invalidation, "status": ob.status, "quality": ob.quality
        },
        "fvg": None if fvg_pick is None else {
            "kind": fvg_pick.kind, "created_date": fvg_pick.created_date, "lower": fvg_pick.lower,
            "upper": fvg_pick.upper, "status": fvg_pick.status, "fill_ratio": fvg_pick.fill_ratio, "age": fvg_pick.age
        },
        "fvg_active": fvg_pick is not None,
        "dist_to_ob_atr": dist_to_ob,
        "dist_to_fvg_atr": dist_to_fvg,
        "tag_confluence_ob_fvg": confluence,
        "rs": rs,
        "notes": [
            f"Fractal pivot confirmation lag: last {cfg.analysis.fractal_n} bars are unconfirmed for pivots."
        ]
    }
    # scoring is done after regime injection (runner)
    return ctx
