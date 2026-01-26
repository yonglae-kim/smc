from __future__ import annotations
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

from .analysis.indicators import sma, rsi, atr, macd, bollinger_bands
from .analysis.smc.pivots import fractal_pivots, classify_structure
from .analysis.smc.structure import bos_choch
from .analysis.smc.fvg import detect_fvgs
from .analysis.smc.ob import detect_ob_from_bos
from .regime.regime import relative_strength
from .scoring import score_candidate
from .signals.ma_slope_gate import compute_ma_slope_metrics, evaluate_ma_slope_gate, normalize_ma_slope_gate_config

def analyze_symbol(symbol_meta: Dict[str,Any], df: pd.DataFrame, index_df: pd.DataFrame, cfg) -> Optional[Dict[str,Any]]:
    """Compute features + SMC context. Returns context dict suitable for scoring/report."""
    if df is None or len(df) < 60:
        return None
    df = df.sort_values("date").reset_index(drop=True).copy()

    # indicators
    df["ma20"] = sma(df["close"], int(cfg.analysis.ma_fast))
    df["ma200"] = sma(df["close"], int(cfg.analysis.ma_slow))
    df["ma60"] = sma(df["close"], int(getattr(cfg.analysis, "ma_mid", 60)))
    df["ma120"] = sma(df["close"], int(getattr(cfg.analysis, "ma_long", 120)))
    df["rsi14"] = rsi(df["close"], int(cfg.analysis.rsi_period))
    df["atr14"] = atr(df, int(cfg.analysis.atr_period))
    df["atr50"] = atr(df, 50)
    macd_line, macd_signal, macd_hist = macd(df["close"])
    df["macd_line"] = macd_line
    df["macd_signal"] = macd_signal
    df["macd_hist"] = macd_hist
    if "volume" in df.columns:
        df["vol_sma20"] = df["volume"].rolling(20, min_periods=20).mean()
    df["momentum_20"] = df["close"].pct_change(20)
    df["momentum_60"] = df["close"].pct_change(60)
    df["ma20_slope_atr"] = (df["ma20"] - df["ma20"].shift(10)) / (df["atr14"] + 1e-9)
    df["ma120_slope_atr"] = (df["ma120"] - df["ma120"].shift(20)) / (df["atr14"] + 1e-9)
    df["recent_high_20"] = df["high"].rolling(20, min_periods=20).max()
    bb_mid, bb_upper, bb_lower, bb_width = bollinger_bands(
        df["close"],
        int(getattr(cfg.analysis, "bb_period", 20)),
        float(getattr(cfg.analysis, "bb_k", 2.0)),
    )
    df["bb_mid"] = bb_mid
    df["bb_upper"] = bb_upper
    df["bb_lower"] = bb_lower
    df["bb_width"] = bb_width

    last = df.iloc[-1]
    atr_last = float(last["atr14"]) if not pd.isna(last["atr14"]) else None
    atr50_last = float(last["atr50"]) if not pd.isna(last["atr50"]) else None

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
    ma60 = float(last["ma60"]) if not pd.isna(last["ma60"]) else None
    ma120 = float(last["ma120"]) if not pd.isna(last["ma120"]) else None
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
    momentum_20 = float(last["momentum_20"]) if not pd.isna(last["momentum_20"]) else None
    momentum_60 = float(last["momentum_60"]) if not pd.isna(last["momentum_60"]) else None
    ma20_slope_atr = float(last["ma20_slope_atr"]) if not pd.isna(last["ma20_slope_atr"]) else None
    ma120_slope_atr = float(last["ma120_slope_atr"]) if not pd.isna(last["ma120_slope_atr"]) else None
    gate_cfg = normalize_ma_slope_gate_config(
        (getattr(cfg.backtest, "strategy_params", {}) or {}).get("ma_slope_gate", {})
    )
    ma_gate_metrics = compute_ma_slope_metrics(
        df["close"],
        ma_fast=int(gate_cfg["ma_fast"]),
        ma_slow=int(gate_cfg["ma_slow"]),
        slope_window=int(gate_cfg["slope_window"]),
    )
    buy_pass, buy_reasons, buy_metrics = evaluate_ma_slope_gate(
        ma_gate_metrics,
        "buy",
        buy_slope_threshold=float(gate_cfg["buy_slope_threshold"]),
        sell_slope_threshold=float(gate_cfg["sell_slope_threshold"]),
        require_close_confirm_for_buy=bool(gate_cfg["require_close_confirm_for_buy"]),
        require_close_confirm_for_sell=bool(gate_cfg["require_close_confirm_for_sell"]),
    )
    sell_pass, sell_reasons, sell_metrics = evaluate_ma_slope_gate(
        ma_gate_metrics,
        "sell",
        buy_slope_threshold=float(gate_cfg["buy_slope_threshold"]),
        sell_slope_threshold=float(gate_cfg["sell_slope_threshold"]),
        require_close_confirm_for_buy=bool(gate_cfg["require_close_confirm_for_buy"]),
        require_close_confirm_for_sell=bool(gate_cfg["require_close_confirm_for_sell"]),
    )
    recent_high_20 = float(last["recent_high_20"]) if not pd.isna(last["recent_high_20"]) else None
    bb_width_last = float(last["bb_width"]) if not pd.isna(last["bb_width"]) else None
    bb_upper_last = float(last["bb_upper"]) if not pd.isna(last["bb_upper"]) else None
    bb_lower_last = float(last["bb_lower"]) if not pd.isna(last["bb_lower"]) else None
    squeeze_lookback = int(getattr(cfg.analysis, "bb_squeeze_lookback", 20))
    bb_width_min = None
    if "bb_width" in df.columns and df["bb_width"].notna().any():
        bb_width_min = float(df["bb_width"].tail(squeeze_lookback).min())
    room_to_high_atr = None
    if atr_last and recent_high_20 is not None:
        room_to_high_atr = float((recent_high_20 - close) / (atr_last + 1e-9))
    atr_ratio = None
    if atr_last and atr50_last:
        atr_ratio = float(atr_last / (atr50_last + 1e-9))
    atr_pct = None
    if atr_last and close:
        atr_pct = float(atr_last / (close + 1e-9))

    # structure bias heuristic from last classified pivots
    bias = "NEUTRAL"
    last_cls = [p["cls"] for p in struct_pts if p["cls"]][-6:]
    if last_cls.count("HH") + last_cls.count("HL") >= 3:
        bias = "BULL"
    elif last_cls.count("LL") + last_cls.count("LH") >= 3:
        bias = "BEAR"

    # distances in ATR
    dist_to_ob = None
    ob_quality = None
    ob_age = None
    if ob and atr_last and atr_last>0:
        # distance to nearest edge of zone
        if close < ob.lower:
            dist = (ob.lower - close)
        elif close > ob.upper:
            dist = (close - ob.upper)
        else:
            dist = 0.0
        dist_to_ob = float(dist/(atr_last+1e-9))
        ob_quality = float(ob.quality)
        ob_age = int(ob.age)

    dist_to_fvg = None
    fvg_age = None
    if fvg_pick and atr_last and atr_last>0:
        lower, upper = fvg_pick.lower, fvg_pick.upper
        if close < lower:
            dist = (lower - close)
        elif close > upper:
            dist = (close - upper)
        else:
            dist = 0.0
        dist_to_fvg = float(dist/(atr_last+1e-9))
        fvg_age = int(fvg_pick.age)

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
        "atr50": atr50_last,
        "atr_ratio": atr_ratio,
        "atr_pct": atr_pct,
        "ma20": ma20,
        "ma60": ma60,
        "ma120": ma120,
        "ma200": ma200,
        "above_ma200": above_ma200,
        "above_ma20": above_ma20,
        "ma20_above_ma200": ma20_above_ma200,
        "ma_slope_pct": buy_metrics.get("slope_pct"),
        "ma_slope_fast": buy_metrics.get("ma_fast"),
        "ma_slope_slow": buy_metrics.get("ma_slow"),
        "rsi14": rsi_last,
        "macd_line": macd_line_last,
        "macd_signal": macd_signal_last,
        "macd_hist": macd_hist_last,
        "volume": volume_last,
        "volume_sma20": vol_sma20_last,
        "volume_ratio": volume_ratio,
        "momentum_20": momentum_20,
        "momentum_60": momentum_60,
        "ma20_slope_atr": ma20_slope_atr,
        "ma120_slope_atr": ma120_slope_atr,
        "room_to_high_atr": room_to_high_atr,
        "recent_high_20": recent_high_20,
        "bb_upper": bb_upper_last,
        "bb_lower": bb_lower_last,
        "bb_width": bb_width_last,
        "bb_width_min": bb_width_min,
        "structure_bias": bias,
        "bos": bos,
        "ob": None if ob is None else {
            "kind": ob.kind, "date": ob.date, "lower": ob.lower, "upper": ob.upper,
            "invalidation": ob.invalidation, "status": ob.status, "quality": ob.quality, "age": ob.age
        },
        "ob_quality": ob_quality,
        "ob_age": ob_age,
        "fvg": None if fvg_pick is None else {
            "kind": fvg_pick.kind, "created_date": fvg_pick.created_date, "lower": fvg_pick.lower,
            "upper": fvg_pick.upper, "status": fvg_pick.status, "fill_ratio": fvg_pick.fill_ratio, "age": fvg_pick.age
        },
        "fvg_active": fvg_pick is not None,
        "fvg_age": fvg_age,
        "dist_to_ob_atr": dist_to_ob,
        "dist_to_fvg_atr": dist_to_fvg,
        "tag_confluence_ob_fvg": confluence,
        "ma_slope_gate": {
            "enabled": bool(gate_cfg.get("enabled", True)),
            "buy_pass": bool(buy_pass),
            "sell_pass": bool(sell_pass),
            "buy_reasons": buy_reasons,
            "sell_reasons": sell_reasons,
            "buy_metrics": buy_metrics,
            "sell_metrics": sell_metrics,
            "params": gate_cfg,
        },
        "rs": rs,
        "pivots": [{"idx": p.idx, "date": str(p.date.date()), "kind": p.kind, "price": p.price, "strength": p.strength} for p in piv[-40:]],
        "structure_points": struct_pts[-20:],
        "notes": [
            f"Fractal pivot confirmation lag: last {cfg.analysis.fractal_n} bars are unconfirmed for pivots."
        ]
    }
    # scoring is done after regime injection (runner)
    return ctx
