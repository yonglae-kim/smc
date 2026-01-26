from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


DEFAULT_MA_SLOPE_GATE: Dict[str, Any] = {
    "enabled": True,
    "ma_fast": 20,
    "ma_slow": 200,
    "slope_window": 5,
    "buy_slope_threshold": 0.015,
    "sell_slope_threshold": -0.015,
    "require_close_confirm_for_buy": True,
    "require_close_confirm_for_sell": True,
}


def normalize_ma_slope_gate_config(cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    merged = dict(DEFAULT_MA_SLOPE_GATE)
    if cfg:
        merged.update(cfg)
    return merged


def compute_ma_slope_metrics(
    close: pd.Series,
    ma_fast: int = 20,
    ma_slow: int = 200,
    slope_window: int = 5,
) -> pd.DataFrame:
    close_series = close.astype(float)
    if not close_series.index.is_monotonic_increasing:
        close_series = close_series.sort_index()
    ma_fast_series = close_series.rolling(ma_fast, min_periods=ma_fast).mean()
    ma_slow_series = close_series.rolling(ma_slow, min_periods=ma_slow).mean()
    ma_fast_slope = ma_fast_series / ma_fast_series.shift(slope_window) - 1.0
    ma_slow_slope = ma_slow_series / ma_slow_series.shift(slope_window) - 1.0
    slope_pct = ma_fast_slope - ma_slow_slope
    return pd.DataFrame(
        {
            "close": close_series,
            "ma_fast": ma_fast_series,
            "ma_slow": ma_slow_series,
            "ma_fast_slope": ma_fast_slope,
            "ma_slow_slope": ma_slow_slope,
            "slope_pct": slope_pct,
        }
    )


def evaluate_ma_slope_gate(
    metrics: pd.DataFrame,
    side: str,
    *,
    buy_slope_threshold: float = DEFAULT_MA_SLOPE_GATE["buy_slope_threshold"],
    sell_slope_threshold: float = DEFAULT_MA_SLOPE_GATE["sell_slope_threshold"],
    require_close_confirm_for_buy: bool = DEFAULT_MA_SLOPE_GATE["require_close_confirm_for_buy"],
    require_close_confirm_for_sell: bool = DEFAULT_MA_SLOPE_GATE["require_close_confirm_for_sell"],
    index: Optional[Any] = None,
) -> Tuple[bool, List[str], Dict[str, Any]]:
    row = _select_metrics_row(metrics, index=index)
    if row is None:
        return _missing_data_result(side)

    return evaluate_ma_slope_gate_from_values(
        close=float(row["close"]),
        ma_fast=float(row["ma_fast"]),
        ma_slow=float(row["ma_slow"]),
        slope_pct=float(row["slope_pct"]),
        side=side,
        buy_slope_threshold=buy_slope_threshold,
        sell_slope_threshold=sell_slope_threshold,
        require_close_confirm_for_buy=require_close_confirm_for_buy,
        require_close_confirm_for_sell=require_close_confirm_for_sell,
    )


def evaluate_ma_slope_gate_from_values(
    *,
    close: Optional[float],
    ma_fast: Optional[float],
    ma_slow: Optional[float],
    slope_pct: Optional[float],
    side: str,
    buy_slope_threshold: float = DEFAULT_MA_SLOPE_GATE["buy_slope_threshold"],
    sell_slope_threshold: float = DEFAULT_MA_SLOPE_GATE["sell_slope_threshold"],
    require_close_confirm_for_buy: bool = DEFAULT_MA_SLOPE_GATE["require_close_confirm_for_buy"],
    require_close_confirm_for_sell: bool = DEFAULT_MA_SLOPE_GATE["require_close_confirm_for_sell"],
) -> Tuple[bool, List[str], Dict[str, Any]]:
    if (
        close is None
        or ma_fast is None
        or ma_slow is None
        or slope_pct is None
        or pd.isna(close)
        or pd.isna(ma_fast)
        or pd.isna(ma_slow)
        or pd.isna(slope_pct)
    ):
        return _missing_data_result(side)

    if side == "buy":
        ma_relation_pass = ma_fast < ma_slow
        slope_pass = slope_pct >= buy_slope_threshold
        close_confirm_pass = (close > ma_fast) if require_close_confirm_for_buy else True
        reasons = [
            _format_reason("MA20<MA200", ma_relation_pass, ma_fast, ma_slow),
            _format_reason("slope20>=th", slope_pass, slope_pct, buy_slope_threshold),
            _format_reason("Close confirm", close_confirm_pass, close, ma_fast),
        ]
        gate_pass = bool(ma_relation_pass and slope_pass and close_confirm_pass)
    else:
        ma_relation_pass = ma_fast > ma_slow
        slope_pass = slope_pct <= sell_slope_threshold
        close_confirm_pass = (close < ma_fast) if require_close_confirm_for_sell else True
        reasons = [
            _format_reason("MA20>MA200", ma_relation_pass, ma_fast, ma_slow),
            _format_reason("slope20<=th", slope_pass, slope_pct, sell_slope_threshold),
            _format_reason("Close confirm", close_confirm_pass, close, ma_fast),
        ]
        gate_pass = bool(ma_relation_pass and slope_pass and close_confirm_pass)

    return gate_pass, reasons, {
        "ma_relation_pass": bool(ma_relation_pass),
        "slope_pass": bool(slope_pass),
        "close_confirm_pass": bool(close_confirm_pass),
        "ma_fast": float(ma_fast),
        "ma_slow": float(ma_slow),
        "slope_pct": float(slope_pct),
        "close": float(close),
    }


def _select_metrics_row(metrics: pd.DataFrame, index: Optional[Any] = None) -> Optional[pd.Series]:
    if metrics.empty:
        return None
    if index is not None:
        if index not in metrics.index:
            return None
        row = metrics.loc[index]
    else:
        trimmed = metrics.dropna(subset=["ma_fast", "ma_slow", "slope_pct", "close"])
        if trimmed.empty:
            return None
        row = trimmed.iloc[-1]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[-1]
    if (
        pd.isna(row.get("ma_fast"))
        or pd.isna(row.get("ma_slow"))
        or pd.isna(row.get("slope_pct"))
        or pd.isna(row.get("close"))
    ):
        return None
    return row


def _format_reason(label: str, passed: bool, left: float, right: float) -> str:
    status = "pass" if passed else "fail"
    if "slope" in label:
        return f"{label} {status} ({left * 100:.2f}% vs {right * 100:.2f}%)"
    return f"{label} {status} ({left:.2f} vs {right:.2f})"


def _missing_data_result(side: str) -> Tuple[bool, List[str], Dict[str, Any]]:
    reasons = [
        "MA20<MA200 fail (데이터 부족)" if side == "buy" else "MA20>MA200 fail (데이터 부족)",
        "slope20>=th fail (데이터 부족)" if side == "buy" else "slope20<=th fail (데이터 부족)",
        "Close confirm fail (데이터 부족)",
    ]
    return False, reasons, {
        "ma_relation_pass": False,
        "slope_pass": False,
        "close_confirm_pass": False,
        "ma_fast": None,
        "ma_slow": None,
        "slope_pct": None,
        "close": None,
    }
