from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List

import pandas as pd

from ..engine import analyze_symbol
from ..regime.regime import compute_regime
from ..scoring import score_candidate
from ..strategy.base import Strategy
from ..utils.progress import Progress


@dataclass
class Position:
    symbol: str
    entry_date: str
    entry_px: float
    size: float
    stop_px: float
    tp_px: float
    hold: int = 0
    reason: str = ""
    entry_score: float = 0.0
    entry_breakdown: Dict[str, Any] = None


def _apply_cost(px: float, fee_bps: float, slippage_bps: float) -> float:
    return px * (1.0 + (fee_bps + slippage_bps) / 10000.0)


def run_backtest(
    symbols_meta: List[Dict[str, Any]],
    ohlcv_map: Dict[str, pd.DataFrame],
    idx_kospi: pd.DataFrame,
    idx_kosdaq: pd.DataFrame,
    cfg,
    strategy: Strategy,
) -> Dict[str, Any]:
    start = pd.to_datetime(cfg.backtest.start)
    end = pd.to_datetime(cfg.backtest.end)

    # Calendar: prefer index dates, fallback to union of symbol dates.
    idx_dates = set()
    if idx_kospi is not None and len(idx_kospi) > 0:
        idx_dates |= set(idx_kospi["date"])
    if idx_kosdaq is not None and len(idx_kosdaq) > 0:
        idx_dates |= set(idx_kosdaq["date"])

    if idx_dates:
        cal = pd.Index(sorted(idx_dates))
    else:
        sym_dates = set()
        for df in ohlcv_map.values():
            if df is not None and len(df) > 0:
                sym_dates |= set(df["date"])
        cal = pd.Index(sorted(sym_dates))

    cal = cal[(cal >= start) & (cal <= end)]
    cal = list(cal)

    print(f"[Backtest] calendar size: {len(cal)} start={cal[0] if cal else None} end={cal[-1] if cal else None}", flush=True)

    if not cal:
        print("[Backtest][ERROR] Trading calendar is empty. Check date range and data fetch.", flush=True)
        return {
            "start": str(start.date()),
            "end": str(end.date()),
            "equity_start": 1_000_000.0,
            "equity_end": 1_000_000.0,
            "trades": [],
            "equity_curve": [],
        }

    # normalize indexes
    if idx_kospi is not None and len(idx_kospi) > 0:
        idx_kospi = idx_kospi.sort_values("date").reset_index(drop=True)
    if idx_kosdaq is not None and len(idx_kosdaq) > 0:
        idx_kosdaq = idx_kosdaq.sort_values("date").reset_index(drop=True)

    positions: Dict[str, Position] = {}
    trades: List[Dict[str, Any]] = []
    equity = 1_000_000.0
    equity_curve: List[Dict[str, Any]] = []

    max_positions = int(cfg.backtest.max_positions)
    fee_bps = float(cfg.backtest.fee_bps)
    slippage_bps = float(cfg.backtest.slippage_bps)
    fill_price = str(cfg.backtest.fill_price)

    prog = Progress(total=len(cal), label="SimDays", every=25)
    day_i = 0

    stats = {
        "no_df": 0,
        "short_df": 0,
        "short_slice": 0,
        "no_ctx": 0,
        "strategy_exclude": 0,
        "candidates": 0,
        "entered": 0,
        "exited": 0,
    }

    for dt in cal:
        day_i += 1
        prog.tick(day_i, extra=f"date={str(dt.date())} pos={len(positions)} eq={equity:,.0f}")
        date_str = str(dt.date())

        # exits
        to_close = []
        for sym, pos in list(positions.items()):
            df = ohlcv_map.get(sym)
            if df is None or len(df) == 0:
                pos.hold += 1
                continue

            d = df[df["date"] == dt]
            if d.empty:
                pos.hold += 1
                continue

            row = d.iloc[0]
            close_px = float(row["close"])
            low_px = float(row["low"])
            high_px = float(row["high"])

            if low_px <= pos.stop_px:
                to_close.append((sym, pos.stop_px, "STOP"))
                continue
            if high_px >= pos.tp_px:
                to_close.append((sym, pos.tp_px, "TP"))
                continue

            pos.hold += 1
            if pos.hold >= 20:
                to_close.append((sym, close_px, "TIME"))

        for sym, exit_px, why in to_close:
            pos = positions.pop(sym, None)
            if pos is None:
                continue
            exit_px_costed = exit_px * (1.0 - (fee_bps + slippage_bps) / 10000.0)
            pnl = (exit_px_costed - pos.entry_px) * pos.size
            equity += pnl
            trades.append(
                {
                    "symbol": sym,
                    "entry_date": pos.entry_date,
                    "exit_date": date_str,
                    "entry_px": pos.entry_px,
                    "exit_px": exit_px_costed,
                    "size": pos.size,
                    "pnl": pnl,
                    "exit_reason": why,
                    "entry_reason": pos.reason,
                    "entry_score": pos.entry_score,
                    "entry_breakdown": pos.entry_breakdown or {},
                }
            )
            stats["exited"] += 1

        # entries
        if len(positions) < max_positions:
            day_candidates = []

            for meta in symbols_meta:
                sym = meta["symbol"]
                if sym in positions:
                    continue

                df_full = ohlcv_map.get(sym)
                if df_full is None:
                    stats["no_df"] += 1
                    continue
                if len(df_full) < 80:
                    stats["short_df"] += 1
                    continue

                df_slice = df_full[df_full["date"] <= dt].copy()
                if len(df_slice) < (int(cfg.backtest.warmup_bars) // 2):
                    stats["short_slice"] += 1
                    continue

                index_df = None
                if meta.get("market") == "KOSPI" and idx_kospi is not None and len(idx_kospi) > 0:
                    index_df = idx_kospi[idx_kospi["date"] <= dt].copy()
                elif meta.get("market") == "KOSDAQ" and idx_kosdaq is not None and len(idx_kosdaq) > 0:
                    index_df = idx_kosdaq[idx_kosdaq["date"] <= dt].copy()

                if index_df is not None and len(index_df) < 60:
                    index_df = None

                ctx = analyze_symbol(meta, df_slice, index_df, cfg)
                if ctx is None:
                    stats["no_ctx"] += 1
                    continue

                ctx["regime"] = (
                    compute_regime(index_df, cfg)
                    if index_df is not None
                    else {"tag": "UNKNOWN", "ma200": None, "rsi": None, "atr_spike": None}
                )
                ctx = score_candidate(ctx, cfg.scoring.weights)

                ranked = strategy.rank(date_str, sym, ctx)
                if ranked is None:
                    stats["strategy_exclude"] += 1
                    continue

                s, reason, breakdown = ranked
                day_candidates.append((float(s), sym, ctx, df_full, reason, breakdown))

            stats["candidates"] += len(day_candidates)

            day_candidates.sort(key=lambda x: x[0], reverse=True)
            slots = max_positions - len(positions)

            for s, sym, ctx, df_full, reason, breakdown in day_candidates[:slots]:
                entry_dt = dt
                if fill_price == "next_open":
                    df_after = df_full[df_full["date"] > dt].head(1)
                    if df_after.empty:
                        continue
                    entry_dt = df_after["date"].iloc[0]
                    entry_px = float(df_after["open"].iloc[0])
                else:
                    d = df_full[df_full["date"] == dt]
                    if d.empty:
                        continue
                    entry_px = float(d["close"].iloc[0])

                ob = ctx.get("ob") or {}
                stop_px = float(ob.get("invalidation", entry_px * 0.95))
                min_risk_ratio = 0.001
                if stop_px >= entry_px:
                    adjusted_stop = entry_px * (1 - min_risk_ratio)
                    print(
                        "[Backtest][Warn] stop_px >= entry_px; adjusting stop",
                        {"symbol": sym, "entry_px": entry_px, "stop_px": stop_px, "adjusted_stop": adjusted_stop},
                        flush=True,
                    )
                    stop_px = adjusted_stop
                risk_per_share = max(1e-6, entry_px - stop_px)
                min_risk_per_share = entry_px * min_risk_ratio
                if risk_per_share < min_risk_per_share:
                    print(
                        "[Backtest][Warn] risk_per_share too small",
                        {"symbol": sym, "entry_px": entry_px, "stop_px": stop_px, "risk_per_share": risk_per_share},
                        flush=True,
                    )
                risk_budget = equity * float(cfg.backtest.risk_per_trade)
                size = max(0.0, risk_budget / risk_per_share)

                tp_px = entry_px + 2.0 * risk_per_share
                entry_px_costed = _apply_cost(entry_px, fee_bps, slippage_bps)

                positions[sym] = Position(
                    symbol=sym,
                    entry_date=str(entry_dt.date()),
                    entry_px=entry_px_costed,
                    size=size,
                    stop_px=stop_px,
                    tp_px=tp_px,
                    reason=reason,
                    entry_score=float(s),
                    entry_breakdown=breakdown,
                )
                stats["entered"] += 1

        equity_curve.append({"date": date_str, "equity": equity, "positions": len(positions)})

        if day_i % 50 == 0:
            print("[Backtest][Stats]", stats, flush=True)

    return {
        "start": str(start.date()),
        "end": str(end.date()),
        "equity_start": 1_000_000.0,
        "equity_end": equity,
        "trades": trades,
        "equity_curve": equity_curve,
    }
