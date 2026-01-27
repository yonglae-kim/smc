from __future__ import annotations

from typing import Dict, Any, List, Optional, Callable

import pandas as pd

from ..engine import analyze_symbol
from ..regime.regime import compute_regime
from ..scoring import score_candidate
from ..strategy.base import Strategy
from ..trading.models import Position
from ..trading.rules import TradeRules
from ..utils.progress import Progress


def _apply_cost(px: float, fee_bps: float, slippage_bps: float) -> float:
    return px * (1.0 + (fee_bps + slippage_bps) / 10000.0)


def _required_regime_bars(cfg) -> int:
    ma_slow = int(getattr(cfg.analysis, "ma_slow", 200))
    rsi_period = int(getattr(cfg.analysis, "rsi_period", 14))
    atr_period = int(getattr(cfg.analysis, "atr_period", 14))
    rs_lookback = int(getattr(cfg.regime, "rs_lookback_days", 60))
    min_regime_bars = int(getattr(cfg.regime, "min_regime_bars", 0))
    return max(200, ma_slow, rsi_period + 1, atr_period + 60, rs_lookback + 5, min_regime_bars)


def _slice_index_df(
    meta: Dict[str, Any],
    idx_kospi: Optional[pd.DataFrame],
    idx_kosdaq: Optional[pd.DataFrame],
    dt: pd.Timestamp,
    min_bars: int,
) -> Optional[pd.DataFrame]:
    index_df = None
    if meta.get("market") == "KOSPI" and idx_kospi is not None and len(idx_kospi) > 0:
        index_df = idx_kospi[idx_kospi["date"] <= dt].copy()
    elif meta.get("market") == "KOSDAQ" and idx_kosdaq is not None and len(idx_kosdaq) > 0:
        index_df = idx_kosdaq[idx_kosdaq["date"] <= dt].copy()
    if index_df is not None and len(index_df) < min_bars:
        return None
    return index_df


def run_backtest(
    symbols_meta: List[Dict[str, Any]],
    ohlcv_map: Dict[str, pd.DataFrame],
    idx_kospi: pd.DataFrame,
    idx_kosdaq: pd.DataFrame,
    cfg,
    strategy: Strategy,
    on_update: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
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

    trade_rules = TradeRules(cfg, strategy=strategy)
    positions: Dict[str, Position] = {}
    trades: List[Dict[str, Any]] = []
    equity = 1_000_000.0
    equity_curve: List[Dict[str, Any]] = []

    max_positions = int(cfg.backtest.max_positions)
    fee_bps = float(cfg.backtest.fee_bps)
    slippage_bps = float(cfg.backtest.slippage_bps)
    fill_price = str(trade_rules.entry_price_mode)
    stop_grace_days = int(getattr(cfg.backtest, "stop_grace_days", 0))

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

    min_regime_bars = _required_regime_bars(cfg)

    for dt in cal:
        day_i += 1
        prog.tick(day_i, extra=f"date={str(dt.date())} pos={len(positions)} eq={equity:,.0f}")
        date_str = str(dt.date())

        # exits
        to_close = []
        partial_closes = []
        for sym, pos in list(positions.items()):
            df = ohlcv_map.get(sym)
            if df is None or len(df) == 0:
                pos.hold_days += 1
                continue

            d = df[df["date"] == dt]
            if d.empty:
                pos.hold_days += 1
                continue

            row = d.iloc[0]
            close_px = float(row["close"])
            low_px = float(row["low"])
            high_px = float(row["high"])
            if pos.entry_price > 0:
                pos.mae = min(pos.mae, low_px - pos.entry_price)
                pos.mfe = max(pos.mfe, high_px - pos.entry_price)
            ctx = None
            if trade_rules.exit_on_structure_break or trade_rules.exit_on_score_drop or trade_rules.trail_atr_mult > 0:
                meta = next((m for m in symbols_meta if m["symbol"] == sym), None)
                if meta is not None:
                    df_slice = df[df["date"] <= dt].copy()
                    index_df = _slice_index_df(meta, idx_kospi, idx_kosdaq, dt, min_regime_bars)
                    ctx = analyze_symbol(meta, df_slice, index_df, cfg)
                    if ctx:
                        ctx["regime"] = (
                            compute_regime(index_df, cfg)
                            if index_df is not None
                            else {"tag": "UNKNOWN", "ma200": None, "rsi": None, "atr_spike": None}
                        )
                        ctx = score_candidate(ctx, cfg.scoring.weights)
                        ctx["soft_score"] = trade_rules.strategy.evaluate(ctx)["score"]
                        trade_rules.update_trailing_stop(pos, ctx)

            if stop_grace_days > 0 and pos.hold_days < stop_grace_days:
                pos.hold_days += 1
                continue

            exit_decisions = trade_rules.evaluate_exit(
                pos,
                {"open": float(row["open"]), "high": high_px, "low": low_px, "close": close_px},
                ctx,
                date_str,
            )

            for d in exit_decisions:
                if d.action == "PARTIAL" and d.size:
                    partial_closes.append((sym, d.price, "TP1", d.size))
                if d.action == "EXIT":
                    to_close.append((sym, d.price, d.reason))
                    break

            pos.hold_days += 1

        for sym, exit_px, why, size_to_close in partial_closes:
            pos = positions.get(sym)
            if pos is None:
                continue
            exit_px_costed = exit_px * (1.0 - (fee_bps + slippage_bps) / 10000.0)
            pnl = (exit_px_costed - pos.entry_price) * size_to_close
            risk_per_share = pos.entry_price - (pos.entry_stop_loss or pos.stop_loss)
            rr_realized = (exit_px_costed - pos.entry_price) / risk_per_share if risk_per_share > 0 else None
            equity += pnl
            pos.remaining_size = max(0.0, pos.remaining_size - size_to_close)
            pos.took_partial = True
            if trade_rules.move_stop_to_entry and pos.stop_loss < pos.entry_price:
                pos.stop_loss = pos.entry_price
            trade_rules.apply_tp1_risk_reduction(pos, None)
            if pos.remaining_size <= 0:
                positions.pop(sym, None)
            trades.append(
                {
                    "symbol": sym,
                    "name": pos.name,
                    "entry_date": pos.entry_time,
                    "exit_date": date_str,
                    "entry_px": pos.entry_price,
                    "exit_px": exit_px_costed,
                    "size": size_to_close,
                    "pnl": pnl,
                    "exit_reason": why,
                    "entry_reason": pos.exit_rules.get("entry_reason", ""),
                    "entry_score": pos.entry_score,
                    "entry_breakdown": pos.entry_breakdown or {},
                    "hold_days": pos.hold_days,
                    "stop_distance_atr": pos.stop_distance_atr,
                    "rr_realized": rr_realized,
                    "entry_regime_tag": pos.entry_regime_tag,
                    "entry_structure_bias": pos.entry_structure_bias,
                    "mae": pos.mae,
                    "mfe": pos.mfe,
                }
            )
            stats["exited"] += 1

        for sym, exit_px, why in to_close:
            pos = positions.pop(sym, None)
            if pos is None:
                continue
            size_to_close = pos.remaining_size if pos.remaining_size > 0 else 0.0
            if size_to_close <= 0:
                continue
            exit_px_costed = exit_px * (1.0 - (fee_bps + slippage_bps) / 10000.0)
            pnl = (exit_px_costed - pos.entry_price) * size_to_close
            risk_per_share = pos.entry_price - (pos.entry_stop_loss or pos.stop_loss)
            rr_realized = (exit_px_costed - pos.entry_price) / risk_per_share if risk_per_share > 0 else None
            equity += pnl
            trades.append(
                {
                    "symbol": sym,
                    "name": pos.name,
                    "entry_date": pos.entry_time,
                    "exit_date": date_str,
                    "entry_px": pos.entry_price,
                    "exit_px": exit_px_costed,
                    "size": size_to_close,
                    "pnl": pnl,
                    "exit_reason": why,
                    "entry_reason": pos.exit_rules.get("entry_reason", ""),
                    "entry_score": pos.entry_score,
                    "entry_breakdown": pos.entry_breakdown or {},
                    "hold_days": pos.hold_days,
                    "stop_distance_atr": pos.stop_distance_atr,
                    "rr_realized": rr_realized,
                    "entry_regime_tag": pos.entry_regime_tag,
                    "entry_structure_bias": pos.entry_structure_bias,
                    "mae": pos.mae,
                    "mfe": pos.mfe,
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

                index_df = _slice_index_df(meta, idx_kospi, idx_kosdaq, dt, min_regime_bars)

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

                signal, entry_plan = trade_rules.build_signal(date_str, ctx, cal, entry_price=float(ctx.get("close", 0.0)))
                if not trade_rules.signal_passes(signal):
                    stats["strategy_exclude"] += 1
                day_candidates.append((float(signal.score), sym, ctx, df_full, signal, entry_plan, meta.get("name", "")))

            stats["candidates"] += len(day_candidates)
            selected_pairs = trade_rules.select_buy_candidates([(c[4], c[5]) for c in day_candidates])
            selected_symbols = {p[0].symbol for p in selected_pairs}
            if not selected_symbols:
                continue
            day_candidates = [c for c in day_candidates if c[1] in selected_symbols]
            day_candidates.sort(key=lambda x: (-x[0], x[1]))
            slots = max_positions - len(positions)

            for s, sym, ctx, df_full, signal, entry_plan, name in day_candidates[:slots]:
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
                entry_plan = trade_rules.build_entry_plan(ctx, entry_px)
                if entry_px <= 0 or not pd.notna(entry_px):
                    continue
                risk_budget = equity * float(cfg.backtest.risk_per_trade)
                risk_per_share = entry_px - entry_plan.stop_loss
                if not pd.notna(risk_per_share) or risk_per_share <= 0:
                    risk_per_share = entry_px * float(trade_rules.min_risk_ratio)
                if risk_per_share <= 0:
                    continue
                size = max(0.0, risk_budget / risk_per_share)
                entry_px_costed = _apply_cost(entry_px, fee_bps, slippage_bps)

                position = trade_rules.build_position(signal, entry_plan, str(entry_dt.date()), entry_px_costed, size, ctx)
                position.exit_rules["entry_reason"] = "; ".join(signal.reasons)
                positions[sym] = position
                stats["entered"] += 1

        equity_curve.append({"date": date_str, "equity": equity, "positions": len(positions)})
        if on_update is not None:
            on_update(
                {
                    "start": str(start.date()),
                    "end": str(end.date()),
                    "equity_start": 1_000_000.0,
                    "equity_end": equity,
                    "trades": trades,
                    "equity_curve": equity_curve,
                },
                {
                    "date": date_str,
                    "equity": equity,
                    "positions": len(positions),
                    "trades": len(trades),
                },
            )

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
