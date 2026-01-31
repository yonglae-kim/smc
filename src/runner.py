from __future__ import annotations
import os, math
import shutil
from typing import Dict, Any, List
import pandas as pd

from .config import load_config
from .utils.http import HttpClient
from .utils.time import today_kst, now_kst_iso
from .utils.progress import Progress
from .providers.naver import NaverChartProvider, NaverMarketSumFetcher
from .storage.fs import FSStorage
from .universe.builder import UniverseBuilder
from .engine import analyze_symbol
from .scoring import score_candidate
from .reporting.charts import plot_symbol_chart
from .reporting.html import render_report
from .strategy.soft_score import SoftScoreStrategy
from .trading.models import EntryPlan, Position, TradeSignal
from .trading.rules import TradeRules

def run(config_path: str) -> None:
    cfg = load_config(config_path)
    ymd = today_kst().strftime("%Y-%m-%d")
    os.makedirs(cfg.app.cache_dir, exist_ok=True)
    os.makedirs(cfg.app.out_dir, exist_ok=True)
    strategy = SoftScoreStrategy(cfg)
    trade_rules = TradeRules(cfg, strategy=strategy)

    cache_mode = cfg.network.cache_mode
    snapshot_id = cfg.network.cache_snapshot_id or ymd
    cache_dir = os.path.join(cfg.app.cache_dir, "http", snapshot_id if cache_mode == "snapshot" else "latest")
    from .utils.http_cache import HttpCache
    http_cache = HttpCache(cache_dir, ttl_sec=cfg.network.cache_ttl_sec, mode=cache_mode)
    http = HttpClient(
        timeout_sec=cfg.network.timeout_sec,
        max_retries=cfg.network.max_retries,
        backoff_base_sec=cfg.network.backoff_base_sec,
        jitter_sec=cfg.network.jitter_sec,
        rate_limit_per_sec=cfg.network.rate_limit_per_sec,
        cache=http_cache,
    )
    storage = FSStorage(cfg.app.cache_dir)
    provider = NaverChartProvider(http)
    fetcher = NaverMarketSumFetcher(http)

    print("[Runner] Start daily pipeline", flush=True)
    print(f"[Runner] Universe build: Top{cfg.universe.top_liquidity} (incremental/weekly policy)", flush=True)

    # --- universe ---
    ub = UniverseBuilder(storage, provider, fetcher, cfg.universe)
    universe, uni_meta = ub.build()

    print("[Runner] Per-symbol analysis (Top500)", flush=True)

    # --- per-symbol analysis (resume capable) ---
    out_dir = storage.out_dir(cfg.app.out_dir, ymd)

    # resume state for analysis
    st = storage.load_json(f"state/analysis_progress_{ymd}.json", default={"done": []})
    done = set(st.get("done", []))
    if done and len(done) >= len(universe):
        print("[Runner] Analysis progress is complete; resetting progress for re-run.", flush=True)
        done = set()
    rows=[]
    ctx_map={}
    cal_dates = set()
    prog = Progress(total=len(universe), label="Analyze", every=25)
    done_count = len(done)
    min_bars = max(80, int(cfg.universe.ohlcv_lookback_days))
    for i, meta in enumerate(universe, start=1):
        sym = meta["symbol"]
        if sym in done:
            continue
        df = storage.load_ohlcv_cache(sym)
        last_date = None
        if df is not None and not df.empty:
            last_date = pd.to_datetime(df["date"], errors="coerce").max()
            if pd.isna(last_date):
                last_date = None
            else:
                last_date = last_date.date()
        stale = last_date is None or last_date < today_kst()
        if df is None or len(df) < cfg.universe.ohlcv_lookback_days:
            try:
                df_new = provider.get_ohlcv(sym, count=max(cfg.universe.ohlcv_lookback_days, 300))
                if df_new is not None and len(df_new) >= 60:
                    storage.save_ohlcv_cache(sym, df_new)
                    df = df_new
            except Exception:
                df = None
        elif stale:
            try:
                df_new = provider.get_ohlcv(sym, count=10)
                if df_new is not None and len(df_new) >= 1:
                    df = pd.concat([df, df_new], ignore_index=True)
                    df["date"] = pd.to_datetime(df["date"], errors="coerce")
                    df = (
                        df.dropna(subset=["date"])
                        .drop_duplicates(subset=["date"])
                        .sort_values("date")
                        .reset_index(drop=True)
                    )
                    storage.save_ohlcv_cache(sym, df)
            except Exception:
                pass

        if df is None or len(df) < min_bars:
            done.add(sym); done_count += 1; prog.tick(done_count, extra=f"skip={sym} reason=insufficient_data"); continue

        ctx = analyze_symbol(meta, df, cfg)
        if ctx is None:
            done.add(sym); done_count += 1; prog.tick(done_count, extra=f"skip={sym} reason=insufficient_data"); continue
        ctx = score_candidate(ctx, cfg.scoring.weights)
        cal_dates |= set(df["date"])

        # tags for report
        tags=[]
        if ctx["structure_bias"]!="NEUTRAL": tags.append(f"STRUCT_{ctx['structure_bias']}")
        if ctx.get("bos",{}).get("direction"): tags.append(f"BOS_{ctx['bos']['direction']}")
        if ctx.get("ob"): tags.append(f"OB_{ctx['ob']['kind']}_{ctx['ob']['status']}")
        if ctx.get("fvg"): tags.append(f"FVG_{ctx['fvg']['kind']}_{ctx['fvg']['status']}")
        if ctx.get("tag_confluence_ob_fvg"): tags.append("Confluence_OB_FVG")
        tags.append(ctx.get("rs",{}).get("tag","RS_UNKNOWN"))
        ctx["tags"]=tags

        rows.append(ctx)
        ctx_map[sym]=ctx
        done.add(sym)

        if len(done) % 60 == 0:
            storage.save_json(f"state/analysis_progress_{ymd}.json", {"done": sorted(list(done))})

    storage.save_json(f"state/analysis_progress_{ymd}.json", {"done": []})

    cal = sorted(cal_dates)

    # Rank (candidates are still within top500 universe)
    rows_sorted = sorted(rows, key=lambda x: (-x.get("score", 0), x.get("symbol", "")))

    signal_rows = []
    for ctx in rows_sorted:
        signal, entry_plan = trade_rules.build_signal(ctx["asof"], ctx, cal, entry_price=float(ctx.get("close", 0.0)))
        ctx["soft_score"] = signal.score
        ctx["soft_score_breakdown"] = signal.score_breakdown
        signal_rows.append({"ctx": ctx, "signal": signal, "entry_plan": entry_plan})

    # snapshot
    storage.save_json(f"snapshots/{ymd}/universe.json", uni_meta)
    storage.save_json(f"snapshots/{ymd}/candidates.json", rows_sorted)
    storage.save_json(
        f"snapshots/{ymd}/signals.json",
        [
            {
                "symbol": s["ctx"]["symbol"],
                "signal": s["signal"].to_dict(),
                "entry_plan": s["entry_plan"].to_dict(),
            }
            for s in signal_rows
        ],
    )

    print(f"[Runner] Ranking {len(rows_sorted)} analyzed rows and generating HTML report", flush=True)

    # --- report build ---
    table_rows=[]
    for rank, c in enumerate(rows_sorted[: int(cfg.report.max_table_rows)], start=1):
        levels=[]
        if c.get("ob"):
            levels.append(f"OB[{c['ob']['lower']:.0f}-{c['ob']['upper']:.0f}] inv:{c['ob']['invalidation']:.0f}")
        if c.get("fvg"):
            levels.append(f"FVG[{c['fvg']['lower']:.0f}-{c['fvg']['upper']:.0f}] {c['fvg']['status']}")
        if c.get("bos",{}).get("direction"):
            levels.append(f"BOS:{c['bos']['level']:.0f}")
        table_rows.append({
            "rank": rank, "score": c.get("score",0.0), "symbol": c["symbol"], "name": c.get("name",""),
            "market": c.get("market",""), "tags": c.get("tags",[]), "close": c.get("close",0.0),
            "ma20": c.get("ma20"),
            "ma200": c.get("ma200"),
            "ma_slope_pct": c.get("ma_slope_pct"),
            "rsi14": c.get("rsi14"),
            "levels": " | ".join(levels),
        })

    signal_map = {r["signal"].symbol: r for r in signal_rows}
    selected = trade_rules.select_buy_candidates([(r["signal"], r["entry_plan"]) for r in signal_rows])
    buy_candidates = [signal_map[s[0].symbol] for s in selected]
    buy_valid_from = trade_rules.next_trading_day(cal, ymd) if cal else ymd

    state = storage.load_json("state/positions_live.json", default={"positions": [], "pending_entries": [], "pending_exits": [], "last_date": None})
    positions = [Position(**p) for p in state.get("positions", [])]
    pending_entries = list(state.get("pending_entries", []))
    pending_exits = list(state.get("pending_exits", []))

    last_date = state.get("last_date")
    if last_date:
        delta_days = (pd.to_datetime(ymd) - pd.to_datetime(last_date)).days
        if delta_days > 0:
            for pos in positions:
                pos.hold_days += delta_days

    # apply pending exits that are due
    remaining_exits = []
    for pe in pending_exits:
        if pe.get("valid_from") and pe["valid_from"] <= ymd:
            positions = [p for p in positions if p.symbol != pe.get("symbol")]
        else:
            remaining_exits.append(pe)
    pending_exits = remaining_exits

    # apply pending entries that are due
    remaining_entries = []
    for pe in pending_entries:
        if pe.get("valid_from") and pe["valid_from"] <= ymd:
            sym = pe.get("symbol")
            ctx = ctx_map.get(sym)
            if ctx is None:
                remaining_entries.append(pe)
                continue
            df = storage.load_ohlcv_cache(sym)
            if df is None or df.empty:
                remaining_entries.append(pe)
                continue
            row = df[df["date"] == pd.to_datetime(ymd)]
            if row.empty:
                remaining_entries.append(pe)
                continue
            entry_px = float(row["open"].iloc[0])
            signal = TradeSignal(**pe["signal"])
            entry_plan = EntryPlan(**pe["entry_plan"])
            position = trade_rules.build_position(signal, entry_plan, ymd, entry_px, 1.0, ctx)
            positions.append(position)
        else:
            remaining_entries.append(pe)
    pending_entries = remaining_entries

    sell_rows = []
    portfolio_rows = []
    sell_details = []
    for pos in positions:
        ctx = ctx_map.get(pos.symbol)
        df = storage.load_ohlcv_cache(pos.symbol)
        if df is None or df.empty:
            continue
        row = df[df["date"] == pd.to_datetime(ymd)]
        if row.empty:
            continue
        bar = {
            "open": float(row["open"].iloc[0]),
            "high": float(row["high"].iloc[0]),
            "low": float(row["low"].iloc[0]),
            "close": float(row["close"].iloc[0]),
        }
        if ctx:
            trade_rules.update_trailing_stop(pos, ctx)
        exit_decisions = trade_rules.evaluate_exit(pos, bar, ctx, ymd)
        last_price = bar["close"]
        pnl_pct = (last_price - pos.entry_price) / max(pos.entry_price, 1e-6) * 100.0
        risk_pct = (last_price - pos.stop_loss) / max(last_price, 1e-6) * 100.0
        exit_action = next((d for d in exit_decisions if d.action == "EXIT"), None)
        next_action = "보유"
        if exit_action:
            next_action = "청산"
            pending_exits.append(
                {
                    "symbol": pos.symbol,
                    "valid_from": trade_rules.next_trading_day(cal, ymd),
                    "reason": exit_action.reason,
                }
            )
            pos.state = "exit_pending"
        for decision in exit_decisions:
            if decision.action == "PARTIAL" and decision.size:
                pos.remaining_size = max(0.0, pos.remaining_size - decision.size)
                pos.took_partial = True
                if trade_rules.move_stop_to_entry and pos.stop_loss < pos.entry_price:
                    pos.stop_loss = pos.entry_price
                trade_rules.apply_tp1_risk_reduction(pos, ctx)

        portfolio_rows.append(
            {
                "symbol": pos.symbol,
                "name": pos.name,
                "entry_price": pos.entry_price,
                "last_price": last_price,
                "pnl_pct": pnl_pct,
                "risk_pct": risk_pct,
                "next_action": next_action,
            }
        )

        if exit_action:
            sell_rows.append(
                {
                    "symbol": pos.symbol,
                    "name": pos.name,
                    "entry_price": pos.entry_price,
                    "last_price": last_price,
                    "pnl_pct": pnl_pct,
                    "exit_reason": exit_action.reason,
                    "next_action": next_action,
                }
            )
            chart_b64 = None
            if ctx:
                ctx = dict(ctx)
                ctx["position"] = pos.to_dict()
                df_chart = storage.load_ohlcv_cache(pos.symbol)
                if df_chart is not None and not df_chart.empty:
                    df_chart = df_chart.sort_values("date").reset_index(drop=True)
                    from .analysis.indicators import sma, rsi, atr
                    df_chart["ma20"] = sma(df_chart["close"], int(cfg.analysis.ma_fast))
                    df_chart["ma200"] = sma(df_chart["close"], int(cfg.analysis.ma_slow))
                    df_chart["rsi14"] = rsi(df_chart["close"], int(cfg.analysis.rsi_period))
                    df_chart["atr14"] = atr(df_chart, int(cfg.analysis.atr_period))
                    chart_b64 = plot_symbol_chart(df_chart, ctx, lookback=int(cfg.report.chart_lookback))
            breakdown = (ctx or {}).get("soft_score_breakdown", {})
            score_text = "\n".join(trade_rules.describe_score_breakdown(breakdown)) if breakdown else "(no components)"
            sell_details.append(
                {
                    "symbol": pos.symbol,
                    "name": pos.name,
                    "market": pos.market,
                    "last_ohlc": bar,
                    "close": last_price,
                    "position": pos.to_dict(),
                    "pnl_pct": pnl_pct,
                    "next_action": next_action,
                    "tags": (ctx or {}).get("tags", []),
                    "chart_b64": chart_b64 or "",
                    "score_text": score_text,
                    "reason_text": "\n".join(trade_rules.build_sell_reasons(exit_decisions, pos, ctx or {})),
                }
            )

    # add new pending entries from buy candidates
    existing_pending = {(e.get("symbol"), e.get("valid_from")) for e in pending_entries}
    for row in buy_candidates:
        signal = row["signal"]
        key = (signal.symbol, signal.valid_from)
        if key in existing_pending:
            continue
        existing_pending.add(key)
        pending_entries.append(
            {
                "symbol": signal.symbol,
                "valid_from": signal.valid_from,
                "signal": signal.to_dict(),
                "entry_plan": row["entry_plan"].to_dict(),
            }
        )

    storage.save_json(
        "state/positions_live.json",
        {
            "positions": [p.to_dict() for p in positions],
            "pending_entries": pending_entries,
            "pending_exits": pending_exits,
            "last_date": ymd,
        },
    )
    storage.save_json(
        f"snapshots/{ymd}/positions.json",
        {
            "positions": [p.to_dict() for p in positions],
            "pending_entries": pending_entries,
            "pending_exits": pending_exits,
            "last_date": ymd,
        },
    )

    detail_n = int(cfg.scoring.top_detail)
    buy_details = []
    for row in buy_candidates[:detail_n]:
        c = dict(row["ctx"])
        c["signal"] = row["signal"].to_dict()
        df = storage.load_ohlcv_cache(c["symbol"])
        df = df.sort_values("date").reset_index(drop=True)
        last_ohlc = None
        if not df.empty:
            last_row = df.iloc[-1]
            last_ohlc = {
                "open": float(last_row["open"]),
                "high": float(last_row["high"]),
                "low": float(last_row["low"]),
                "close": float(last_row["close"]),
            }
            c["close"] = last_ohlc["close"]
        c["last_ohlc"] = last_ohlc
        entry_plan_latest = trade_rules.build_entry_plan(c, entry_price=float(c.get("close", 0.0)))
        c["entry_plan"] = entry_plan_latest.to_dict()
        from .analysis.indicators import sma, rsi, atr
        df["ma20"] = sma(df["close"], int(cfg.analysis.ma_fast))
        df["ma200"] = sma(df["close"], int(cfg.analysis.ma_slow))
        df["rsi14"] = rsi(df["close"], int(cfg.analysis.rsi_period))
        df["atr14"] = atr(df, int(cfg.analysis.atr_period))
        c["chart_b64"] = plot_symbol_chart(df, c, lookback=int(cfg.report.chart_lookback))
        lev=[]
        if c.get("bos",{}).get("direction"):
            lev.append(f"BOS {c['bos']['direction']} level={c['bos']['level']:.0f} q={c['bos'].get('quality',0):.2f}")
        if c.get("ob"):
            ob=c["ob"]; lev.append(f"OB {ob['kind']} zone=[{ob['lower']:.0f},{ob['upper']:.0f}] inv={ob['invalidation']:.0f} q={ob.get('quality',0):.2f}")
        if c.get("fvg"):
            f=c["fvg"]; lev.append(f"FVG {f['kind']} zone=[{f['lower']:.0f},{f['upper']:.0f}] status={f['status']} age={f.get('age',0)}")
        c["context_text"] = "\n".join(lev) if lev else "(no recent zones detected)"
        c["score_text"] = "\n".join(
            trade_rules.describe_score_breakdown(row["signal"].score_breakdown)
        ) if row["signal"].score_breakdown else "(no components)"
        gate_lines = [f"{k}: {'통과' if v else '실패'}" for k, v in row["signal"].gates.items()]
        if row["signal"].gate_reasons:
            gate_lines.append("---")
            gate_lines.extend(row["signal"].gate_reasons)
        c["gate_text"] = "\n".join(gate_lines)
        plan_reasons = row["entry_plan"].rationale + [f"무효화 조건: {row['entry_plan'].invalidation}"]
        all_reasons = list(row["signal"].reasons) + plan_reasons
        c["reason_text"] = "\n".join(all_reasons) if all_reasons else "(no reasons)"
        buy_details.append(c)

    buy_rows = []
    for rank, row in enumerate(buy_candidates, start=1):
        buy_rows.append(
            {
                "rank": rank,
                "symbol": row["signal"].symbol,
                "name": row["ctx"].get("name", ""),
                "signal": row["signal"],
                "entry_plan": row["entry_plan"],
                "gates": [{"key": k, "pass": v} for k, v in row["signal"].gates.items()],
            }
        )

    payload = {
        "title": cfg.report.title,
        "generated_at": now_kst_iso(),
        "universe_n": len(universe),
        "liquidity_window": cfg.universe.liquidity_window,
        "execution_guide": cfg.report.execution_guide,
        "tp_sl_conflict_note": cfg.report.tp_sl_conflict_note,
        "buy_valid_from": buy_valid_from,
        "table_rows": table_rows,
        "buy_rows": buy_rows,
        "sell_rows": sell_rows,
        "portfolio_rows": portfolio_rows,
        "buy_details": buy_details,
        "sell_details": sell_details,
    }
    out_html = os.path.join(out_dir, "report.html")
    render_report(out_html, payload, include_js=bool(cfg.report.include_sort_search_js))
    print(f"Report written: {out_html}")
    web_root = "/var/www/html/jusik"
    try:
        os.makedirs(web_root, exist_ok=True)
        shutil.copy2(out_html, os.path.join(web_root, "report.html"))
        shutil.copy2(out_html, os.path.join(web_root, "index.html"))
        os.chmod(web_root, 0o755)
        os.chmod(os.path.join(web_root, "report.html"), 0o644)
        os.chmod(os.path.join(web_root, "index.html"), 0o644)
        print(f"Report copied to: {os.path.join(web_root, 'report.html')}")
    except Exception as exc:
        print(f"[Runner] Failed to copy report to {web_root}: {exc}")
