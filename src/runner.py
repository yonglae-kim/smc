from __future__ import annotations
import os, math
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
from .regime.regime import compute_regime
from .scoring import score_candidate
from .reporting.charts import plot_symbol_chart
from .reporting.html import render_report

def run(config_path: str) -> None:
    cfg = load_config(config_path)
    os.makedirs(cfg.app.cache_dir, exist_ok=True)
    os.makedirs(cfg.app.out_dir, exist_ok=True)

    http = HttpClient(
        timeout_sec=cfg.network.timeout_sec,
        max_retries=cfg.network.max_retries,
        backoff_base_sec=cfg.network.backoff_base_sec,
        jitter_sec=cfg.network.jitter_sec,
        rate_limit_per_sec=cfg.network.rate_limit_per_sec,
    )
    storage = FSStorage(cfg.app.cache_dir)
    provider = NaverChartProvider(http)
    fetcher = NaverMarketSumFetcher(http)

    print("[Runner] Start daily pipeline", flush=True)
    print("[Runner] Building market regime (KOSPI/KOSDAQ)", flush=True)

    # --- regime (must be split by market) ---
    idx_kospi = provider.get_index_ohlc("KOSPI", count=int(cfg.regime.index_lookback_days))
    idx_kosdaq = provider.get_index_ohlc("KOSDAQ", count=int(cfg.regime.index_lookback_days))
    regime_kospi = compute_regime(idx_kospi, cfg)
    regime_kosdaq = compute_regime(idx_kosdaq, cfg)

    print(f"[Runner] Universe build: Top{cfg.universe.top_liquidity} (incremental/weekly policy)", flush=True)

    # --- universe ---
    ub = UniverseBuilder(storage, provider, fetcher, cfg.universe)
    universe, uni_meta = ub.build()

    print("[Runner] Per-symbol analysis (Top500)", flush=True)

    # --- per-symbol analysis (resume capable) ---
    ymd = today_kst().strftime("%Y-%m-%d")
    snap_dir = storage.snapshot_dir(ymd)
    out_dir = storage.out_dir(cfg.app.out_dir, ymd)

    # resume state for analysis
    st = storage.load_json(f"state/analysis_progress_{ymd}.json", default={"done": []})
    done = set(st.get("done", []))
    rows=[]
    ctx_map={}
    prog = Progress(total=len(universe), label="Analyze", every=25)
    done_count = len(done)
    for i, meta in enumerate(universe, start=1):
        sym = meta["symbol"]
        if sym in done:
            continue
        df = storage.load_ohlcv_cache(sym)
        if df is None or len(df) < cfg.universe.ohlcv_lookback_days:
            try:
                df_new = provider.get_ohlcv(sym, count=max(cfg.universe.ohlcv_lookback_days, 300))
                if df_new is not None and len(df_new) >= 60:
                    storage.save_ohlcv_cache(sym, df_new)
                    df = df_new
            except Exception:
                df = None

        if df is None or len(df) < 80:
            done.add(sym); done_count += 1; prog.tick(done_count, extra=f"skip={sym} reason=insufficient_data"); continue

        index_df = idx_kospi if meta.get("market")=="KOSPI" else idx_kosdaq
        ctx = analyze_symbol(meta, df, index_df, cfg)
        if ctx is None:
            done.add(sym); done_count += 1; prog.tick(done_count, extra=f"skip={sym} reason=insufficient_data"); continue

        # inject market regime
        ctx["regime"] = regime_kospi if meta.get("market")=="KOSPI" else regime_kosdaq
        ctx = score_candidate(ctx, cfg.scoring.weights)

        # tags for report
        tags=[]
        if ctx["structure_bias"]!="NEUTRAL": tags.append(f"STRUCT_{ctx['structure_bias']}")
        if ctx.get("bos",{}).get("direction"): tags.append(f"BOS_{ctx['bos']['direction']}")
        if ctx.get("ob"): tags.append(f"OB_{ctx['ob']['kind']}_{ctx['ob']['status']}")
        if ctx.get("fvg"): tags.append(f"FVG_{ctx['fvg']['kind']}_{ctx['fvg']['status']}")
        if ctx.get("tag_confluence_ob_fvg"): tags.append("Confluence_OB_FVG")
        tags.append(ctx.get("rs",{}).get("tag","RS_UNKNOWN"))
        tags.append(f"REGIME_{ctx.get('regime',{}).get('tag','MIXED')}")
        if ctx.get("regime",{}).get("atr_spike"): tags.append("Risk_ATR_Spike")
        ctx["tags"]=tags

        rows.append(ctx)
        ctx_map[sym]=ctx
        done.add(sym)

        if len(done) % 60 == 0:
            storage.save_json(f"state/analysis_progress_{ymd}.json", {"done": sorted(list(done))})

    storage.save_json(f"state/analysis_progress_{ymd}.json", {"done": sorted(list(done))})

    # Rank (candidates are still within top500 universe)
    rows_sorted = sorted(rows, key=lambda x: x.get("score", 0), reverse=True)

    # snapshot
    storage.save_json(f"snapshots/{ymd}/universe.json", uni_meta)
    storage.save_json(f"snapshots/{ymd}/candidates.json", rows_sorted)
    storage.save_json(f"snapshots/{ymd}/regime.json", {"KOSPI": regime_kospi, "KOSDAQ": regime_kosdaq})

    print(f"[Runner] Ranking {len(rows_sorted)} analyzed rows and generating HTML report", flush=True)

    # --- report build ---
    detail_n = int(cfg.scoring.top_detail)
    details = rows_sorted[:detail_n]
    # charts: need df again; load from cache and recompute indicators for plotting
    for c in details:
        df = storage.load_ohlcv_cache(c["symbol"])
        # indicators should already exist in engine but not stored; recompute minimal for plotting
        df = df.sort_values("date").reset_index(drop=True)
        from .analysis.indicators import sma, rsi, atr
        df["ma20"] = sma(df["close"], int(cfg.analysis.ma_fast))
        df["ma200"] = sma(df["close"], int(cfg.analysis.ma_slow))
        df["rsi14"] = rsi(df["close"], int(cfg.analysis.rsi_period))
        df["atr14"] = atr(df, int(cfg.analysis.atr_period))
        c["chart_b64"] = plot_symbol_chart(df, c, lookback=int(cfg.report.chart_lookback))
        # context text
        lev=[]
        if c.get("bos",{}).get("direction"):
            lev.append(f"BOS {c['bos']['direction']} level={c['bos']['level']:.0f} q={c['bos'].get('quality',0):.2f}")
        if c.get("ob"):
            ob=c["ob"]; lev.append(f"OB {ob['kind']} zone=[{ob['lower']:.0f},{ob['upper']:.0f}] inv={ob['invalidation']:.0f} q={ob.get('quality',0):.2f}")
        if c.get("fvg"):
            f=c["fvg"]; lev.append(f"FVG {f['kind']} zone=[{f['lower']:.0f},{f['upper']:.0f}] status={f['status']} age={f.get('age',0)}")
        c["context_text"] = "\n".join(lev) if lev else "(no recent zones detected)"
        c["score_text"] = "\n".join([f"{x['key']}: {x['w']}" + (f" (val={x.get('val')})" if x.get('val') is not None else "") for x in c.get("score_components",[])]) or "(no components)"

    table_rows=[]
    for rank, c in enumerate(rows_sorted[: int(cfg.report.max_table_rows)], start=1):
        levels=[]
        if c.get("ob"): levels.append(f"OB[{c['ob']['lower']:.0f}-{c['ob']['upper']:.0f}] inv:{c['ob']['invalidation']:.0f}")
        if c.get("fvg"): levels.append(f"FVG[{c['fvg']['lower']:.0f}-{c['fvg']['upper']:.0f}] {c['fvg']['status']}")
        if c.get("bos",{}).get("direction"): levels.append(f"BOS:{c['bos']['level']:.0f}")
        table_rows.append({
            "rank": rank, "score": c.get("score",0.0), "symbol": c["symbol"], "name": c.get("name",""),
            "market": c.get("market",""), "tags": c.get("tags",[]), "close": c.get("close",0.0),
            "ma200": c.get("ma200"), "rsi14": c.get("rsi14"), "levels": " | ".join(levels)
        })

    payload = {
        "title": cfg.report.title,
        "generated_at": now_kst_iso(),
        "universe_n": len(universe),
        "liquidity_window": cfg.universe.liquidity_window,
        "detail_n": detail_n,
        "regime_kospi": regime_kospi,
        "regime_kosdaq": regime_kosdaq,
        "table_rows": table_rows,
        "details": details
    }
    out_html = os.path.join(out_dir, "report.html")
    render_report(out_html, payload, include_js=bool(cfg.report.include_sort_search_js))
    print(f"Report written: {out_html}")
