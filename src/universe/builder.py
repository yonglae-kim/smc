from __future__ import annotations
import math, os
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np

from ..storage.fs import FSStorage
from ..providers.base import DataProvider, UniverseFetcher
from ..utils.time import today_kst, now_kst_iso
from ..utils.progress import Progress

class UniverseBuilder:
    def __init__(self, storage: FSStorage, provider: DataProvider, fetcher: UniverseFetcher, cfg):
        self.storage = storage
        self.provider = provider
        self.fetcher = fetcher
        self.cfg = cfg

    def _liquidity_median(self, df: pd.DataFrame, window: int) -> float:
        if df is None or len(df) < window:
            return 0.0
        x = (df["close"] * df["volume"]).tail(window)
        x = x.replace([np.inf, -np.inf], np.nan).dropna()
        if len(x) < max(5, window//2):
            return 0.0
        return float(np.median(x.values))

    def build(self) -> Tuple[List[Dict], Dict]:
        """Returns (universe_top500, meta). Implements incremental policy:
        - Daily: recalc liquidity for previous top `daily_recalc_top` + addon set from today's traded value.
        - Weekly: full scan + rebuild.
        - Resume: keep a progress file for the liquidity scan.
        """
        today = today_kst()
        ymd = today.strftime("%Y-%m-%d")
        state = self.storage.load_json("state/universe_progress.json", default={})
        last_universe = self.storage.load_json("state/last_universe.json", default=None)
        weekday = today.weekday()

        full_scan = (last_universe is None) or (weekday == int(self.cfg.weekly_full_scan_weekday))
        symbols_all = None
        if full_scan:
            symbols_all = self.fetcher.fetch_all_symbols()
            candidates = symbols_all
            scan_label = "FULL"
        else:
            prev = last_universe.get("universe_ranked", [])
            prev_top = prev[: int(self.cfg.daily_recalc_top)]
            # addon: today's traded value top from both markets
            addon = set()
            k1 = self.fetcher.fetch_top_value_symbols("KOSPI", int(self.cfg.include_daily_value_rank_addon))
            k2 = self.fetcher.fetch_top_value_symbols("KOSDAQ", int(self.cfg.include_daily_value_rank_addon))
            addon.update(k1); addon.update(k2)
            # map addon to meta using latest full symbol list cache if exists
            sym_cache = self.storage.load_json("state/symbol_cache.json", default=None)
            if sym_cache is None or sym_cache.get("asof") != last_universe.get("symbol_cache_asof"):
                sym_all = self.fetcher.fetch_all_symbols()
                sym_cache = {"asof": ymd, "symbols": sym_all}
                self.storage.save_json("state/symbol_cache.json", sym_cache)
            meta_map = {x["symbol"]: x for x in sym_cache["symbols"]}
            addon_rows = [meta_map[s] for s in addon if s in meta_map]
            candidates = prev_top + addon_rows
            # de-dup
            seen=set(); cand=[]
            for x in candidates:
                if x["symbol"] in seen: 
                    continue
                seen.add(x["symbol"]); cand.append(x)
            candidates = cand
            scan_label = "INCR"

        # resume state
        progress_key = f"{ymd}:{scan_label}"
        if state.get("key") != progress_key:
            state = {"key": progress_key, "done": [], "results": {}}

        done = set(state.get("done", []))
        results = dict(state.get("results", {}))

        prog = Progress(total=len(candidates), label=f"UniverseScan-{scan_label}", every=50)
        done_count = len(done)

        for row in candidates:
            sym = row["symbol"]
            if sym in done:
                continue
            # load from cache if exists, else fetch
            df = self.storage.load_ohlcv_cache(sym)
            need = int(self.cfg.ohlcv_lookback_days)
            if df is None or len(df) < need:
                try:
                    df_new = self.provider.get_ohlcv(sym, count=max(need, 260))
                    if df_new is not None and len(df_new) >= 30:
                        self.storage.save_ohlcv_cache(sym, df_new)
                        df = df_new
                except Exception:
                    # keep as missing
                    df = None
            liq = self._liquidity_median(df, int(self.cfg.liquidity_window)) if df is not None else 0.0
            results[sym] = {"liquidity": liq, "name": row.get("name",""), "market": row.get("market","")}
            done.add(sym)
            done_count += 1
            prog.tick(done_count, extra=f"last={sym} liq={liq:.0f}")

            # save progress frequently for resume
            if len(done) % 50 == 0:
                state["done"] = sorted(list(done))
                state["results"] = results
                self.storage.save_json("state/universe_progress.json", state)

        # finalize
        ranked = sorted(results.items(), key=lambda kv: kv[1]["liquidity"], reverse=True)
        ranked_rows = [{"symbol": s, **meta} for s, meta in ranked]
        top500 = ranked_rows[: int(self.cfg.top_liquidity)]

        meta = {
            "asof": ymd,
            "scan_label": scan_label,
            "full_scan": full_scan,
            "candidates_scanned": len(candidates),
            "universe_ranked": ranked_rows,
            "generated_at": now_kst_iso(),
            "symbol_cache_asof": ymd
        }
        self.storage.save_json("state/last_universe.json", meta)
        self.storage.save_json("state/universe_progress.json", {"key": progress_key, "done": [], "results": {}})
        return top500, meta
