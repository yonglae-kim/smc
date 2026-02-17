from __future__ import annotations

import os
import shutil
from typing import Any, Dict, List, Tuple

import pandas as pd

from ..analysis.indicators import atr, rsi, sma
from ..config import load_config
from ..engine import analyze_symbol
from ..providers.naver import NaverChartProvider, NaverMarketSumFetcher
from ..reporting.charts import plot_symbol_chart
from ..reporting.html import render_report
from ..scoring import score_candidate
from ..storage.fs import FSStorage
from ..strategy.soft_score import SoftScoreStrategy
from ..trading.models import EntryPlan, Position, TradeSignal
from ..trading.rules import TradeRules
from ..universe.builder import UniverseBuilder
from ..utils.http import HttpClient
from ..utils.progress import Progress
from ..utils.time import now_kst_iso, today_kst
from .state_keys import POSITIONS_LIVE_KEY, analysis_progress_key


class DailyPipelineService:
    def __init__(self, config_path: str) -> None:
        self.cfg = load_config(config_path)
        self.ymd = today_kst().strftime("%Y-%m-%d")
        os.makedirs(self.cfg.app.cache_dir, exist_ok=True)
        os.makedirs(self.cfg.app.out_dir, exist_ok=True)

        strategy = SoftScoreStrategy(self.cfg)
        self.trade_rules = TradeRules(self.cfg, strategy=strategy)

        cache_mode = self.cfg.network.cache_mode
        snapshot_id = self.cfg.network.cache_snapshot_id or self.ymd
        cache_dir = os.path.join(
            self.cfg.app.cache_dir,
            "http",
            snapshot_id if cache_mode == "snapshot" else "latest",
        )
        from ..utils.http_cache import HttpCache

        http_cache = HttpCache(cache_dir, ttl_sec=self.cfg.network.cache_ttl_sec, mode=cache_mode)
        http = HttpClient(
            timeout_sec=self.cfg.network.timeout_sec,
            max_retries=self.cfg.network.max_retries,
            backoff_base_sec=self.cfg.network.backoff_base_sec,
            jitter_sec=self.cfg.network.jitter_sec,
            rate_limit_per_sec=self.cfg.network.rate_limit_per_sec,
            cache=http_cache,
        )
        self.storage = FSStorage(self.cfg.app.cache_dir)
        self.provider = NaverChartProvider(http)
        self.fetcher = NaverMarketSumFetcher(http)

        self.ctx_map: Dict[str, Dict[str, Any]] = {}
        self.cal_dates = set()

    def build_universe(self) -> Tuple[List[Dict[str, Any]], Dict[str, Any], str]:
        universe_label = (
            "전체 종목"
            if int(self.cfg.universe.top_liquidity) <= 0
            else f"Top{self.cfg.universe.top_liquidity}"
        )
        print(f"[Runner] Universe build: {universe_label} (incremental/weekly policy)", flush=True)
        ub = UniverseBuilder(self.storage, self.provider, self.fetcher, self.cfg.universe)
        universe, uni_meta = ub.build()
        return universe, uni_meta, universe_label

    def load_or_refresh_ohlcv(self, symbol: str) -> pd.DataFrame | None:
        df = self.storage.load_ohlcv_cache(symbol)
        last_date = None
        if df is not None and not df.empty:
            last_date = pd.to_datetime(df["date"], errors="coerce").max()
            if pd.isna(last_date):
                last_date = None
            else:
                last_date = last_date.date()
        stale = last_date is None or last_date < today_kst()

        if df is None or len(df) < self.cfg.universe.ohlcv_lookback_days:
            try:
                df_new = self.provider.get_ohlcv(symbol, count=max(self.cfg.universe.ohlcv_lookback_days, 300))
                if df_new is not None and len(df_new) >= 60:
                    self.storage.save_ohlcv_cache(symbol, df_new)
                    df = df_new
            except Exception:
                return None
        elif stale:
            try:
                df_new = self.provider.get_ohlcv(symbol, count=10)
                if df_new is not None and len(df_new) >= 1:
                    df = pd.concat([df, df_new], ignore_index=True)
                    df["date"] = pd.to_datetime(df["date"], errors="coerce")
                    df = (
                        df.dropna(subset=["date"])
                        .drop_duplicates(subset=["date"])
                        .sort_values("date")
                        .reset_index(drop=True)
                    )
                    self.storage.save_ohlcv_cache(symbol, df)
            except Exception:
                pass
        return df

    def analyze_and_score(self, symbol: str, df: pd.DataFrame, meta: Dict[str, Any]) -> Dict[str, Any] | None:
        ctx = analyze_symbol(meta, df, self.cfg)
        if ctx is None:
            return None

        ctx = score_candidate(ctx, self.cfg.scoring.weights)
        self.cal_dates |= set(df["date"])

        tags = []
        if ctx["structure_bias"] != "NEUTRAL":
            tags.append(f"STRUCT_{ctx['structure_bias']}")
        if ctx.get("bos", {}).get("direction"):
            tags.append(f"BOS_{ctx['bos']['direction']}")
        if ctx.get("ob"):
            tags.append(f"OB_{ctx['ob']['kind']}_{ctx['ob']['status']}")
        if ctx.get("fvg"):
            tags.append(f"FVG_{ctx['fvg']['kind']}_{ctx['fvg']['status']}")
        if ctx.get("tag_confluence_ob_fvg"):
            tags.append("Confluence_OB_FVG")
        tags.append(ctx.get("rs", {}).get("tag", "RS_UNKNOWN"))
        ctx["tags"] = tags
        return ctx

    def reconcile_state(
        self,
        positions: List[Position],
        pending_entries: List[Dict[str, Any]],
        pending_exits: List[Dict[str, Any]],
        cal: List[Any],
    ) -> Tuple[List[Position], List[Dict[str, Any]], List[Dict[str, Any]]]:
        remaining_exits = []
        for pe in pending_exits:
            if pe.get("valid_from") and pe["valid_from"] <= self.ymd:
                positions = [p for p in positions if p.symbol != pe.get("symbol")]
            else:
                remaining_exits.append(pe)
        pending_exits = remaining_exits

        remaining_entries = []
        for pe in pending_entries:
            if pe.get("valid_from") and pe["valid_from"] <= self.ymd:
                sym = pe.get("symbol")
                ctx = self.ctx_map.get(sym)
                if ctx is None:
                    remaining_entries.append(pe)
                    continue
                df = self.storage.load_ohlcv_cache(sym)
                if df is None or df.empty:
                    remaining_entries.append(pe)
                    continue
                row = df[df["date"] == pd.to_datetime(self.ymd)]
                if row.empty:
                    remaining_entries.append(pe)
                    continue
                open_px = float(row["open"].iloc[0])
                low_px = float(row["low"].iloc[0])
                signal = TradeSignal(**pe["signal"])
                entry_plan = EntryPlan(**pe["entry_plan"])
                if low_px > entry_plan.entry_price:
                    remaining_entries.append(pe)
                    continue
                entry_px = open_px if open_px <= entry_plan.entry_price else entry_plan.entry_price
                position = self.trade_rules.build_position(signal, entry_plan, self.ymd, entry_px, 1.0, ctx)
                positions.append(position)
            else:
                remaining_entries.append(pe)
        return positions, remaining_entries, pending_exits

    def assemble_report_rows(
        self,
        rows_sorted: List[Dict[str, Any]],
        signal_rows: List[Dict[str, Any]],
        cal: List[Any],
    ) -> Dict[str, Any]:
        table_rows = []
        for rank, c in enumerate(rows_sorted[: int(self.cfg.report.max_table_rows)], start=1):
            levels = []
            if c.get("ob"):
                levels.append(f"OB[{c['ob']['lower']:.0f}-{c['ob']['upper']:.0f}] inv:{c['ob']['invalidation']:.0f}")
            if c.get("fvg"):
                levels.append(f"FVG[{c['fvg']['lower']:.0f}-{c['fvg']['upper']:.0f}] {c['fvg']['status']}")
            if c.get("bos", {}).get("direction"):
                levels.append(f"BOS:{c['bos']['level']:.0f}")
            table_rows.append(
                {
                    "rank": rank,
                    "score": c.get("score", 0.0),
                    "symbol": c["symbol"],
                    "name": c.get("name", ""),
                    "market": c.get("market", ""),
                    "tags": c.get("tags", []),
                    "close": c.get("close", 0.0),
                    "ma20": c.get("ma20"),
                    "ma200": c.get("ma200"),
                    "ma_slope_pct": c.get("ma_slope_pct"),
                    "rsi14": c.get("rsi14"),
                    "levels": " | ".join(levels),
                }
            )

        signal_map = {r["signal"].symbol: r for r in signal_rows}
        selected = self.trade_rules.select_buy_candidates([(r["signal"], r["entry_plan"]) for r in signal_rows])
        buy_candidates = [signal_map[s[0].symbol] for s in selected]
        immediate_buys = [row for row in buy_candidates if row["entry_plan"].entry_type == "next_open"]
        pullback_buys = [row for row in buy_candidates if row["entry_plan"].entry_type != "next_open"]
        buy_valid_from = self.trade_rules.next_trading_day(cal, self.ymd) if cal else self.ymd

        def build_buy_rows(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            rows = []
            for rank, row in enumerate(candidates, start=1):
                rows.append(
                    {
                        "rank": rank,
                        "symbol": row["signal"].symbol,
                        "name": row["ctx"].get("name", ""),
                        "signal": row["signal"],
                        "entry_plan": row["entry_plan"],
                        "gates": [{"key": k, "pass": v} for k, v in row["signal"].gates.items()],
                    }
                )
            return rows

        return {
            "table_rows": table_rows,
            "buy_candidates": buy_candidates,
            "immediate_buy_rows": build_buy_rows(immediate_buys),
            "pullback_buy_rows": build_buy_rows(pullback_buys),
            "buy_valid_from": buy_valid_from,
        }

    def run(self) -> None:
        print("[Runner] Start daily pipeline", flush=True)
        universe, uni_meta, universe_label = self.build_universe()
        print(f"[Runner] Per-symbol analysis ({universe_label})", flush=True)

        out_dir = self.storage.out_dir(self.cfg.app.out_dir, self.ymd)
        progress_key = analysis_progress_key(self.ymd)
        st = self.storage.load_json(progress_key, default={"done": []})
        done = set(st.get("done", []))
        if done and len(done) >= len(universe):
            print("[Runner] Analysis progress is complete; resetting progress for re-run.", flush=True)
            done = set()

        rows = []
        self.ctx_map = {}
        self.cal_dates = set()
        prog = Progress(total=len(universe), label="Analyze", every=25)
        done_count = len(done)
        min_bars = max(80, int(self.cfg.universe.ohlcv_lookback_days))

        for meta in universe:
            sym = meta["symbol"]
            if sym in done:
                continue
            df = self.load_or_refresh_ohlcv(sym)
            if df is None or len(df) < min_bars:
                done.add(sym)
                done_count += 1
                prog.tick(done_count, extra=f"skip={sym} reason=insufficient_data")
                continue

            ctx = self.analyze_and_score(sym, df, meta)
            if ctx is None:
                done.add(sym)
                done_count += 1
                prog.tick(done_count, extra=f"skip={sym} reason=insufficient_data")
                continue

            rows.append(ctx)
            self.ctx_map[sym] = ctx
            done.add(sym)

            if len(done) % 60 == 0:
                self.storage.save_json(progress_key, {"done": sorted(list(done))})

        self.storage.save_json(progress_key, {"done": []})
        cal = sorted(self.cal_dates)
        rows_sorted = sorted(rows, key=lambda x: (-x.get("score", 0), x.get("symbol", "")))

        signal_rows = []
        for ctx in rows_sorted:
            signal, entry_plan = self.trade_rules.build_signal(
                ctx["asof"], ctx, cal, entry_price=float(ctx.get("close", 0.0))
            )
            ctx["soft_score"] = signal.score
            ctx["soft_score_breakdown"] = signal.score_breakdown
            signal_rows.append({"ctx": ctx, "signal": signal, "entry_plan": entry_plan})

        self.storage.save_json(f"snapshots/{self.ymd}/universe.json", uni_meta)
        self.storage.save_json(f"snapshots/{self.ymd}/candidates.json", rows_sorted)
        self.storage.save_json(
            f"snapshots/{self.ymd}/signals.json",
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
        report_data = self.assemble_report_rows(rows_sorted, signal_rows, cal)

        state = self.storage.load_json(
            POSITIONS_LIVE_KEY,
            default={"positions": [], "pending_entries": [], "pending_exits": [], "last_date": None},
        )
        positions = [Position(**p) for p in state.get("positions", [])]
        pending_entries = list(state.get("pending_entries", []))
        pending_exits = list(state.get("pending_exits", []))

        last_date = state.get("last_date")
        if last_date:
            delta_days = (pd.to_datetime(self.ymd) - pd.to_datetime(last_date)).days
            if delta_days > 0:
                for pos in positions:
                    pos.hold_days += delta_days

        positions, pending_entries, pending_exits = self.reconcile_state(positions, pending_entries, pending_exits, cal)

        chart_mode = str(getattr(self.cfg.report, "chart_image_mode", "inline_base64"))
        use_chart_files = chart_mode == "file_link"
        chart_dir = os.path.join(out_dir, "charts")
        if use_chart_files:
            os.makedirs(chart_dir, exist_ok=True)

        sell_rows = []
        portfolio_rows = []
        sell_details = []
        for pos in positions:
            ctx = self.ctx_map.get(pos.symbol)
            df = self.storage.load_ohlcv_cache(pos.symbol)
            if df is None or df.empty:
                continue
            row = df[df["date"] == pd.to_datetime(self.ymd)]
            if row.empty:
                continue
            bar = {
                "open": float(row["open"].iloc[0]),
                "high": float(row["high"].iloc[0]),
                "low": float(row["low"].iloc[0]),
                "close": float(row["close"].iloc[0]),
            }
            if ctx:
                self.trade_rules.update_trailing_stop(pos, ctx)
            exit_decisions = self.trade_rules.evaluate_exit(pos, bar, ctx, self.ymd)
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
                        "valid_from": self.trade_rules.next_trading_day(cal, self.ymd),
                        "reason": exit_action.reason,
                    }
                )
                pos.state = "exit_pending"
            for decision in exit_decisions:
                if decision.action == "PARTIAL" and decision.size:
                    pos.remaining_size = max(0.0, pos.remaining_size - decision.size)
                    pos.took_partial = True
                    if self.trade_rules.move_stop_to_entry and pos.stop_loss < pos.entry_price:
                        pos.stop_loss = pos.entry_price
                    self.trade_rules.apply_tp1_risk_reduction(pos, ctx)

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
                chart_src = ""
                if ctx:
                    ctx = dict(ctx)
                    ctx["position"] = pos.to_dict()
                    df_chart = self.storage.load_ohlcv_cache(pos.symbol)
                    if df_chart is not None and not df_chart.empty:
                        df_chart = df_chart.sort_values("date").reset_index(drop=True)
                        df_chart["ma20"] = sma(df_chart["close"], int(self.cfg.analysis.ma_fast))
                        df_chart["ma200"] = sma(df_chart["close"], int(self.cfg.analysis.ma_slow))
                        df_chart["rsi14"] = rsi(df_chart["close"], int(self.cfg.analysis.rsi_period))
                        df_chart["atr14"] = atr(df_chart, int(self.cfg.analysis.atr_period))
                        chart_path = os.path.join(chart_dir, f"sell_{pos.symbol}_{self.ymd}.png") if use_chart_files else None
                        chart_payload = plot_symbol_chart(
                            df_chart,
                            ctx,
                            lookback=int(self.cfg.report.chart_lookback),
                            image_mode="file_link" if use_chart_files else "base64",
                            image_path=chart_path,
                        )
                        chart_src = (
                            f"charts/{os.path.basename(chart_payload)}"
                            if use_chart_files
                            else f"data:image/png;base64,{chart_payload}"
                        )
                breakdown = (ctx or {}).get("soft_score_breakdown", {})
                score_text = "\n".join(self.trade_rules.describe_score_breakdown(breakdown)) if breakdown else "(no components)"
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
                        "chart_src": chart_src,
                        "score_text": score_text,
                        "reason_text": "\n".join(self.trade_rules.build_sell_reasons(exit_decisions, pos, ctx or {})),
                    }
                )

        buy_candidates = report_data["buy_candidates"]
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

        state_payload = {
            "positions": [p.to_dict() for p in positions],
            "pending_entries": pending_entries,
            "pending_exits": pending_exits,
            "last_date": self.ymd,
        }
        self.storage.save_json(POSITIONS_LIVE_KEY, state_payload)
        self.storage.save_json(f"snapshots/{self.ymd}/positions.json", state_payload)

        detail_n = int(self.cfg.scoring.top_detail)
        buy_details = []
        for row in buy_candidates[:detail_n]:
            c = dict(row["ctx"])
            c["signal"] = row["signal"].to_dict()
            df = self.storage.load_ohlcv_cache(c["symbol"])
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
            entry_plan_latest = self.trade_rules.build_entry_plan(c, entry_price=float(c.get("close", 0.0)))
            c["entry_plan"] = entry_plan_latest.to_dict()
            df["ma20"] = sma(df["close"], int(self.cfg.analysis.ma_fast))
            df["ma200"] = sma(df["close"], int(self.cfg.analysis.ma_slow))
            df["rsi14"] = rsi(df["close"], int(self.cfg.analysis.rsi_period))
            df["atr14"] = atr(df, int(self.cfg.analysis.atr_period))
            chart_path = os.path.join(chart_dir, f"buy_{c['symbol']}_{self.ymd}.png") if use_chart_files else None
            chart_payload = plot_symbol_chart(
                df,
                c,
                lookback=int(self.cfg.report.chart_lookback),
                image_mode="file_link" if use_chart_files else "base64",
                image_path=chart_path,
            )
            c["chart_src"] = (
                f"charts/{os.path.basename(chart_payload)}"
                if use_chart_files
                else f"data:image/png;base64,{chart_payload}"
            )
            lev = []
            if c.get("bos", {}).get("direction"):
                lev.append(f"BOS {c['bos']['direction']} level={c['bos']['level']:.0f} q={c['bos'].get('quality', 0):.2f}")
            if c.get("ob"):
                ob = c["ob"]
                lev.append(
                    f"OB {ob['kind']} zone=[{ob['lower']:.0f},{ob['upper']:.0f}] inv={ob['invalidation']:.0f} q={ob.get('quality', 0):.2f}"
                )
            if c.get("fvg"):
                fvg = c["fvg"]
                lev.append(
                    f"FVG {fvg['kind']} zone=[{fvg['lower']:.0f},{fvg['upper']:.0f}] status={fvg['status']} age={fvg.get('age', 0)}"
                )
            c["context_text"] = "\n".join(lev) if lev else "(no recent zones detected)"
            c["score_text"] = (
                "\n".join(self.trade_rules.describe_score_breakdown(row["signal"].score_breakdown))
                if row["signal"].score_breakdown
                else "(no components)"
            )
            gate_lines = [f"{k}: {'통과' if v else '실패'}" for k, v in row["signal"].gates.items()]
            if row["signal"].gate_reasons:
                gate_lines.append("---")
                gate_lines.extend(row["signal"].gate_reasons)
            c["gate_text"] = "\n".join(gate_lines)
            plan_reasons = row["entry_plan"].rationale + [f"무효화 조건: {row['entry_plan'].invalidation}"]
            all_reasons = list(row["signal"].reasons) + plan_reasons
            c["reason_text"] = "\n".join(all_reasons) if all_reasons else "(no reasons)"
            buy_details.append(c)

        payload = {
            "title": self.cfg.report.title,
            "generated_at": now_kst_iso(),
            "universe_n": len(universe),
            "liquidity_window": self.cfg.universe.liquidity_window,
            "execution_guide": self.cfg.report.execution_guide,
            "tp_sl_conflict_note": self.cfg.report.tp_sl_conflict_note,
            "buy_valid_from": report_data["buy_valid_from"],
            "table_rows": report_data["table_rows"],
            "immediate_buy_rows": report_data["immediate_buy_rows"],
            "pullback_buy_rows": report_data["pullback_buy_rows"],
            "sell_rows": sell_rows,
            "portfolio_rows": portfolio_rows,
            "buy_details": buy_details,
            "sell_details": sell_details,
            "mobile_light_mode": bool(getattr(self.cfg.report, "mobile_light_mode", True)),
        }
        out_html = os.path.join(out_dir, "report.html")
        render_report(out_html, payload, include_js=bool(self.cfg.report.include_sort_search_js))
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
