import argparse, os
import json
from src.config import load_config
from src.utils.http import HttpClient
from src.providers.naver import NaverChartProvider, NaverMarketSumFetcher
from src.storage.fs import FSStorage
from src.backtest.data import BacktestDataLoader
from src.backtest.engine import run_backtest
from src.backtest.metrics import compute_metrics
from src.backtest.report import render_backtest_report
from src.universe.builder import UniverseBuilder
from src.utils.time import today_kst
from src.utils.progress import Progress
from src.strategy.soft_score import SoftScoreStrategy

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    cache_mode = cfg.network.cache_mode
    snapshot_id = cfg.network.cache_snapshot_id or today_kst().strftime("%Y-%m-%d")
    cache_dir = os.path.join(cfg.app.cache_dir, "http", snapshot_id if cache_mode == "snapshot" else "latest")
    from src.utils.http_cache import HttpCache
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
    loader = BacktestDataLoader(storage, provider, max_fetch_count=int(cfg.backtest.max_fetch_count))

    # universe for backtest
    # TOP500 (liquidity) uses current universe builder; for historical studies you can pin symbols list in config.backtest.symbols
    if cfg.backtest.symbols == "TOP500":
        ub = UniverseBuilder(storage, provider, fetcher, cfg.universe)
        universe, _ = ub.build()
    else:
        all_syms = fetcher.fetch_all_symbols()
        want = set(cfg.backtest.symbols)
        universe = [m for m in all_syms if m["symbol"] in want]

    # ensure index data
    idx_kospi = loader.ensure_index("KOSPI", count=max(int(cfg.regime.index_lookback_days), int(cfg.backtest.max_fetch_count)))
    idx_kosdaq = loader.ensure_index("KOSDAQ", count=max(int(cfg.regime.index_lookback_days), int(cfg.backtest.max_fetch_count)))
    print("[Backtest] idx_kospi rows:", 0 if idx_kospi is None else len(idx_kospi), flush=True)
    print("[Backtest] idx_kosdaq rows:", 0 if idx_kosdaq is None else len(idx_kosdaq), flush=True)
    if (idx_kospi is None or len(idx_kospi)==0) and (idx_kosdaq is None or len(idx_kosdaq)==0):
        print("[Backtest][WARN] Index data is empty. Calendar will fallback to symbol dates; regime/RS will be limited.", flush=True)


    # ensure OHLCV even if not cached (on-demand fetch for each symbol)
    print(f"[Backtest] Ensuring OHLCV for {len(universe)} symbols (auto-fetch if missing)", flush=True)
    ohlcv_map={}
    prog = Progress(total=len(universe), label="FetchOHLCV", every=20)
    done = 0
    for meta in universe:
        sym = meta["symbol"]
        df = loader.ensure_symbol(sym, warmup_bars=int(cfg.backtest.warmup_bars), start=cfg.backtest.start, end=cfg.backtest.end)
        ohlcv_map[sym]=df
        done += 1
        prog.tick(done, extra=f"last={sym} rows={(0 if df is None else len(df))}")

    run_id = f"{cfg.backtest.strategy}_{cfg.backtest.start}_{cfg.backtest.end}"
    out_dir = os.path.join(cfg.app.out_dir, "backtests", run_id)
    os.makedirs(out_dir, exist_ok=True)

    last_state = {"equity": None, "trades": None, "positions": None}
    result_path = os.path.join(out_dir, "result.json")

    def on_update(partial_result, update_meta):
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(partial_result, f, ensure_ascii=False, indent=2)

        changed = (
            last_state["equity"] != update_meta["equity"]
            or last_state["trades"] != update_meta["trades"]
            or last_state["positions"] != update_meta["positions"]
        )
        if changed:
            print(
                "[Backtest][Update]",
                {
                    "date": update_meta["date"],
                    "equity": round(update_meta["equity"], 2),
                    "positions": update_meta["positions"],
                    "trades": update_meta["trades"],
                },
                flush=True,
            )
            last_state.update(
                {
                    "equity": update_meta["equity"],
                    "trades": update_meta["trades"],
                    "positions": update_meta["positions"],
                }
            )

    print("[Backtest] Running simulation", flush=True)
    strat = SoftScoreStrategy(cfg)
    result = run_backtest(universe, ohlcv_map, idx_kospi, idx_kosdaq, cfg, strat, on_update=on_update)
    metrics = compute_metrics(result)

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    render_backtest_report(os.path.join(out_dir, "report.html"), {
        "title": f"Backtest Report - {cfg.backtest.strategy}",
        "start": result["start"],
        "end": result["end"],
        "metrics": metrics,
        "equity_curve": result.get("equity_curve", []),
        "trades": result.get("trades", [])[:2000],
    })
    print("Backtest written:", out_dir)

if __name__ == "__main__":
    main()
