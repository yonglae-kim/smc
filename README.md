# kquant_smc_reporter
Daily (19:00 KST) batch analyzer for KOSPI+KOSDAQ (Top500 liquidity), generating **candidates + context** (no buy/sell signals) and a single-file HTML report with embedded candlestick charts.

## Quickstart
```bash
python -m venv .venv
source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt

# Run once (generates report under ./out/YYYY-MM-DD/)
python main.py --config config.yaml
```

## Scheduling
### cron (Linux)
Run at 19:00 KST every day:
```cron
0 19 * * * /path/to/.venv/bin/python /path/to/kquant_smc_reporter/main.py --config /path/to/kquant_smc_reporter/config.yaml >> /path/to/logs/kquant.log 2>&1
```
Or use the helper script to register a single cron entry (skips if already present):
```bash
deploy/cron/update_cron.sh
```

### systemd timer (recommended)
See `deploy/systemd/`.

## Notes
- Data source uses unofficial Naver endpoints and HTML scraping; breakage is expected. Swap providers via interfaces in `src/providers`.
- Last N bars include **confirmation lag** for pivots (fractal). Report notes include this.

## Backtest
```bash
python backtest.py --config config.yaml
# output: ./out/backtests/<run_id>/report.html
```

### Non-cached symbols
Backtest will auto-fetch OHLCV for symbols even if they were not previously cached, by requesting a large `count` window and overwriting cache when coverage is insufficient.

## Progress output
The runner/backtest print periodic progress lines (UniverseScan/Analyze/FetchOHLCV/SimDays) to show current status, last symbol processed, and ETA.

## Backtest strategy: SoftScoreStrategy
Backtest uses hard-gate + soft-score + Top-K selection (to avoid zero-trade runs). Tune `backtest.strategy_params` in config.yaml.
