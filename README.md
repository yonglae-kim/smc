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

## Buy signal logic (winrate-first tightening)
매수 후보는 **동일 날짜 종가 기준** 컨텍스트를 계산한 뒤, **다음 거래일 시가**에 유효하도록 생성됩니다(룩어헤드 방지). 지표/구조(피벗/프랙탈)는 `fractal_n` 만큼 **확정 지연**이 있어 마지막 N개 봉은 피벗 확정에 사용되지 않습니다.(`src/analysis/smc/pivots.py`)  
핵심 로직은 다음 흐름입니다.
1) **Hard-gate**: OB/FVG 존재 여부, 손절 계산 가능 여부, 상방 여유 등 기본 게이트 통과.  
2) **Winrate 타이트닝 게이트** (config 기반):
   - 추세 필터: MA 정배열 + 장기 MA 기울기 상승 여부 확인
   - 변동성 필터: ATR%/ATR ratio로 과도한 변동성 구간 제외
   - 거래량 확인: 평균 대비 거래량 급증 조건
   - RR 최소 조건 (Entry/SL/TP 계산 기반)
   - 최소 확인 수(min_confirmations) 충족
3) **Soft score**: 컨플루언스/레짐/모멘텀/RS 등 가중치 점수로 랭킹.
4) **추가 분석 모듈** (옵션): 상대강도 랭킹, BB 수축 후 돌파 모듈 등을 통과해야 최종 매수 신호로 선택됩니다.

### Backtest profile 예시
동일 기간 비교를 위해 설정 프로필을 제공합니다.
- `config.baseline.yaml`: 타이트닝/모듈 OFF (기본 완화형)
- `config.tight_winrate.yaml`: 타이트닝/모듈 ON (승률 우선형)
