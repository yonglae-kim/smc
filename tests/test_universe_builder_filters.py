from __future__ import annotations

from datetime import date

import pandas as pd

from src.universe.builder import UniverseBuilder


class _Cfg:
    top_liquidity = 0
    daily_recalc_top = 10
    weekly_full_scan_weekday = 0
    include_daily_value_rank_addon = 10
    ohlcv_lookback_days = 30
    liquidity_window = 5


class _Storage:
    def __init__(self):
        self.saved = {}

    def load_json(self, path, default=None):
        return self.saved.get(path, default)

    def save_json(self, path, value):
        self.saved[path] = value

    def load_ohlcv_cache(self, symbol):
        return None

    def save_ohlcv_cache(self, symbol, df):
        return None


class _Provider:
    def get_ohlcv(self, symbol, count):
        return pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=40, freq="D"),
                "open": [100] * 40,
                "high": [110] * 40,
                "low": [90] * 40,
                "close": [100] * 40,
                "volume": [1000] * 40,
            }
        )


class _Fetcher:
    def fetch_all_symbols(self):
        return [
            {"symbol": "111111", "name": "일반주", "market": "KOSPI"},
            {"symbol": "222222", "name": "ETF", "market": "ETF"},
            {"symbol": "333333", "name": "ETN", "market": "ETN"},
            {"symbol": "444444", "name": "일반주2", "market": "KOSDAQ"},
        ]

    def fetch_top_value_symbols(self, market, top_n):
        return []


def test_universe_builder_excludes_etf_etn(monkeypatch):
    monkeypatch.setattr("src.universe.builder.today_kst", lambda: date(2024, 1, 1))

    builder = UniverseBuilder(_Storage(), _Provider(), _Fetcher(), _Cfg())

    selected, meta = builder.build()

    assert [row["symbol"] for row in selected] == ["111111", "444444"]
    assert all(row["market"] in {"KOSPI", "KOSDAQ"} for row in meta["universe_ranked"])
