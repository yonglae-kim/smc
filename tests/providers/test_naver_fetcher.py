from __future__ import annotations

import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.providers.naver import NaverMarketSumFetcher


class _Resp:
    def __init__(self, text: str):
        self.text = text


class _FakeHttp:
    def __init__(self, mapping: dict[str, str]):
        self.mapping = mapping

    def get(self, url, params=None, headers=None):
        if params:
            q = "&".join(f"{k}={v}" for k, v in params.items())
            key = f"{url}?{q}"
        else:
            key = url
        return _Resp(self.mapping[key])


def test_fetch_all_symbols_prioritizes_etf_etn_classification():
    html = """
    <html><body>
      <table><tr><td class='pgRR'><a href='?page=1'>맨뒤</a></td></tr></table>
      <a class='tltle' href='/item/main.nhn?code=123456'>ETF 후보</a>
      <a class='tltle' href='/item/main.nhn?code=654321'>ETN 후보</a>
      <a class='tltle' href='/item/main.nhn?code=111111'>일반주</a>
    </body></html>
    """
    mapping = {
        "https://finance.naver.com/api/sise/etfItemList.nhn": json.dumps([{"itemcode": "123456"}]),
        "https://finance.naver.com/api/sise/etnItemList.nhn": json.dumps([{"itemcode": "654321"}]),
        "https://finance.naver.com/sise/sise_market_sum.nhn?sosok=0&page=1": html,
        "https://finance.naver.com/sise/sise_market_sum.nhn?sosok=1&page=1": html,
    }

    fetcher = NaverMarketSumFetcher(_FakeHttp(mapping))

    symbols = fetcher.fetch_all_symbols()
    market_by_symbol = {row["symbol"]: row["market"] for row in symbols}

    assert market_by_symbol["123456"] == "ETF"
    assert market_by_symbol["654321"] == "ETN"
    assert market_by_symbol["111111"] in ("KOSPI", "KOSDAQ")


def test_fetch_all_symbols_supports_wrapped_etn_response():
    html = """
    <html><body>
      <table><tr><td class='pgRR'><a href='?page=1'>맨뒤</a></td></tr></table>
      <a class='tltle' href='/item/main.nhn?code=500027'>ETN wrapped</a>
      <a class='tltle' href='/item/main.nhn?code=123456'>ETF wrapped</a>
      <a class='tltle' href='/item/main.nhn?code=111111'>일반주</a>
    </body></html>
    """
    mapping = {
        "https://finance.naver.com/api/sise/etfItemList.nhn": json.dumps(
            {"resultCode": "success", "result": {"etfItemList": [{"itemcode": "123456"}]}}
        ),
        "https://finance.naver.com/api/sise/etnItemList.nhn": json.dumps(
            {"resultCode": "success", "result": {"etnItemList": [{"itemcode": "500027"}]}}
        ),
        "https://finance.naver.com/sise/sise_market_sum.nhn?sosok=0&page=1": html,
        "https://finance.naver.com/sise/sise_market_sum.nhn?sosok=1&page=1": html,
    }

    fetcher = NaverMarketSumFetcher(_FakeHttp(mapping))

    symbols = fetcher.fetch_all_symbols()
    market_by_symbol = {row["symbol"]: row["market"] for row in symbols}

    assert market_by_symbol["500027"] == "ETN"
    assert market_by_symbol["123456"] == "ETF"
    assert market_by_symbol["111111"] in ("KOSPI", "KOSDAQ")
