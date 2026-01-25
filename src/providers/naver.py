from __future__ import annotations
import re
import pandas as pd
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup

from .base import DataProvider, UniverseFetcher
from ..utils.http import HttpClient

class NaverChartProvider(DataProvider):
    """Unofficial:
    - Stocks: fchart.stock.naver.com/sise.nhn returns XML with <item data="YYYYMMDD|O|H|L|C|V"/>
    - Index: fallback to finance.naver.com/sise/sise_index_day.nhn?code=KOSPI|KOSDAQ (HTML table)
    """
    def __init__(self, http: HttpClient):
        self.http = http

    def get_ohlcv(self, symbol: str, count: int) -> pd.DataFrame:
        url = "https://fchart.stock.naver.com/sise.nhn"
        params = {"symbol": symbol, "timeframe": "day", "count": str(count), "requestType": "0"}
        resp = self.http.get(url, params=params)
        root = ET.fromstring(resp.text)
        items = []
        for it in root.iter("item"):
            data = it.attrib.get("data")
            if not data:
                continue
            d, o, h, l, c, v = data.split("|")
            items.append([pd.to_datetime(d), int(o), int(h), int(l), int(c), int(v)])
        df = pd.DataFrame(items, columns=["date","open","high","low","close","volume"])
        df = df.sort_values("date").reset_index(drop=True)
        return df

    def get_index_ohlc(self, index_code: str, count: int) -> pd.DataFrame:
        # Naver index daily pages have multiple pages; pull enough pages for count.
        # finance.naver.com uses EUC-KR; requests encoding handling is in HttpClient.
        url = "https://finance.naver.com/sise/sise_index_day.nhn"
        rows = []
        page = 1
        while len(rows) < count and page <= 50:
            resp = self.http.get(url, params={"code": index_code, "page": page})
            html = resp.text
            # Ensure correct decode: if mojibake, caller can override resp.encoding; HttpClient attempts.
            tables = pd.read_html(html)
            if not tables:
                break
            df = tables[0].copy()
            df = df.dropna()
            # Expected columns: 날짜, 종가, 전일비, 시가, 고가, 저가, 거래량
            if "날짜" not in df.columns or "종가" not in df.columns:
                break
            for _, r in df.iterrows():
                date = pd.to_datetime(str(r["날짜"]))
                close = int(str(r["종가"]).replace(",",""))
                open_ = int(str(r.get("시가", close)).replace(",",""))
                high = int(str(r.get("고가", close)).replace(",",""))
                low = int(str(r.get("저가", close)).replace(",",""))
                vol = r.get("거래량", 0)
                try:
                    vol = int(str(vol).replace(",",""))
                except Exception:
                    vol = 0
                rows.append([date, open_, high, low, close, vol])
            page += 1
        df = pd.DataFrame(rows, columns=["date","open","high","low","close","volume"])
        df = df.sort_values("date").drop_duplicates("date").tail(count).reset_index(drop=True)
        return df

class NaverMarketSumFetcher(UniverseFetcher):
    def __init__(self, http: HttpClient):
        self.http = http

    def _market_sum_url(self, sosok: int, page: int) -> str:
        return f"https://finance.naver.com/sise/sise_market_sum.nhn?sosok={sosok}&page={page}"

    def fetch_all_symbols(self):
        out = []
        for sosok, market in [(0,"KOSPI"),(1,"KOSDAQ")]:
            # Find last page via pagination block (pgRR). Fallback to a safe max.
            first = self.http.get(self._market_sum_url(sosok, 1)).text
            soup = BeautifulSoup(first, "lxml")
            last = 1
            pg = soup.select_one("td.pgRR a")
            if pg and pg.get("href"):
                m = re.search(r"page=(\d+)", pg["href"])
                if m:
                    last = int(m.group(1))
            last = min(last, 60)  # safety
            for page in range(1, last+1):
                html = first if page == 1 else self.http.get(self._market_sum_url(sosok, page)).text
                soup = BeautifulSoup(html, "lxml")
                for a in soup.select("a.tltle"):
                    name = a.get_text(strip=True)
                    href = a.get("href","")
                    m = re.search(r"code=(\d+)", href)
                    if not m:
                        continue
                    symbol = m.group(1)
                    out.append({"symbol": symbol, "name": name, "market": market})
        # de-dup
        seen = set()
        uniq=[]
        for x in out:
            if x["symbol"] in seen:
                continue
            seen.add(x["symbol"])
            uniq.append(x)
        return uniq

    def fetch_top_value_symbols(self, market: str, top_n: int):
        sosok = 0 if market=="KOSPI" else 1
        html = self.http.get(self._market_sum_url(sosok, 1)).text
        # Table has '거래대금' column; easier via pandas.read_html
        tables = pd.read_html(html)
        if not tables:
            return []
        df = tables[1] if len(tables) > 1 else tables[0]
        # Find code list from links
        soup = BeautifulSoup(html, "lxml")
        links = soup.select("a.tltle")
        codes=[]
        for a in links:
            m = re.search(r"code=(\d+)", a.get("href",""))
            if m:
                codes.append(m.group(1))
        # codes order matches rows; use top_n directly (already sorted by market cap by default),
        # but we want traded value; if column exists, sort by it.
        if "거래대금" in df.columns:
            s = df["거래대금"].astype(str).str.replace(",","").str.replace("nan","0")
            vals = pd.to_numeric(s, errors="coerce").fillna(0)
            df2 = pd.DataFrame({"code": codes[:len(vals)], "value": vals})
            df2 = df2.sort_values("value", ascending=False)
            return df2["code"].head(top_n).tolist()
        return codes[:top_n]
