from __future__ import annotations
import time, random
from dataclasses import dataclass
from typing import Optional, Dict, Any
import requests

from .http_cache import HttpCache


class CachedResponse:
    def __init__(self, text: str, encoding: Optional[str] = None, url: str = ""):
        self.text = text
        self.encoding = encoding
        self.status_code = 200
        self.url = url

    def raise_for_status(self) -> None:
        return None

@dataclass
class HttpClient:
    timeout_sec: float
    max_retries: int
    backoff_base_sec: float
    jitter_sec: float
    rate_limit_per_sec: float
    cache: Optional[HttpCache] = None

    _last_ts: float = 0.0

    def _throttle(self):
        if self.rate_limit_per_sec <= 0:
            return
        min_gap = 1.0 / self.rate_limit_per_sec
        now = time.time()
        gap = now - self._last_ts
        if gap < min_gap:
            time.sleep(min_gap - gap + random.random()*self.jitter_sec)
        self._last_ts = time.time()

    def get(self, url: str, params: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str,str]] = None) -> requests.Response:
        headers = headers or {}
        headers.setdefault("User-Agent", "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/122 Safari/537.36")
        last_err = None
        if self.cache and self.cache.mode in ("use", "snapshot"):
            cached = self.cache.load(url, params)
            if cached:
                return CachedResponse(cached.get("text", ""), cached.get("encoding"), url=url)
        for i in range(self.max_retries+1):
            try:
                self._throttle()
                resp = requests.get(url, params=params, headers=headers, timeout=self.timeout_sec)
                # Some Naver finance pages are EUC-KR; requests may mis-detect.
                if "finance.naver.com" in url and resp.encoding is None:
                    resp.encoding = "euc-kr"
                resp.raise_for_status()
                if self.cache and self.cache.mode in ("use", "refresh", "snapshot"):
                    self.cache.save(url, params, resp.text, resp.encoding)
                return resp
            except Exception as e:
                last_err = e
                sleep_s = (self.backoff_base_sec * (2**i)) + random.random()*self.jitter_sec
                time.sleep(sleep_s)
        raise last_err
