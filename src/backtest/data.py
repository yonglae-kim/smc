from __future__ import annotations
from dataclasses import dataclass
import pandas as pd

from ..storage.fs import FSStorage
from ..providers.base import DataProvider

@dataclass
class BacktestDataLoader:
    storage: FSStorage
    provider: DataProvider
    max_fetch_count: int = 6000

    def ensure_symbol(self, symbol: str, warmup_bars: int, start: str, end: str) -> pd.DataFrame:
        """Ensure OHLCV exists even if not cached.
        Naver chart endpoint returns *latest* `count` bars. We therefore fetch a large window that should cover
        (end-start)+warmup. This enables backtesting for previously uncached symbols.
        """
        df = self.storage.load_ohlcv_cache(symbol)
        if df is not None and len(df) > warmup_bars + 60:
            # still verify coverage; if not, refetch large window
            d0 = pd.to_datetime(start)
            d1 = pd.to_datetime(end)
            if df["date"].min() <= d0 and df["date"].max() >= d1:
                return df

        # refetch (overwrite cache)
        count = int(self.max_fetch_count)
        df_new = self.provider.get_ohlcv(symbol, count=count)
        if df_new is not None and len(df_new) >= 60:
            self.storage.save_ohlcv_cache(symbol, df_new)
            return df_new
        return df_new

    def ensure_index(self, index_code: str, count: int) -> pd.DataFrame:
        return self.provider.get_index_ohlc(index_code, count=count)
