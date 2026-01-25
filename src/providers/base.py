from __future__ import annotations
from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Dict

class DataProvider(ABC):
    @abstractmethod
    def get_ohlcv(self, symbol: str, count: int) -> pd.DataFrame:
        '''Return df with columns: date, open, high, low, close, volume'''
        raise NotImplementedError

    @abstractmethod
    def get_index_ohlc(self, index_code: str, count: int) -> pd.DataFrame:
        '''Return df with columns: date, open, high, low, close, volume(optional)'''
        raise NotImplementedError

class UniverseFetcher(ABC):
    @abstractmethod
    def fetch_all_symbols(self) -> List[Dict]:
        '''Return list of dicts: {symbol, name, market: KOSPI|KOSDAQ}'''
        raise NotImplementedError

    @abstractmethod
    def fetch_top_value_symbols(self, market: str, top_n: int) -> List[str]:
        '''Return symbols by *today's traded value* as a small addon set.'''
        raise NotImplementedError
