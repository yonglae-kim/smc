from __future__ import annotations
import os, json
import pandas as pd
from typing import Any, Dict, Optional

class FSStorage:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def _p(self, *parts: str) -> str:
        path = os.path.join(self.base_dir, *parts)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    # --- cache ---
    def load_ohlcv_cache(self, symbol: str) -> Optional[pd.DataFrame]:
        fp = self._p("ohlcv", f"{symbol}.csv")
        if not os.path.exists(fp):
            return None
        df = pd.read_csv(fp, parse_dates=["date"])
        df = df.sort_values("date").reset_index(drop=True)
        return df

    def save_ohlcv_cache(self, symbol: str, df: pd.DataFrame) -> None:
        fp = self._p("ohlcv", f"{symbol}.csv")
        df = df.copy()
        df.to_csv(fp, index=False)

    def save_json(self, relpath: str, obj: Any) -> None:
        fp = self._p(relpath)
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

    def load_json(self, relpath: str, default: Any=None) -> Any:
        fp = self._p(relpath)
        if not os.path.exists(fp):
            return default
        with open(fp, "r", encoding="utf-8") as f:
            return json.load(f)

    # --- snapshot helpers ---
    def snapshot_dir(self, ymd: str) -> str:
        path = os.path.join(self.base_dir, "snapshots", ymd)
        os.makedirs(path, exist_ok=True)
        return path

    def out_dir(self, out_base: str, ymd: str) -> str:
        path = os.path.join(out_base, ymd)
        os.makedirs(path, exist_ok=True)
        return path
