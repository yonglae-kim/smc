from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class HttpCache:
    base_dir: str
    ttl_sec: float = 0.0
    mode: str = "use"  # use | refresh | snapshot

    def _key(self, url: str, params: Optional[Dict[str, Any]]) -> str:
        payload = {"url": url, "params": params or {}}
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _path(self, key: str) -> str:
        os.makedirs(self.base_dir, exist_ok=True)
        return os.path.join(self.base_dir, f"{key}.json")

    def load(self, url: str, params: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        key = self._key(url, params)
        fp = self._path(key)
        if not os.path.exists(fp):
            return None
        if self.ttl_sec and self.ttl_sec > 0:
            age = time.time() - os.path.getmtime(fp)
            if age > self.ttl_sec:
                return None
        with open(fp, "r", encoding="utf-8") as f:
            return json.load(f)

    def save(self, url: str, params: Optional[Dict[str, Any]], text: str, encoding: Optional[str]) -> None:
        key = self._key(url, params)
        fp = self._path(key)
        payload = {
            "url": url,
            "params": params or {},
            "text": text,
            "encoding": encoding,
        }
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
