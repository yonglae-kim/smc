from __future__ import annotations

from bisect import bisect_left
from typing import Any, Callable, Iterable, List


def assign_percentile_rank(items: Iterable[Any], value_fn: Callable[[Any], float | None], key: str) -> None:
    values: List[float] = [v for v in (value_fn(item) for item in items) if v is not None]
    if not values:
        for item in items:
            if isinstance(item, dict):
                item[key] = None
        return
    values.sort()
    denom = max(1, len(values) - 1)
    for item in items:
        val = value_fn(item)
        if val is None:
            if isinstance(item, dict):
                item[key] = None
            continue
        rank = bisect_left(values, val)
        pct = rank / denom
        if isinstance(item, dict):
            item[key] = float(pct)
