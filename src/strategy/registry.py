from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Set


@dataclass(frozen=True)
class StrategyMeta:
    name: str
    aliases: tuple[str, ...]
    indicators: tuple[str, ...]
    description: str


def _norm_tokens(tokens: Iterable[str]) -> Set[str]:
    return {str(t).strip().lower() for t in tokens if str(t).strip()}


class StrategyRegistry:
    def __init__(self) -> None:
        self._items: Dict[str, StrategyMeta] = {}

    @property
    def names(self) -> List[str]:
        return sorted(self._items.keys())

    def register(self, meta: StrategyMeta) -> bool:
        keys = _norm_tokens((meta.name, *meta.aliases))
        if any(k in self._items for k in keys):
            return False

        indicators = _norm_tokens(meta.indicators)
        for existing in set(self._items.values()):
            if indicators and indicators == _norm_tokens(existing.indicators):
                return False

        for k in keys:
            self._items[k] = meta
        return True


def build_default_registry() -> StrategyRegistry:
    registry = StrategyRegistry()
    registry.register(
        StrategyMeta(
            name="soft_score",
            aliases=("ob_pullback",),
            indicators=("ob", "fvg", "ma20", "ma200", "macd", "rsi", "atr14"),
            description="기존 SMC 소프트스코어 전략",
        )
    )
    registry.register(
        StrategyMeta(
            name="cross_sectional_momentum_trend",
            aliases=("xsmom_trend",),
            indicators=("ret_252", "ret_21", "mom_252_21", "ma200", "atr14"),
            description="12-1 모멘텀 + 장기 추세 게이트",
        )
    )
    return registry
