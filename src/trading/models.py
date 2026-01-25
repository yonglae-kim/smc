from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class TradeSignal:
    timestamp: str
    valid_from: str
    symbol: str
    direction: str
    score: float
    confidence: float
    reasons: List[str]
    gates: Dict[str, bool]
    score_breakdown: Dict[str, float]
    invalidation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EntryPlan:
    entry_type: str
    entry_price: float
    stop_loss: float
    take_profit: float
    rr: float
    expected_return: float
    rationale: List[str]
    invalidation: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExitDecision:
    action: str  # EXIT | PARTIAL | HOLD
    reason: str
    price: Optional[float] = None
    size: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Position:
    symbol: str
    name: str
    market: str
    entry_time: str
    entry_price: float
    size: float
    remaining_size: float
    stop_loss: float
    take_profit: float
    trail: Optional[float]
    exit_rules: Dict[str, Any]
    state: str = "open"  # open | closed | exit_pending
    hold_days: int = 0
    entry_score: float = 0.0
    entry_breakdown: Dict[str, Any] = field(default_factory=dict)
    tp1_price: Optional[float] = None
    tp1_size: float = 0.0
    took_partial: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
