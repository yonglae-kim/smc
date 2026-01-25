from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

@dataclass
class Signal:
    symbol: str
    side: str  # LONG only (minimal)
    action: str  # ENTER | EXIT
    reason: str
    score: float = 0.0

class Strategy(ABC):
    @abstractmethod
    def rank(self, date: str, symbol: str, ctx: Dict[str,Any], min_score: Optional[float] = None) -> Optional[Tuple[float,str,Dict[str,Any]]]:
        """Return (score, reason, breakdown) for entry consideration, or None to exclude."""
        raise NotImplementedError
