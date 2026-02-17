from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ContextBaseModel(BaseModel):
    model_config = ConfigDict(extra="allow")

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def __iter__(self):
        return iter(self.model_dump())


class ZoneContext(ContextBaseModel):
    kind: str
    lower: float
    upper: float
    age: int = 0
    status: Optional[str] = None
    invalidation: Optional[float] = None
    quality: Optional[float] = None
    date: Optional[str] = None
    created_date: Optional[str] = None
    fill_ratio: Optional[float] = None


class AnalysisContext(ContextBaseModel):
    symbol: str
    name: str = ""
    market: str = ""
    asof: str = ""
    close: float
    atr14: Optional[float] = None
    atr50: Optional[float] = None
    atr_ratio: Optional[float] = None
    ma20: Optional[float] = None
    ma200: Optional[float] = None
    ma_slope_fast: Optional[float] = None
    ma_slope_slow: Optional[float] = None
    ma_slope_pct: Optional[float] = None
    ma_slope_window: Optional[int] = None
    above_ma200: bool = False
    above_ma20: bool = False
    ma20_above_ma200: bool = False
    rsi14: Optional[float] = None
    macd_line: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_hist: Optional[float] = None
    volume: Optional[float] = None
    volume_sma20: Optional[float] = None
    volume_ratio: Optional[float] = None
    momentum_20: Optional[float] = None
    momentum_60: Optional[float] = None
    vol_adj_return_20: Optional[float] = None
    ma20_slope_atr: Optional[float] = None
    room_to_high_atr: Optional[float] = None
    recent_high_20: Optional[float] = None
    recent_rsi14: List[float] = Field(default_factory=list)
    recent_macd_hist: List[float] = Field(default_factory=list)
    recent_close: List[float] = Field(default_factory=list)
    recent_ma20: List[float] = Field(default_factory=list)
    recent_ma20_slope_atr: List[float] = Field(default_factory=list)
    structure_bias: str = "NEUTRAL"
    bos: Optional[Dict[str, Any]] = None
    ob: Optional[ZoneContext] = None
    ob_quality: Optional[float] = None
    ob_age: Optional[int] = None
    fvg: Optional[ZoneContext] = None
    fvg_active: bool = False
    fvg_age: Optional[int] = None
    dist_to_ob_atr: Optional[float] = None
    dist_to_fvg_atr: Optional[float] = None
    tag_confluence_ob_fvg: bool = False
    tags: List[str] = Field(default_factory=list)
    rs: Dict[str, Any] = Field(default_factory=dict)
    symbol_regime: Dict[str, Any] = Field(default_factory=dict)
    pivots: List[Dict[str, Any]] = Field(default_factory=list)
    structure_points: List[Dict[str, Any]] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _apply_defaults(self) -> "AnalysisContext":
        if not self.tag_confluence_ob_fvg and "Confluence_OB_FVG" in self.tags:
            self.tag_confluence_ob_fvg = True
        if self.tag_confluence_ob_fvg and "Confluence_OB_FVG" not in self.tags:
            self.tags.append("Confluence_OB_FVG")
        self.fvg_active = self.fvg is not None
        if self.ob is not None:
            self.ob_quality = self.ob_quality if self.ob_quality is not None else self.ob.quality
            self.ob_age = self.ob_age if self.ob_age is not None else self.ob.age
        if self.fvg is not None:
            self.fvg_age = self.fvg_age if self.fvg_age is not None else self.fvg.age
        return self


class ScoredContext(AnalysisContext):
    score: float = 0.0
    score_components: List[Dict[str, Any]] = Field(default_factory=list)
    soft_score: Optional[float] = None
    soft_score_breakdown: Dict[str, float] = Field(default_factory=dict)


class ExitDecisionInput(ContextBaseModel):
    position: "Position"
    bar: Dict[str, float]
    ctx: Optional[AnalysisContext] = None
    date: str
    eval_ctx: Dict[str, Any] = Field(default_factory=dict)


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
    score_breakdown: Dict[str, float] = field(default_factory=dict)
    gate_reasons: List[str] = field(default_factory=list)
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
    entry_type_label: str = ""

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
    entry_stop_loss: Optional[float] = None
    entry_atr: Optional[float] = None
    entry_structure_bias: Optional[str] = None
    stop_distance_atr: Optional[float] = None
    mae: float = 0.0
    mfe: float = 0.0
    tp1_price: Optional[float] = None
    tp1_size: float = 0.0
    took_partial: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
