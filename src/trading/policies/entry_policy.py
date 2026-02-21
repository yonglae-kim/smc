from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ..models import AnalysisContext, EntryPlan


def safe_float(val: Optional[float], fallback: float) -> float:
    if val is None:
        return fallback
    try:
        return float(val)
    except (TypeError, ValueError):
        return fallback


class EntryPolicy:
    @staticmethod
    def _as_ctx(ctx: AnalysisContext | dict) -> AnalysisContext:
        if isinstance(ctx, AnalysisContext):
            return ctx
        return AnalysisContext.model_validate(ctx)

    def __init__(self, cfg):
        trade = getattr(cfg, "trade", None)
        strategy_params = getattr(cfg.backtest, "strategy_params", {}) or {}
        self.stop_atr_mult = float(getattr(trade, "stop_atr_mult", 1.5))
        self.min_risk_ratio = float(getattr(trade, "min_risk_ratio", 0.001))
        self.rr_target = float(getattr(trade, "tp_rr_target", 2.0))
        self.momentum_breakout_enabled = bool(strategy_params.get("momentum_breakout_entry_enabled", False))
        self.momentum_breakout_tick = float(strategy_params.get("momentum_breakout_tick", 0.0))
        self.momentum_stop_atr_mult = float(strategy_params.get("momentum_stop_atr_mult", 2.5))
        self.momentum_rr_target = float(strategy_params.get("momentum_rr_target", 2.0))

    @dataclass
    class _EntryCandidate:
        entry_type: str
        price: float
        rationale: str
        recovery_score: float

    def _build_zone_candidate(self, zone_name: str, zone, close_px: float) -> "EntryPolicy._EntryCandidate":
        lower = float(zone.lower)
        upper = float(zone.upper)
        mid = (lower + upper) / 2.0
        zone_quality = safe_float(getattr(zone, "quality", None), 0.0)
        zone_age = max(0.0, safe_float(getattr(zone, "age", None), 0.0))

        if close_px > upper:
            distance = max(0.0, close_px - upper)
            recovery_score = -distance + zone_quality * 0.5 - zone_age * 0.05
            return self._EntryCandidate(
                entry_type="limit_pullback",
                price=upper,
                rationale=f"가격이 {zone_name} 상단 위에 있어, {zone_name} 상단 부근 되돌림 지정가로 접근.",
                recovery_score=recovery_score,
            )
        if lower <= close_px <= upper:
            recovery_score = 3.0 + zone_quality * 0.5 - zone_age * 0.05
            return self._EntryCandidate(
                entry_type="limit_in_zone",
                price=mid,
                rationale=f"가격이 {zone_name} 구간 내부에 있어, 구간 중앙값 근처 지정가로 접근.",
                recovery_score=recovery_score,
            )

        distance = max(0.0, lower - close_px)
        recovery_score = -distance * 1.1 + zone_quality * 0.3 - zone_age * 0.08
        return self._EntryCandidate(
            entry_type="reclaim",
            price=lower,
            rationale=f"가격이 {zone_name} 하단 아래라, {zone_name} 리클레임 확인 후 진입 권장.",
            recovery_score=recovery_score,
        )

    def _build_breakout_candidate(self, ctx: AnalysisContext, close_px: float) -> Optional["EntryPolicy._EntryCandidate"]:
        if not self.momentum_breakout_enabled or ctx.recent_high_20 is None:
            return None
        breakout_px = float(ctx.recent_high_20) + self.momentum_breakout_tick
        distance = abs(close_px - breakout_px)
        momentum_20 = safe_float(ctx.momentum_20, 0.0)
        momentum_60 = safe_float(ctx.momentum_60, 0.0)

        recovery_score = 1.6 - distance * 0.8
        if momentum_20 > 0:
            recovery_score += 0.25
        if momentum_60 > 0:
            recovery_score += 0.25

        return self._EntryCandidate(
            entry_type="breakout_20",
            price=breakout_px,
            rationale="20일 고점 돌파 + tick 진입 규칙 후보 평가.",
            recovery_score=recovery_score,
        )

    def suggest_entry(self, ctx: AnalysisContext | dict) -> Tuple[str, float, List[str]]:
        ctx = self._as_ctx(ctx)
        close_px = float(ctx.close)
        rationale: List[str] = []
        candidates: List[EntryPolicy._EntryCandidate] = []

        if ctx.ob:
            candidates.append(self._build_zone_candidate("OB", ctx.ob, close_px))
        if ctx.fvg:
            candidates.append(self._build_zone_candidate("FVG", ctx.fvg, close_px))
        breakout_candidate = self._build_breakout_candidate(ctx, close_px)
        if breakout_candidate is not None:
            candidates.append(breakout_candidate)

        if candidates:
            best = max(candidates, key=lambda item: item.recovery_score)
            rationale.append(best.rationale)
            if len(candidates) > 1:
                rationale.append("복수 진입 후보 중 복귀 가능성이 높은 구간을 우선 선택.")
            return best.entry_type, best.price, rationale

        rationale.append("명확한 구간이 없어 다음 시가 진입 전략 적용.")
        return "next_open", close_px, rationale

    def build_entry_plan(self, ctx: AnalysisContext | dict, entry_price: float) -> EntryPlan:
        ctx = self._as_ctx(ctx)
        entry_type, suggested_price, rationale = self.suggest_entry(ctx)
        entry_px = entry_price if entry_type == "next_open" else suggested_price

        atr = safe_float(ctx.atr14, entry_px * 0.02)
        ob = ctx.ob
        is_breakout_entry = entry_type == "breakout_20"
        stop_mult = self.momentum_stop_atr_mult if is_breakout_entry else self.stop_atr_mult
        stop_loss = safe_float(ob.invalidation if ob else None, entry_px - atr * stop_mult)
        if not math.isfinite(stop_loss) or stop_loss >= entry_px:
            stop_loss = entry_px * (1 - self.min_risk_ratio)
        min_risk_per_share = entry_px * self.min_risk_ratio
        risk_per_share = max(1e-6, entry_px - stop_loss)
        if risk_per_share < min_risk_per_share:
            risk_per_share = min_risk_per_share
            stop_loss = entry_px - risk_per_share

        rr_target = float(self.momentum_rr_target if is_breakout_entry else self.rr_target)
        atr_ratio = ctx.atr_ratio
        room_to_high_atr = ctx.room_to_high_atr
        momentum_20 = ctx.momentum_20
        momentum_60 = ctx.momentum_60
        if atr_ratio is not None and math.isfinite(float(atr_ratio)):
            if atr_ratio >= 1.4:
                rr_target += 0.25
            elif atr_ratio >= 1.1:
                rr_target += 0.1
            elif atr_ratio <= 0.9:
                rr_target -= 0.15
        if room_to_high_atr is not None and math.isfinite(float(room_to_high_atr)):
            if room_to_high_atr >= 2.5:
                rr_target += 0.4
            elif room_to_high_atr >= 1.5:
                rr_target += 0.2
            elif room_to_high_atr < 0.75:
                rr_target -= 0.5
        if momentum_20 is not None and math.isfinite(float(momentum_20)) and momentum_20 > 0:
            rr_target += 0.15
        if momentum_60 is not None and math.isfinite(float(momentum_60)):
            rr_target += 0.2 if momentum_60 > 0 else -0.2 if momentum_60 < 0 else 0.0

        rr_target = min(3.0, max(1.2, rr_target))
        take_profit = entry_px + rr_target * risk_per_share
        rr = (take_profit - entry_px) / risk_per_share if risk_per_share > 0 else 0.0
        expected_return = (take_profit - entry_px) / max(entry_px, 1e-6)
        invalidation = "종가가 손절가 하회 또는 구조 붕괴 시 시나리오 무효."
        if ob:
            invalidation = f"종가가 OB 무효화 가격({stop_loss:.0f}) 하회 시 시나리오 무효."
        label_map: Dict[str, str] = {
            "limit_pullback": "되돌림 지정가",
            "limit_in_zone": "구간 내부 지정가",
            "reclaim": "리클레임 확인 진입",
            "next_open": "다음 시가 진입",
            "breakout_20": "20일 고점 돌파 진입",
        }
        return EntryPlan(
            entry_type=entry_type,
            entry_type_label=label_map.get(entry_type, entry_type),
            entry_price=float(entry_px),
            stop_loss=float(stop_loss),
            take_profit=float(take_profit),
            rr=float(rr),
            expected_return=float(expected_return),
            rationale=rationale,
            invalidation=invalidation,
        )
