from __future__ import annotations

import math
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
    def __init__(self, cfg):
        trade = getattr(cfg, "trade", None)
        self.stop_atr_mult = float(getattr(trade, "stop_atr_mult", 1.5))
        self.min_risk_ratio = float(getattr(trade, "min_risk_ratio", 0.001))
        self.rr_target = float(getattr(trade, "tp_rr_target", 2.0))

    def suggest_entry(self, ctx: AnalysisContext) -> Tuple[str, float, List[str]]:
        close_px = float(ctx.close)
        rationale: List[str] = []
        ob = ctx.ob
        fvg = ctx.fvg
        if ob:
            lower = float(ob.lower)
            upper = float(ob.upper)
            mid = (lower + upper) / 2.0
            if close_px > upper:
                rationale.append("가격이 OB 상단 위에 있어, OB 상단 부근 되돌림 지정가로 접근.")
                return "limit_pullback", upper, rationale
            if lower <= close_px <= upper:
                rationale.append("가격이 OB 구간 내부에 있어, 구간 중앙값 근처 지정가로 접근.")
                return "limit_in_zone", mid, rationale
            rationale.append("가격이 OB 하단 아래라, OB 리클레임 확인 후 진입 권장.")
            return "reclaim", lower, rationale
        if fvg:
            lower = float(fvg.lower)
            upper = float(fvg.upper)
            mid = (lower + upper) / 2.0
            if close_px > upper:
                rationale.append("가격이 FVG 상단 위에 있어, FVG 상단 되돌림 지정가로 접근.")
                return "limit_pullback", upper, rationale
            if lower <= close_px <= upper:
                rationale.append("가격이 FVG 구간 내부에 있어, 구간 중앙값 근처 지정가로 접근.")
                return "limit_in_zone", mid, rationale
            rationale.append("가격이 FVG 하단 아래라, FVG 리클레임 확인 후 진입 권장.")
            return "reclaim", lower, rationale
        rationale.append("명확한 구간이 없어 다음 시가 진입 전략 적용.")
        return "next_open", close_px, rationale

    def build_entry_plan(self, ctx: AnalysisContext, entry_price: float) -> EntryPlan:
        entry_type, suggested_price, rationale = self.suggest_entry(ctx)
        entry_px = entry_price if entry_type == "next_open" else suggested_price
        atr = safe_float(ctx.atr14, entry_px * 0.02)
        ob = ctx.ob
        stop_loss = safe_float(ob.invalidation if ob else None, entry_px - atr * self.stop_atr_mult)
        if not math.isfinite(stop_loss) or stop_loss >= entry_px:
            stop_loss = entry_px * (1 - self.min_risk_ratio)
        min_risk_per_share = entry_px * self.min_risk_ratio
        risk_per_share = max(1e-6, entry_px - stop_loss)
        if risk_per_share < min_risk_per_share:
            risk_per_share = min_risk_per_share
            stop_loss = entry_px - risk_per_share

        rr_target = float(self.rr_target)
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
