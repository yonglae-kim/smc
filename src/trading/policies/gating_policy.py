from __future__ import annotations

from typing import Any, Dict, List, Tuple

from ..models import AnalysisContext, EntryPlan


def safe_float(val: Any, fallback: float) -> float:
    try:
        if val is None:
            return fallback
        return float(val)
    except (TypeError, ValueError):
        return fallback


class GatingPolicy:
    def __init__(self, cfg):
        trade = getattr(cfg, "trade", None)
        self.min_score = float(getattr(trade, "min_score", 0.0))
        self.min_expected_return = float(getattr(trade, "min_expected_return", 0.0))
        self.min_rr = float(getattr(trade, "min_rr", 0.0))
        self.ob_quality_gate_min = float(getattr(trade, "ob_quality_gate_min", 0.0))
        self.ob_quality_gate_penalty = float(getattr(trade, "ob_quality_gate_penalty", 1.0))
        self.ob_age_gate_max = int(getattr(trade, "ob_age_gate_max", 90))
        self.fvg_age_gate_max = int(getattr(trade, "fvg_age_gate_max", 90))
        self.old_zone_penalty = float(getattr(trade, "old_zone_penalty", 0.5))
        self.old_zone_penalty_multiplier = float(getattr(trade, "old_zone_penalty_multiplier", 1.5))

    def apply(
        self,
        ctx: AnalysisContext,
        eval_result: Dict[str, Any],
        entry_plan: EntryPlan,
    ) -> Tuple[float, Dict[str, bool], List[str], Dict[str, float]]:
        score = float(eval_result["score"])
        min_score = max(self.min_score, float(eval_result.get("threshold", 0.0)))
        breakdown = eval_result.get("breakdown", {}) or {}
        structure_score = safe_float(breakdown.get("structure"), 0.0)
        if structure_score < 0:
            min_score += 1.5
        gates = dict(eval_result.get("gates", {}))
        gate_reasons = list(eval_result.get("gate_reasons", []))
        if structure_score < 0:
            gate_reasons.append("구조 점수 음수: 최소 점수 상향")

        ob_quality = ctx.ob_quality
        if ob_quality is not None and ob_quality < self.ob_quality_gate_min:
            min_score += self.ob_quality_gate_penalty
            gate_reasons.append(
                f"OB 품질 낮음({ob_quality:.2f} < {self.ob_quality_gate_min:.2f}): 최소 점수 +{self.ob_quality_gate_penalty:.2f}"
            )
        if ob_quality is not None:
            gates["ob_quality"] = ob_quality >= self.ob_quality_gate_min
            if not gates["ob_quality"]:
                gate_reasons.append("OB 품질 게이트 실패")

        for age_label, age_value, age_max in (("OB", ctx.ob_age, self.ob_age_gate_max), ("FVG", ctx.fvg_age, self.fvg_age_gate_max)):
            if age_value is None or age_max <= 0:
                continue
            if age_value >= age_max:
                penalty = self.old_zone_penalty
                if age_value >= age_max * self.old_zone_penalty_multiplier:
                    penalty *= self.old_zone_penalty_multiplier
                min_score += penalty
                gate_reasons.append(f"{age_label} 노후({int(age_value)} >= {age_max}): 최소 점수 +{penalty:.2f}")

        gates["score_min"] = score >= min_score
        gates["min_rr"] = entry_plan.rr >= self.min_rr
        gates["min_expected_return"] = entry_plan.expected_return >= self.min_expected_return
        return min_score, gates, gate_reasons, breakdown
