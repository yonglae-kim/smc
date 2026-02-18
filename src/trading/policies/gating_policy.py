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
        self.min_score_regime_non_tailwind_add = float(getattr(trade, "min_score_regime_non_tailwind_add", 0.0))
        self.min_score_regime_headwind_add = float(getattr(trade, "min_score_regime_headwind_add", 0.0))
        self.entry_type_score_add = dict(getattr(trade, "entry_type_score_add", {}) or {})

        self.enable_trend_filter = bool(getattr(trade, "enable_trend_filter", False))
        self.trend_ma_stack = bool(getattr(trade, "trend_ma_stack", True))
        self.trend_slope_atr_min = float(getattr(trade, "trend_slope_atr_min", 0.0))
        self.enable_volatility_filter = bool(getattr(trade, "enable_volatility_filter", False))
        self.max_atr_pct = float(getattr(trade, "max_atr_pct", 0.08))
        self.max_atr_ratio = float(getattr(trade, "max_atr_ratio", 2.2))
        self.enable_volume_confirm = bool(getattr(trade, "enable_volume_confirm", False))
        self.min_volume_ratio = float(getattr(trade, "min_volume_ratio", 1.2))
        self.enable_rs_rank = bool(getattr(trade, "enable_rs_rank", False))
        self.rs_rank_min_pct = float(getattr(trade, "rs_rank_min_pct", 0.5))
        self.enable_bb_squeeze_breakout = bool(getattr(trade, "enable_bb_squeeze_breakout", False))
        self.bb_squeeze_max_width = float(getattr(trade, "bb_squeeze_max_width", 0.12))
        self.min_confirmations = int(getattr(trade, "min_confirmations", 0))

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

        regime_tag = str((ctx.symbol_regime or {}).get("tag", "")).upper()
        if regime_tag and regime_tag != "TAILWIND" and self.min_score_regime_non_tailwind_add > 0:
            min_score += self.min_score_regime_non_tailwind_add
            gate_reasons.append(f"레짐({regime_tag}) 비-테일윈드: 최소 점수 +{self.min_score_regime_non_tailwind_add:.2f}")
        if regime_tag == "HEADWIND" and self.min_score_regime_headwind_add > 0:
            min_score += self.min_score_regime_headwind_add
            gate_reasons.append(f"레짐(HEADWIND): 최소 점수 +{self.min_score_regime_headwind_add:.2f}")

        entry_score_add = safe_float(self.entry_type_score_add.get(entry_plan.entry_type), 0.0)
        if entry_score_add > 0:
            min_score += entry_score_add
            gate_reasons.append(f"진입타입({entry_plan.entry_type}) 최소 점수 +{entry_score_add:.2f}")

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

        if self.enable_trend_filter:
            ma_stack_pass = bool(ctx.ma20_above_ma200 if self.trend_ma_stack else ctx.above_ma200)
            slope_val = safe_float(ctx.ma20_slope_atr, 0.0)
            slope_pass = slope_val >= self.trend_slope_atr_min
            gates["trend_filter"] = ma_stack_pass and slope_pass
            if not gates["trend_filter"]:
                gate_reasons.append("추세 필터 실패")

        if self.enable_volatility_filter:
            atr_pct = safe_float((ctx.atr14 / ctx.close) if (ctx.atr14 is not None and ctx.close) else None, 0.0)
            atr_ratio = safe_float(ctx.atr_ratio, 0.0)
            gates["volatility_filter"] = atr_pct <= self.max_atr_pct and atr_ratio <= self.max_atr_ratio
            if not gates["volatility_filter"]:
                gate_reasons.append("변동성 필터 실패")

        if self.enable_volume_confirm:
            volume_ratio = safe_float(ctx.volume_ratio, 0.0)
            gates["volume_confirm"] = volume_ratio >= self.min_volume_ratio
            if not gates["volume_confirm"]:
                gate_reasons.append("거래량 확인 실패")

        if self.enable_rs_rank:
            rs_pct = safe_float((ctx.rs or {}).get("pct"), -1.0)
            gates["rs_rank"] = rs_pct >= self.rs_rank_min_pct
            if not gates["rs_rank"]:
                gate_reasons.append("RS 랭크 필터 실패")

        if self.enable_bb_squeeze_breakout:
            bb_width = safe_float(ctx.get("bb_width"), float("inf"))
            momentum_ok = safe_float(ctx.momentum_20, 0.0) > 0 or ctx.structure_bias == "BULL"
            gates["bb_squeeze_breakout"] = bb_width <= self.bb_squeeze_max_width and momentum_ok
            if not gates["bb_squeeze_breakout"]:
                gate_reasons.append("BB 스퀴즈 브레이크아웃 필터 실패")

        if self.min_confirmations > 0:
            confirmations = 0
            confirmations += 1 if ctx.structure_bias == "BULL" else 0
            confirmations += 1 if safe_float(ctx.momentum_60, 0.0) > 0 else 0
            confirmations += 1 if safe_float(ctx.ma20_slope_atr, 0.0) > 0 else 0
            confirmations += 1 if safe_float(ctx.volume_ratio, 0.0) >= self.min_volume_ratio else 0
            gates["confirmations"] = confirmations >= self.min_confirmations
            if not gates["confirmations"]:
                gate_reasons.append(f"확인신호 부족({confirmations}/{self.min_confirmations})")

        gates["score_min"] = score >= min_score
        gates["min_rr"] = entry_plan.rr >= self.min_rr
        gates["min_expected_return"] = entry_plan.expected_return >= self.min_expected_return
        return min_score, gates, gate_reasons, breakdown
