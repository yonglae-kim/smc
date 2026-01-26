from __future__ import annotations
from typing import Dict, Any, Optional, Tuple

from .base import Strategy
from ..signals.ma_slope_gate import evaluate_ma_slope_gate_from_values, normalize_ma_slope_gate_config

def _bucket_dist(dist: Optional[float], levels: list[tuple[float,float]]) -> float:
    if dist is None:
        return 0.0
    for mx, sc in levels:
        if dist <= mx:
            return float(sc)
    return 0.0

def _bucket_min(val: Optional[float], levels: list[tuple[float,float]]) -> float:
    if val is None:
        return 0.0
    score = 0.0
    for mn, sc in levels:
        if val >= mn:
            score = float(sc)
    return score

class SoftScoreStrategy(Strategy):
    def __init__(self, cfg):
        p = getattr(cfg.backtest, "strategy_params", {}) or {}
        self.threshold = float(p.get("threshold", 3.0))
        self.dist_ob_levels = p.get("dist_ob_levels", [(0.3, 4), (0.6, 2), (1.0, 1)])
        self.dist_fvg_levels = p.get("dist_fvg_levels", [(0.3, 2), (0.6, 1)])
        self.w_confluence = float(p.get("w_confluence", 2.0))
        self.w_regime_tailwind = float(p.get("w_regime_tailwind", 2.0))
        self.w_regime_headwind = float(p.get("w_regime_headwind", -2.0))
        self.w_rs_strong = float(p.get("w_rs_strong", 1.0))
        self.w_rs_weak = float(p.get("w_rs_weak", -1.0))
        self.w_atr_spike = float(p.get("w_atr_spike", -2.0))
        self.w_struct_bull = float(p.get("w_struct_bull", 1.0))
        self.w_struct_bear = float(p.get("w_struct_bear", -1.0))
        self.w_above_ma200 = float(p.get("w_above_ma200", 1.0))
        self.w_above_ma20 = float(p.get("w_above_ma20", 0.5))
        self.w_ma20_above_ma200 = float(p.get("w_ma20_above_ma200", 1.5))
        self.w_rsi_bullish = float(p.get("w_rsi_bullish", 1.0))
        self.w_macd_bullish = float(p.get("w_macd_bullish", 1.5))
        self.w_macd_cross = float(p.get("w_macd_cross", 1.0))
        self.w_volume_surge = float(p.get("w_volume_surge", 0.75))
        self.volume_ratio_threshold = float(p.get("volume_ratio_threshold", 1.3))
        self.min_room_to_high_atr = float(p.get("min_room_to_high_atr", 0.75))
        self.room_to_high_levels = p.get("room_to_high_levels", [(1.5, 0.75), (2.5, 1.5), (4.0, 2.5)])
        self.w_momentum_20 = float(p.get("w_momentum_20", 0.75))
        self.w_momentum_60 = float(p.get("w_momentum_60", 1.5))
        self.w_momentum_60_negative = float(p.get("w_momentum_60_negative", -1.5))
        self.w_ma20_slope = float(p.get("w_ma20_slope", 1.0))
        self.ma20_slope_atr_threshold = float(p.get("ma20_slope_atr_threshold", 0.15))
        self.atr_ratio_low = float(p.get("atr_ratio_low", 0.9))
        self.atr_ratio_high = float(p.get("atr_ratio_high", 1.4))
        self.w_atr_ratio_low = float(p.get("w_atr_ratio_low", 0.75))
        self.w_atr_ratio_high = float(p.get("w_atr_ratio_high", -1.0))
        self.ob_quality_strong = float(p.get("ob_quality_strong", 2.0))
        self.ob_quality_min = float(p.get("ob_quality_min", 1.0))
        self.w_ob_quality_strong = float(p.get("w_ob_quality_strong", 1.0))
        self.w_ob_quality_weak = float(p.get("w_ob_quality_weak", -0.5))
        self.ob_age_max = int(p.get("ob_age_max", 60))
        self.w_ob_age_old = float(p.get("w_ob_age_old", -0.5))
        self.fvg_age_max = int(p.get("fvg_age_max", 60))
        self.w_fvg_age_old = float(p.get("w_fvg_age_old", -0.5))
        self.require_tailwind = bool(p.get("require_tailwind", False))
        self.require_above_ma200 = bool(p.get("require_above_ma200", False))
        self.ma_slope_gate_cfg = normalize_ma_slope_gate_config(p.get("ma_slope_gate"))
        self.ma_slope_gate_enabled = bool(self.ma_slope_gate_cfg.get("enabled", True))
        trade_cfg = getattr(cfg, "trade", None)
        if trade_cfg is not None and getattr(trade_cfg, "min_score", None) is not None:
            self.threshold = max(self.threshold, float(trade_cfg.min_score))

    def _hard_gates(self, ctx: Dict[str, Any]) -> Tuple[Dict[str, bool], list[str], Dict[str, Any]]:
        # Hard Gate
        has_ob = bool(ctx.get("ob"))
        has_fvg = bool(ctx.get("fvg") or ctx.get("fvg_active"))
        gates = {
            "has_zone": bool(has_ob or has_fvg),
        }
        gate_reasons: list[str] = []
        gate_metrics: Dict[str, Any] = {}

        ob = ctx.get("ob") or {}
        invalidation = ob.get("invalidation")
        gates["invalidation_available"] = invalidation is not None or ctx.get("atr14") is not None

        room_to_high_atr = ctx.get("room_to_high_atr")
        gates["room_to_high"] = room_to_high_atr is None or room_to_high_atr >= self.min_room_to_high_atr

        regime = (ctx.get("regime") or {})
        rtag = regime.get("tag")
        gates["regime_tailwind"] = (not self.require_tailwind) or (rtag == "TAILWIND")
        gates["above_ma200"] = (not self.require_above_ma200) or bool(ctx.get("above_ma200"))
        if self.ma_slope_gate_enabled:
            gate_pass, reasons, metrics = evaluate_ma_slope_gate_from_values(
                close=ctx.get("close"),
                ma_fast=ctx.get("ma_slope_fast", ctx.get("ma20")),
                ma_slow=ctx.get("ma_slope_slow", ctx.get("ma200")),
                slope_pct=ctx.get("ma_slope_pct"),
                side="buy",
                buy_slope_threshold=float(self.ma_slope_gate_cfg["buy_slope_threshold"]),
                sell_slope_threshold=float(self.ma_slope_gate_cfg["sell_slope_threshold"]),
                require_close_confirm_for_buy=bool(self.ma_slope_gate_cfg["require_close_confirm_for_buy"]),
                require_close_confirm_for_sell=bool(self.ma_slope_gate_cfg["require_close_confirm_for_sell"]),
            )
            gates["ma_slope_gate"] = gate_pass
            gate_reasons.extend(reasons)
            gate_metrics.update(metrics)
        return gates, gate_reasons, gate_metrics

    def evaluate(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        gates, gate_reasons, gate_metrics = self._hard_gates(ctx)
        breakdown: Dict[str,Any] = {}
        score = 0.0
        has_ob = bool(ctx.get("ob"))
        has_fvg = bool(ctx.get("fvg") or ctx.get("fvg_active"))
        room_to_high_atr = ctx.get("room_to_high_atr")

        s_ob = _bucket_dist(ctx.get("dist_to_ob_atr"), self.dist_ob_levels) if has_ob else 0.0
        s_fvg = _bucket_dist(ctx.get("dist_to_fvg_atr"), self.dist_fvg_levels) if has_fvg else 0.0
        score += s_ob + s_fvg
        breakdown["dist_ob"] = s_ob
        breakdown["dist_fvg"] = s_fvg

        tags = set(ctx.get("tags", []))
        if "Confluence_OB_FVG" in tags:
            score += self.w_confluence
            breakdown["confluence"] = self.w_confluence
        else:
            breakdown["confluence"] = 0.0

        regime = (ctx.get("regime") or {})
        rtag = regime.get("tag")
        if rtag == "TAILWIND":
            score += self.w_regime_tailwind
            breakdown["regime"] = self.w_regime_tailwind
        elif rtag == "HEADWIND":
            score += self.w_regime_headwind
            breakdown["regime"] = self.w_regime_headwind
        else:
            breakdown["regime"] = 0.0

        rs = (ctx.get("rs") or {})
        rst = rs.get("tag")
        if rst == "RS_STRONG":
            score += self.w_rs_strong
            breakdown["rs"] = self.w_rs_strong
        elif rst == "RS_WEAK":
            score += self.w_rs_weak
            breakdown["rs"] = self.w_rs_weak
        else:
            breakdown["rs"] = 0.0

        if regime.get("atr_spike") is True:
            score += self.w_atr_spike
            breakdown["atr_spike"] = self.w_atr_spike
        else:
            breakdown["atr_spike"] = 0.0

        st = ctx.get("structure_bias")
        if st == "BULL":
            score += self.w_struct_bull
            breakdown["structure"] = self.w_struct_bull
        elif st == "BEAR":
            score += self.w_struct_bear
            breakdown["structure"] = self.w_struct_bear
        else:
            breakdown["structure"] = 0.0

        if ctx.get("above_ma200"):
            score += self.w_above_ma200
            breakdown["above_ma200"] = self.w_above_ma200
        else:
            breakdown["above_ma200"] = 0.0

        if ctx.get("above_ma20"):
            score += self.w_above_ma20
            breakdown["above_ma20"] = self.w_above_ma20
        else:
            breakdown["above_ma20"] = 0.0

        if ctx.get("ma20_above_ma200"):
            score += self.w_ma20_above_ma200
            breakdown["ma20_above_ma200"] = self.w_ma20_above_ma200
        else:
            breakdown["ma20_above_ma200"] = 0.0

        rsi = ctx.get("rsi14")
        if rsi is not None and 50 <= rsi <= 70:
            score += self.w_rsi_bullish
            breakdown["rsi_bullish"] = self.w_rsi_bullish
        else:
            breakdown["rsi_bullish"] = 0.0

        macd_hist = ctx.get("macd_hist")
        if macd_hist is not None and macd_hist > 0:
            score += self.w_macd_bullish
            breakdown["macd_bullish"] = self.w_macd_bullish
        else:
            breakdown["macd_bullish"] = 0.0

        macd_line = ctx.get("macd_line")
        macd_signal = ctx.get("macd_signal")
        if macd_line is not None and macd_signal is not None and macd_line > macd_signal:
            score += self.w_macd_cross
            breakdown["macd_cross"] = self.w_macd_cross
        else:
            breakdown["macd_cross"] = 0.0

        volume_ratio = ctx.get("volume_ratio")
        if volume_ratio is not None and volume_ratio >= self.volume_ratio_threshold:
            score += self.w_volume_surge
            breakdown["volume_surge"] = self.w_volume_surge
        else:
            breakdown["volume_surge"] = 0.0

        s_room = _bucket_min(room_to_high_atr, self.room_to_high_levels)
        score += s_room
        breakdown["room_to_high"] = s_room

        momentum_20 = ctx.get("momentum_20")
        if momentum_20 is not None and momentum_20 > 0:
            score += self.w_momentum_20
            breakdown["momentum_20"] = self.w_momentum_20
        else:
            breakdown["momentum_20"] = 0.0

        momentum_60 = ctx.get("momentum_60")
        if momentum_60 is not None and momentum_60 > 0:
            score += self.w_momentum_60
            breakdown["momentum_60"] = self.w_momentum_60
        elif momentum_60 is not None and momentum_60 < 0:
            score += self.w_momentum_60_negative
            breakdown["momentum_60"] = self.w_momentum_60_negative
        else:
            breakdown["momentum_60"] = 0.0

        ma20_slope_atr = ctx.get("ma20_slope_atr")
        if ma20_slope_atr is not None and ma20_slope_atr >= self.ma20_slope_atr_threshold:
            score += self.w_ma20_slope
            breakdown["ma20_slope"] = self.w_ma20_slope
        else:
            breakdown["ma20_slope"] = 0.0

        atr_ratio = ctx.get("atr_ratio")
        if atr_ratio is not None and atr_ratio <= self.atr_ratio_low:
            score += self.w_atr_ratio_low
            breakdown["atr_ratio"] = self.w_atr_ratio_low
        elif atr_ratio is not None and atr_ratio >= self.atr_ratio_high:
            score += self.w_atr_ratio_high
            breakdown["atr_ratio"] = self.w_atr_ratio_high
        else:
            breakdown["atr_ratio"] = 0.0

        ob_quality = ctx.get("ob_quality")
        if ob_quality is not None and ob_quality >= self.ob_quality_strong:
            score += self.w_ob_quality_strong
            breakdown["ob_quality"] = self.w_ob_quality_strong
        elif ob_quality is not None and ob_quality < self.ob_quality_min:
            score += self.w_ob_quality_weak
            breakdown["ob_quality"] = self.w_ob_quality_weak
        else:
            breakdown["ob_quality"] = 0.0

        ob_age = ctx.get("ob_age")
        if ob_age is not None and ob_age > self.ob_age_max:
            score += self.w_ob_age_old
            breakdown["ob_age"] = self.w_ob_age_old
        else:
            breakdown["ob_age"] = 0.0

        fvg_age = ctx.get("fvg_age")
        if fvg_age is not None and fvg_age > self.fvg_age_max:
            score += self.w_fvg_age_old
            breakdown["fvg_age"] = self.w_fvg_age_old
        else:
            breakdown["fvg_age"] = 0.0

        breakdown["total"] = score
        return {
            "score": float(score),
            "breakdown": breakdown,
            "gates": gates,
            "gate_reasons": gate_reasons,
            "gate_metrics": gate_metrics,
            "threshold": float(self.threshold),
        }

    def rank(self, date: str, symbol: str, ctx: Dict[str,Any], min_score: Optional[float] = None) -> Optional[Tuple[float,str,Dict[str,Any]]]:
        eval_result = self.evaluate(ctx)
        gates = eval_result["gates"]
        if gates and not all(gates.values()):
            return None
        score = float(eval_result["score"])
        threshold = float(min_score) if min_score is not None else float(eval_result["threshold"])
        if score < threshold:
            return None
        reason = f"SoftScore>=th({self.threshold}): {score:.1f}"
        return score, reason, eval_result["breakdown"]
