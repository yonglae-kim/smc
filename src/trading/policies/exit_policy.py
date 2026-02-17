from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from ..models import AnalysisContext, ExitDecision, Position
from ...signals.ma_slope_gate import evaluate_ma_slope_gate_from_values, normalize_ma_slope_gate_config


def safe_float(val: Optional[float], fallback: float) -> float:
    if val is None:
        return fallback
    try:
        return float(val)
    except (TypeError, ValueError):
        return fallback


class ExitPolicy:
    def __init__(self, cfg):
        trade = getattr(cfg, "trade", None)
        self.max_hold_days = int(getattr(trade, "max_hold_days", 20))
        self.score_exit_threshold = float(getattr(trade, "score_exit_threshold", 0.0))
        self.exit_on_structure_break = bool(getattr(trade, "exit_on_structure_break", True))
        self.structure_break_quality_min = float(getattr(trade, "structure_break_quality_min", 0.0))
        self.exit_on_score_drop = bool(getattr(trade, "exit_on_score_drop", True))
        self.tp_sl_conflict = str(getattr(trade, "tp_sl_conflict", "conservative"))
        self.trail_atr_mult = float(getattr(trade, "trail_atr_mult", 0.0))
        self.early_exit_rsi_macd_enabled = bool(getattr(trade, "early_exit_rsi_macd_enabled", True))
        self.early_exit_rsi_macd_days = int(getattr(trade, "early_exit_rsi_macd_days", 4))
        self.early_exit_rsi_threshold = float(getattr(trade, "early_exit_rsi_threshold", 45.0))
        self.early_exit_macd_hist_threshold = float(getattr(trade, "early_exit_macd_hist_threshold", -0.02))
        self.early_exit_bear_trend_enabled = bool(getattr(trade, "early_exit_bear_trend_enabled", True))
        self.early_exit_ma20_slope_atr_threshold = float(getattr(trade, "early_exit_ma20_slope_atr_threshold", 0.0))
        self.tp1_risk_reduction_enabled = bool(getattr(trade, "tp1_risk_reduction_enabled", True))
        self.tp1_stop_atr_buffer = float(getattr(trade, "tp1_stop_atr_buffer", 0.5))
        self.tp1_trail_atr_mult = float(getattr(trade, "tp1_trail_atr_mult", 1.0))

        strategy_params = getattr(cfg.backtest, "strategy_params", {}) or {}
        self.ma_slope_gate_cfg = normalize_ma_slope_gate_config(strategy_params.get("ma_slope_gate"))
        self.ma_slope_gate_enabled = bool(self.ma_slope_gate_cfg.get("enabled", True))

    def update_trailing_stop(self, position: Position, ctx: AnalysisContext) -> None:
        if position.trail is None:
            return
        atr = safe_float(ctx.atr14, 0.0)
        if atr <= 0:
            return
        new_stop = safe_float(ctx.close, 0.0) - atr * position.trail
        if new_stop > position.stop_loss:
            position.stop_loss = new_stop

    def apply_tp1_risk_reduction(self, position: Position, ctx: Optional[AnalysisContext]) -> None:
        if not self.tp1_risk_reduction_enabled:
            return
        ctx = ctx or AnalysisContext(symbol=position.symbol, close=position.entry_price)
        atr = safe_float(ctx.atr14, position.entry_atr or 0.0)
        stop_candidates = [position.stop_loss]
        if atr > 0 and self.tp1_stop_atr_buffer > 0:
            stop_candidates.append(position.entry_price + self.tp1_stop_atr_buffer * atr)
        if atr > 0 and self.tp1_trail_atr_mult > 0:
            close_px = safe_float(ctx.close, 0.0)
            if close_px > 0:
                stop_candidates.append(close_px - atr * self.tp1_trail_atr_mult)
        new_stop = max(stop_candidates)
        if new_stop > position.stop_loss:
            position.stop_loss = new_stop
        if self.tp1_trail_atr_mult > 0 and (position.trail is None or self.tp1_trail_atr_mult > position.trail):
            position.trail = self.tp1_trail_atr_mult

    def _extract_recent_series(self, ctx: AnalysisContext, recent_key: str, fallback_key: str) -> List[float]:
        series = getattr(ctx, recent_key, None)
        if isinstance(series, list):
            return [float(v) for v in series if v is not None and math.isfinite(float(v))]
        val = getattr(ctx, fallback_key, None)
        if val is None:
            return []
        try:
            val_f = float(val)
            return [val_f] if math.isfinite(val_f) else []
        except (TypeError, ValueError):
            return []

    def _has_consecutive_weakness(self, ctx: AnalysisContext) -> Tuple[bool, str]:
        days = max(0, self.early_exit_rsi_macd_days)
        if days <= 0:
            return False, ""
        rsi_series = self._extract_recent_series(ctx, "recent_rsi14", "rsi14")
        macd_series = self._extract_recent_series(ctx, "recent_macd_hist", "macd_hist")
        if len(rsi_series) >= days and all(v < self.early_exit_rsi_threshold for v in rsi_series[-days:]):
            return True, f"RSI({self.early_exit_rsi_threshold:.0f}) {days}일 연속 하회"
        if len(macd_series) >= days and all(v < self.early_exit_macd_hist_threshold for v in macd_series[-days:]):
            return True, f"MACD 히스토그램 {days}일 연속 약세"
        return False, ""

    def evaluate_exit(
        self,
        position: Position,
        bar: Dict[str, float],
        ctx: Optional[AnalysisContext],
        eval_ctx: Optional[Dict[str, Any]] = None,
    ) -> List[ExitDecision]:
        decisions: List[ExitDecision] = []
        low_px = float(bar.get("low", bar.get("close", 0.0)))
        high_px = float(bar.get("high", bar.get("close", 0.0)))
        close_px = float(bar.get("close", 0.0))

        eval_ctx = eval_ctx or {}
        stop_grace_active = bool(eval_ctx.get("stop_grace_active", False))
        allow_non_stop_exits_during_stop_grace = bool(eval_ctx.get("allow_non_stop_exits_during_stop_grace", True))

        stop_hit = low_px <= position.stop_loss and not stop_grace_active
        tp_hit = high_px >= position.take_profit

        if stop_grace_active and not allow_non_stop_exits_during_stop_grace:
            return [ExitDecision(action="HOLD", reason="손절 유예기간: 비손절 EXIT 비활성화")]

        if stop_hit and tp_hit:
            if self.tp_sl_conflict == "optimistic":
                return [ExitDecision(action="EXIT", reason="TP/SL 동시 터치: TP 우선(낙관적, optimistic)", price=position.take_profit)]
            return [ExitDecision(action="EXIT", reason="TP/SL 동시 터치: SL 우선(보수적, conservative)", price=position.stop_loss)]

        if stop_hit:
            return [ExitDecision(action="EXIT", reason="손절가 도달(Stop Loss)", price=position.stop_loss)]

        if position.tp1_price and not position.took_partial and high_px >= position.tp1_price:
            decisions.append(ExitDecision(action="PARTIAL", reason="1차 목표가 부분 청산", price=position.tp1_price, size=position.tp1_size))

        if tp_hit:
            decisions.append(ExitDecision(action="EXIT", reason="목표가 도달(전량 익절)", price=position.take_profit))
            return decisions

        if ctx is not None:
            if self.early_exit_rsi_macd_enabled:
                weak_pass, weak_reason = self._has_consecutive_weakness(ctx)
                if weak_pass:
                    return decisions + [ExitDecision(action="EXIT", reason=f"조기 EXIT: {weak_reason}", price=close_px)]

            if self.early_exit_bear_trend_enabled and ctx.ma20 is not None and ctx.ma20_slope_atr is not None:
                try:
                    if close_px < float(ctx.ma20) and float(ctx.ma20_slope_atr) < self.early_exit_ma20_slope_atr_threshold:
                        return decisions + [ExitDecision(action="EXIT", reason="조기 EXIT: MA20 이탈 + 기울기 약세 전환", price=close_px)]
                except (TypeError, ValueError):
                    pass

            if self.ma_slope_gate_enabled:
                gate_pass, reasons, _ = evaluate_ma_slope_gate_from_values(
                    close=ctx.close,
                    ma_fast=ctx.ma_slope_fast if ctx.ma_slope_fast is not None else ctx.ma20,
                    ma_slow=ctx.ma_slope_slow if ctx.ma_slope_slow is not None else ctx.ma200,
                    slope_pct=ctx.ma_slope_pct,
                    side="sell",
                    buy_slope_threshold=float(self.ma_slope_gate_cfg["buy_slope_threshold"]),
                    sell_slope_threshold=float(self.ma_slope_gate_cfg["sell_slope_threshold"]),
                    require_close_confirm_for_buy=bool(self.ma_slope_gate_cfg["require_close_confirm_for_buy"]),
                    require_close_confirm_for_sell=bool(self.ma_slope_gate_cfg["require_close_confirm_for_sell"]),
                )
                if gate_pass:
                    return decisions + [ExitDecision(action="EXIT", reason="MA Slope Hard-Gate: " + "; ".join(reasons), price=close_px)]

        if position.hold_days >= self.max_hold_days:
            return decisions + [ExitDecision(action="EXIT", reason=f"보유기간 만료({self.max_hold_days}일)", price=close_px)]

        if ctx is not None and self.exit_on_structure_break:
            bos = ctx.bos or {}
            bos_dir = bos.get("direction")
            bos_quality = safe_float(bos.get("quality"), 0.0)
            if ctx.structure_bias == "BEAR":
                return decisions + [ExitDecision(action="EXIT", reason="구조 붕괴/하락 전환(BEAR bias)", price=close_px)]
            if bos_dir in ("BEAR", "DOWN") and bos_quality >= self.structure_break_quality_min:
                return decisions + [
                    ExitDecision(action="EXIT", reason=f"구조 붕괴/하락 전환(BOS, q≥{self.structure_break_quality_min:.2f})", price=close_px)
                ]

        if ctx is not None and self.exit_on_score_drop:
            score = float(ctx.soft_score if ctx.soft_score is not None else ctx.score)
            if score < self.score_exit_threshold:
                return decisions + [ExitDecision(action="EXIT", reason=f"점수 하락(임계 {self.score_exit_threshold:.2f} 미만)", price=close_px)]

        return decisions + [ExitDecision(action="HOLD", reason="보유 유지")]

    def build_sell_reasons(self, exit_decisions: List[ExitDecision], position: Position, ctx: Optional[AnalysisContext]) -> List[str]:
        reasons = [d.reason for d in exit_decisions if d.action == "EXIT"]
        if ctx and ctx.structure_bias == "BEAR":
            reasons.append("구조 바이어스 약세 전환.")
        if ctx is not None:
            score_val = ctx.soft_score if ctx.soft_score is not None else ctx.score
            if score_val is not None:
                reasons.append(f"현재 소프트 점수 {float(score_val):.2f}.")
        if position.exit_rules.get("tp_sl_conflict"):
            conflict_map = {"optimistic": "낙관적(TP 우선)", "conservative": "보수적(SL 우선)"}
            conflict_label = conflict_map.get(position.exit_rules["tp_sl_conflict"], position.exit_rules["tp_sl_conflict"])
            reasons.append(f"TP/SL 동시 터치 기준: {conflict_label}.")
        return reasons
