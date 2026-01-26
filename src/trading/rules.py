from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

import math

import pandas as pd

from .models import EntryPlan, ExitDecision, Position, TradeSignal
from ..signals.ma_slope_gate import evaluate_ma_slope_gate_from_values, normalize_ma_slope_gate_config
from ..strategy.soft_score import SoftScoreStrategy


def _safe_float(val: Optional[float], fallback: float) -> float:
    if val is None:
        return fallback
    try:
        return float(val)
    except (TypeError, ValueError):
        return fallback


class TradeRules:
    def __init__(self, cfg, strategy: Optional[SoftScoreStrategy] = None):
        self.cfg = cfg
        self.strategy = strategy or SoftScoreStrategy(cfg)
        trade = getattr(cfg, "trade", None)
        backtest = getattr(cfg, "backtest", None)
        self.execution_delay_days = int(getattr(trade, "execution_delay_days", 1))
        self.entry_price_mode = str(getattr(trade, "entry_price_mode", "next_open"))
        if backtest is not None and getattr(backtest, "fill_price", None):
            self.entry_price_mode = str(getattr(trade, "entry_price_mode", backtest.fill_price))
        self.force_top_k = int(getattr(trade, "force_top_k", 0))
        self.min_score = float(getattr(trade, "min_score", 0.0))
        self.min_expected_return = float(getattr(trade, "min_expected_return", 0.0))
        self.min_rr = float(getattr(trade, "min_rr", 0.0))
        self.stop_atr_mult = float(getattr(trade, "stop_atr_mult", 1.5))
        self.min_risk_ratio = float(getattr(trade, "min_risk_ratio", 0.001))
        self.rr_target = float(getattr(trade, "tp_rr_target", 2.0))
        self.partial_rr = float(getattr(trade, "tp_partial_rr", 1.0))
        self.partial_size = float(getattr(trade, "tp_partial_size", 0.0))
        self.move_stop_to_entry = bool(getattr(trade, "move_stop_to_entry", True))
        self.max_hold_days = int(getattr(trade, "max_hold_days", 20))
        self.score_exit_threshold = float(getattr(trade, "score_exit_threshold", 0.0))
        self.exit_on_structure_break = bool(getattr(trade, "exit_on_structure_break", True))
        self.exit_on_score_drop = bool(getattr(trade, "exit_on_score_drop", True))
        self.tp_sl_conflict = str(getattr(trade, "tp_sl_conflict", "conservative"))
        self.trail_atr_mult = float(getattr(trade, "trail_atr_mult", 0.0))
        strategy_params = getattr(cfg.backtest, "strategy_params", {}) or {}
        self.ma_slope_gate_cfg = normalize_ma_slope_gate_config(strategy_params.get("ma_slope_gate"))
        self.ma_slope_gate_enabled = bool(self.ma_slope_gate_cfg.get("enabled", True))

    def next_trading_day(self, calendar: Iterable[pd.Timestamp], date: str) -> str:
        if not calendar:
            return date
        dt = pd.to_datetime(date)
        cal = list(calendar)
        for c in cal:
            if c > dt:
                return str(c.date())
        return str(cal[-1].date())

    def _entry_suggestion(self, ctx: Dict[str, Any]) -> Tuple[str, float, List[str]]:
        close_px = float(ctx.get("close", 0.0))
        rationale = []
        ob = ctx.get("ob") or {}
        fvg = ctx.get("fvg") or {}
        if ob:
            lower = float(ob.get("lower", close_px))
            upper = float(ob.get("upper", close_px))
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
            lower = float(fvg.get("lower", close_px))
            upper = float(fvg.get("upper", close_px))
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

    def build_entry_plan(self, ctx: Dict[str, Any], entry_price: float) -> EntryPlan:
        entry_type, suggested_price, rationale = self._entry_suggestion(ctx)
        entry_px = entry_price if entry_type == "next_open" else suggested_price
        atr = _safe_float(ctx.get("atr14"), entry_px * 0.02)
        ob = ctx.get("ob") or {}
        stop_loss = _safe_float(ob.get("invalidation"), entry_px - atr * self.stop_atr_mult)
        if not math.isfinite(stop_loss):
            stop_loss = entry_px - atr * self.stop_atr_mult
        if not math.isfinite(stop_loss) or stop_loss >= entry_px:
            stop_loss = entry_px * (1 - self.min_risk_ratio)
        risk_per_share = max(1e-6, entry_px - stop_loss)
        take_profit = entry_px + self.rr_target * risk_per_share
        rr = (take_profit - entry_px) / risk_per_share if risk_per_share > 0 else 0.0
        expected_return = (take_profit - entry_px) / max(entry_px, 1e-6)
        invalidation = "종가가 손절가 하회 또는 구조 붕괴 시 시나리오 무효."
        if ob:
            invalidation = f"종가가 OB 무효화 가격({stop_loss:.0f}) 하회 시 시나리오 무효."
        entry_type_label_map = {
            "limit_pullback": "되돌림 지정가",
            "limit_in_zone": "구간 내부 지정가",
            "reclaim": "리클레임 확인 진입",
            "next_open": "다음 시가 진입",
        }
        return EntryPlan(
            entry_type=entry_type,
            entry_type_label=entry_type_label_map.get(entry_type, entry_type),
            entry_price=float(entry_px),
            stop_loss=float(stop_loss),
            take_profit=float(take_profit),
            rr=float(rr),
            expected_return=float(expected_return),
            rationale=rationale,
            invalidation=invalidation,
        )

    def build_signal(self, date: str, ctx: Dict[str, Any], calendar: Iterable[pd.Timestamp], entry_price: float) -> Tuple[TradeSignal, EntryPlan]:
        eval_result = self.strategy.evaluate(ctx)
        entry_plan = self.build_entry_plan(ctx, entry_price)
        score = float(eval_result["score"])
        min_score = max(self.min_score, float(eval_result.get("threshold", 0.0)))
        gates = dict(eval_result.get("gates", {}))
        gate_reasons = list(eval_result.get("gate_reasons", []))
        gates["score_min"] = score >= min_score
        gates["min_rr"] = entry_plan.rr >= self.min_rr
        gates["min_expected_return"] = entry_plan.expected_return >= self.min_expected_return
        all_pass = all(gates.values()) if gates else False
        confidence = min(1.0, score / max(min_score * 2.0, 1e-6)) if min_score > 0 else 0.0
        reasons = self._build_buy_reasons(ctx, eval_result, entry_plan, all_pass)
        valid_from = self.next_trading_day(calendar, date)
        signal = TradeSignal(
            timestamp=date,
            valid_from=valid_from,
            symbol=ctx.get("symbol", ""),
            direction="BUY",
            score=score,
            confidence=float(confidence),
            reasons=reasons,
            gates=gates,
            gate_reasons=gate_reasons,
            score_breakdown=eval_result.get("breakdown", {}),
            invalidation=entry_plan.invalidation,
        )
        return signal, entry_plan

    def signal_passes(self, signal: TradeSignal) -> bool:
        return all(signal.gates.values()) if signal.gates else False

    def select_buy_candidates(self, signals: List[Tuple[TradeSignal, EntryPlan]]) -> List[Tuple[TradeSignal, EntryPlan]]:
        passing = [s for s in signals if self.signal_passes(s[0])]
        passing.sort(key=lambda x: (-x[0].score, x[0].symbol))
        if self.force_top_k > 0:
            ranked = sorted(signals, key=lambda x: (-x[0].score, x[0].symbol))
            forced = ranked[: self.force_top_k]
            merged = passing + [s for s in forced if s[0].symbol not in {p[0].symbol for p in passing}]
            merged.sort(key=lambda x: (-x[0].score, x[0].symbol))
            return merged
        return passing

    def build_position(
        self,
        signal: TradeSignal,
        entry_plan: EntryPlan,
        entry_date: str,
        entry_price: float,
        size: float,
        ctx: Dict[str, Any],
    ) -> Position:
        stop_loss = entry_plan.stop_loss
        if stop_loss >= entry_price:
            stop_loss = entry_price * (1 - self.min_risk_ratio)
        take_profit = entry_plan.take_profit
        if take_profit <= entry_price:
            risk = max(1e-6, entry_price - stop_loss)
            take_profit = entry_price + self.rr_target * risk
        exit_rules = {
            "max_hold_days": self.max_hold_days,
            "score_exit_threshold": self.score_exit_threshold,
            "exit_on_structure_break": self.exit_on_structure_break,
            "exit_on_score_drop": self.exit_on_score_drop,
            "tp_sl_conflict": self.tp_sl_conflict,
        }
        tp1_price = None
        tp1_size = 0.0
        if self.partial_size > 0 and self.partial_rr > 0:
            risk_per_share = max(1e-6, entry_price - entry_plan.stop_loss)
            tp1_price = entry_price + self.partial_rr * risk_per_share
            tp1_size = size * self.partial_size
        atr = _safe_float(ctx.get("atr14"), 0.0)
        stop_distance_atr = None
        if atr > 0:
            stop_distance_atr = (entry_price - stop_loss) / atr
        return Position(
            symbol=signal.symbol,
            name=ctx.get("name", ""),
            market=ctx.get("market", ""),
            entry_time=entry_date,
            entry_price=entry_price,
            size=size,
            remaining_size=size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            trail=self.trail_atr_mult if self.trail_atr_mult > 0 else None,
            exit_rules=exit_rules,
            state="open",
            entry_score=signal.score,
            entry_breakdown=signal.score_breakdown,
            entry_stop_loss=stop_loss,
            entry_atr=atr if atr > 0 else None,
            entry_regime_tag=(ctx.get("regime") or {}).get("tag"),
            entry_structure_bias=ctx.get("structure_bias"),
            stop_distance_atr=stop_distance_atr,
            tp1_price=tp1_price,
            tp1_size=tp1_size,
        )

    def update_trailing_stop(self, position: Position, ctx: Dict[str, Any]) -> None:
        if position.trail is None:
            return
        atr = _safe_float(ctx.get("atr14"), 0.0)
        if atr <= 0:
            return
        close_px = _safe_float(ctx.get("close"), 0.0)
        new_stop = close_px - atr * position.trail
        if new_stop > position.stop_loss:
            position.stop_loss = new_stop

    def evaluate_exit(
        self,
        position: Position,
        bar: Dict[str, float],
        ctx: Optional[Dict[str, Any]],
        date: str,
    ) -> List[ExitDecision]:
        decisions: List[ExitDecision] = []
        low_px = float(bar.get("low", bar.get("close", 0.0)))
        high_px = float(bar.get("high", bar.get("close", 0.0)))
        close_px = float(bar.get("close", 0.0))

        stop_hit = low_px <= position.stop_loss
        tp_hit = high_px >= position.take_profit

        if stop_hit and tp_hit:
            if self.tp_sl_conflict == "optimistic":
                decisions.append(
                    ExitDecision(
                        action="EXIT",
                        reason="TP/SL 동시 터치: TP 우선(낙관적, optimistic)",
                        price=position.take_profit,
                    )
                )
                return decisions
            decisions.append(
                ExitDecision(
                    action="EXIT",
                    reason="TP/SL 동시 터치: SL 우선(보수적, conservative)",
                    price=position.stop_loss,
                )
            )
            return decisions

        if stop_hit:
            decisions.append(ExitDecision(action="EXIT", reason="손절가 도달(Stop Loss)", price=position.stop_loss))
            return decisions

        if position.tp1_price and not position.took_partial and high_px >= position.tp1_price:
            decisions.append(
                ExitDecision(
                    action="PARTIAL",
                    reason="1차 목표가 부분 청산",
                    price=position.tp1_price,
                    size=position.tp1_size,
                )
            )

        if tp_hit:
            decisions.append(ExitDecision(action="EXIT", reason="목표가 도달(전량 익절)", price=position.take_profit))
            return decisions

        if ctx and self.ma_slope_gate_enabled:
            gate_pass, reasons, _ = evaluate_ma_slope_gate_from_values(
                close=ctx.get("close"),
                ma_fast=ctx.get("ma_slope_fast", ctx.get("ma20")),
                ma_slow=ctx.get("ma_slope_slow", ctx.get("ma200")),
                slope_pct=ctx.get("ma_slope_pct"),
                side="sell",
                buy_slope_threshold=float(self.ma_slope_gate_cfg["buy_slope_threshold"]),
                sell_slope_threshold=float(self.ma_slope_gate_cfg["sell_slope_threshold"]),
                require_close_confirm_for_buy=bool(self.ma_slope_gate_cfg["require_close_confirm_for_buy"]),
                require_close_confirm_for_sell=bool(self.ma_slope_gate_cfg["require_close_confirm_for_sell"]),
            )
            if gate_pass:
                decisions.append(
                    ExitDecision(
                        action="EXIT",
                        reason="MA Slope Hard-Gate: " + "; ".join(reasons),
                        price=close_px,
                    )
                )
                return decisions

        if position.hold_days >= self.max_hold_days:
            decisions.append(ExitDecision(action="EXIT", reason=f"보유기간 만료({self.max_hold_days}일)", price=close_px))
            return decisions

        if ctx is not None:
            if self.exit_on_structure_break:
                bos_dir = (ctx.get("bos") or {}).get("direction")
                if ctx.get("structure_bias") == "BEAR" or bos_dir == "DOWN":
                    decisions.append(ExitDecision(action="EXIT", reason="구조 붕괴/하락 전환(BOS)", price=close_px))
                    return decisions

            if self.exit_on_score_drop:
                score = float(ctx.get("soft_score", ctx.get("score", 0.0)))
                if score < self.score_exit_threshold:
                    decisions.append(
                        ExitDecision(
                            action="EXIT",
                            reason=f"점수 하락(임계 {self.score_exit_threshold:.2f} 미만)",
                            price=close_px,
                        )
                    )
                    return decisions

        decisions.append(ExitDecision(action="HOLD", reason="보유 유지"))
        return decisions

    def _build_buy_reasons(
        self,
        ctx: Dict[str, Any],
        eval_result: Dict[str, Any],
        entry_plan: EntryPlan,
        all_pass: bool,
    ) -> List[str]:
        reasons = []
        if not all_pass:
            reasons.append("게이트 조건 일부 미달(상세는 게이트 표 참고).")
            gate_reasons = eval_result.get("gate_reasons", [])
            if gate_reasons:
                reasons.extend(gate_reasons)
        if ctx.get("structure_bias") == "BULL":
            reasons.append("구조 바이어스: 상승(HH/HL 구조).")
        if ctx.get("tag_confluence_ob_fvg"):
            reasons.append("OB/FVG 컨플루언스 구간으로 신뢰도 가점.")
        rs_tag = (ctx.get("rs") or {}).get("tag")
        if rs_tag == "RS_STRONG":
            reasons.append("상대강도 우수(지수 대비 강세).")
        regime_tag = (ctx.get("regime") or {}).get("tag")
        if regime_tag == "TAILWIND":
            reasons.append("시장 레짐 우호(상승/완만 변동성 구간).")
        entry_type_map = {
            "limit_pullback": "되돌림 지정가",
            "limit_in_zone": "구간 내부 지정가",
            "reclaim": "리클레임 확인 진입",
            "next_open": "다음 시가 진입",
        }
        entry_type_label = entry_type_map.get(entry_plan.entry_type, entry_plan.entry_type)
        reasons.append(
            "진입 계획: "
            f"{entry_type_label} · RR {entry_plan.rr:.2f} · "
            f"손절 {entry_plan.stop_loss:.0f} · 목표 {entry_plan.take_profit:.0f}."
        )
        return reasons

    def build_sell_reasons(self, exit_decisions: List[ExitDecision], position: Position, ctx: Dict[str, Any]) -> List[str]:
        reasons = []
        for d in exit_decisions:
            if d.action == "EXIT":
                reasons.append(d.reason)
        if ctx.get("structure_bias") == "BEAR":
            reasons.append("구조 바이어스 약세 전환.")
        score_val = ctx.get("soft_score", ctx.get("score"))
        if score_val is not None:
            reasons.append(f"현재 소프트 점수 {float(score_val):.2f}.")
        if position.exit_rules.get("tp_sl_conflict"):
            conflict_map = {"optimistic": "낙관적(TP 우선)", "conservative": "보수적(SL 우선)"}
            conflict_label = conflict_map.get(position.exit_rules["tp_sl_conflict"], position.exit_rules["tp_sl_conflict"])
            reasons.append(f"TP/SL 동시 터치 기준: {conflict_label}.")
        return reasons

    def describe_score_breakdown(self, breakdown: Dict[str, float]) -> List[str]:
        descriptions = {
            "dist_ob": "OB 근접도 가산점",
            "dist_fvg": "FVG 근접도 가산점",
            "confluence": "OB/FVG 컨플루언스",
            "regime": "시장 레짐 가중치",
            "rs": "상대강도 가중치",
            "atr_spike": "ATR 변동성 스파이크 패널티",
            "structure": "구조 바이어스 가중치",
            "above_ma200": "MA200 상단 가중치",
            "above_ma20": "MA20 상단 가중치",
            "ma20_above_ma200": "MA20>MA200 정배열 가중치",
            "rsi_bullish": "RSI 중립/상승 구간 가중치",
            "macd_bullish": "MACD 양수 가중치",
            "macd_cross": "MACD 시그널 상향 가중치",
            "volume_surge": "거래량 급증 가중치",
            "room_to_high": "상방 여유 공간 가중치",
            "momentum_20": "20일 모멘텀 가중치",
            "momentum_60": "60일 모멘텀 가중치",
            "ma20_slope": "MA20 기울기 가중치",
            "atr_ratio": "ATR 비율 가중치",
            "ob_quality": "OB 품질 가중치",
            "ob_age": "OB 노후 패널티",
            "fvg_age": "FVG 노후 패널티",
            "total": "총점",
        }
        lines = []
        for key, val in breakdown.items():
            desc = descriptions.get(key, key)
            lines.append(f"{key}: {val:.2f} · {desc}")
        return lines
