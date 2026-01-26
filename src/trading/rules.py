from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

import math

import pandas as pd

from .models import EntryPlan, ExitDecision, Position, TradeSignal
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
        self.enable_trend_filter = bool(getattr(trade, "enable_trend_filter", True))
        self.trend_ma_stack = bool(getattr(trade, "trend_ma_stack", True))
        self.trend_slope_atr_min = float(getattr(trade, "trend_slope_atr_min", 0.1))
        self.enable_volatility_filter = bool(getattr(trade, "enable_volatility_filter", True))
        self.max_atr_pct = float(getattr(trade, "max_atr_pct", 0.05))
        self.max_atr_ratio = float(getattr(trade, "max_atr_ratio", 1.8))
        self.enable_volume_confirm = bool(getattr(trade, "enable_volume_confirm", True))
        self.min_volume_ratio = float(getattr(trade, "min_volume_ratio", 1.4))
        self.enable_rs_rank = bool(getattr(trade, "enable_rs_rank", True))
        self.rs_rank_min_pct = float(getattr(trade, "rs_rank_min_pct", 0.7))
        self.enable_bb_squeeze_breakout = bool(getattr(trade, "enable_bb_squeeze_breakout", True))
        self.bb_squeeze_max_width = float(getattr(trade, "bb_squeeze_max_width", 0.08))
        self.min_confirmations = int(getattr(trade, "min_confirmations", 0))

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
        gates["score_min"] = score >= min_score
        gates["min_rr"] = entry_plan.rr >= self.min_rr
        gates["min_expected_return"] = entry_plan.expected_return >= self.min_expected_return
        trend_ok = self._trend_ok(ctx)
        vol_ok = self._volatility_ok(ctx)
        volume_ok = self._volume_confirm(ctx)
        modules = self._module_checks(ctx, volume_ok)
        if self.enable_trend_filter:
            gates["trend_filter"] = trend_ok
        if self.enable_volatility_filter:
            gates["volatility_filter"] = vol_ok
        if self.enable_volume_confirm:
            gates["volume_confirm"] = volume_ok
        for key, result in modules.items():
            if result["enabled"]:
                gates[key] = bool(result["pass"])
        confirmations = self._count_confirmations(modules, volume_ok)
        if self.min_confirmations > 0:
            gates["min_confirmations"] = confirmations >= self.min_confirmations
        all_pass = all(gates.values()) if gates else False
        confidence = min(1.0, score / max(min_score * 2.0, 1e-6)) if min_score > 0 else 0.0
        reasons = self._build_buy_reasons(ctx, eval_result, entry_plan, all_pass, modules, confirmations)
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
            score_breakdown=eval_result.get("breakdown", {}),
            modules=modules,
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
                        reason="TP/SL 동시 터치: TP 우선(낙관적)",
                        price=position.take_profit,
                    )
                )
                return decisions
            decisions.append(
                ExitDecision(
                    action="EXIT",
                    reason="TP/SL 동시 터치: SL 우선(보수적)",
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

        if position.hold_days >= self.max_hold_days:
            decisions.append(ExitDecision(action="EXIT", reason=f"보유기간 만료({self.max_hold_days}일)", price=close_px))
            return decisions

        if ctx is not None:
            if self._ma_slope_sell_gate(ctx):
                decisions.append(
                    ExitDecision(
                        action="EXIT",
                        reason=self._ma_slope_sell_reason(ctx),
                        price=close_px,
                    )
                )
                return decisions
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
        modules: Dict[str, Dict[str, Any]],
        confirmations: int,
    ) -> List[str]:
        reasons = []
        if not all_pass:
            reasons.append("게이트 조건 일부 미달(상세는 게이트 표 참고).")
        if self.enable_trend_filter:
            reasons.append(
                "추세 필터: MA 정배열/기울기 확인으로 추세 구간만 선별."
                if self._trend_ok(ctx)
                else "추세 필터: 정배열/기울기 조건 미달."
            )
        if self.enable_volatility_filter:
            reasons.append(
                "변동성 필터: 과도한 ATR 구간 제외로 노이즈 축소."
                if self._volatility_ok(ctx)
                else "변동성 필터: ATR 비율 조건 미달."
            )
        if self.enable_volume_confirm:
            reasons.append(
                "거래량 확인: 평균 대비 거래량 급증 구간만 통과."
                if self._volume_confirm(ctx)
                else "거래량 확인: 평균 대비 거래량 부족."
            )
        ma_gate = ctx.get("ma_slope_gate") or {}
        for line in ma_gate.get("buy_reasons") or []:
            reasons.append(line)
        if self.min_confirmations > 0:
            reasons.append(f"추가 모듈 확인 {confirmations}건 충족(최소 {self.min_confirmations}).")
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
        for key, result in modules.items():
            if not result.get("enabled"):
                continue
            label = result.get("label", key)
            status = "충족" if result.get("pass") else "미충족"
            detail = result.get("detail", "")
            reasons.append(f"{label}: {status}. {detail}".strip())
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

    def _trend_ok(self, ctx: Dict[str, Any]) -> bool:
        ma20 = ctx.get("ma20")
        ma60 = ctx.get("ma60")
        ma120 = ctx.get("ma120")
        ma120_slope_atr = ctx.get("ma120_slope_atr")
        if ma20 is None or ma60 is None or ma120 is None or ma120_slope_atr is None:
            return False
        stack_ok = (ma20 >= ma60 >= ma120) if self.trend_ma_stack else True
        slope_ok = ma120_slope_atr >= self.trend_slope_atr_min
        return bool(stack_ok and slope_ok)

    def _volatility_ok(self, ctx: Dict[str, Any]) -> bool:
        atr_pct = ctx.get("atr_pct")
        atr_ratio = ctx.get("atr_ratio")
        if atr_pct is None or atr_ratio is None:
            return False
        return bool(atr_pct <= self.max_atr_pct and atr_ratio <= self.max_atr_ratio)

    def _volume_confirm(self, ctx: Dict[str, Any]) -> bool:
        volume_ratio = ctx.get("volume_ratio")
        if volume_ratio is None:
            return False
        return bool(volume_ratio >= self.min_volume_ratio)

    def _ma_slope_sell_gate(self, ctx: Dict[str, Any]) -> bool:
        gate = ctx.get("ma_slope_gate") or {}
        if not gate.get("enabled", False):
            return False
        return bool(gate.get("sell_pass"))

    def _ma_slope_sell_reason(self, ctx: Dict[str, Any]) -> str:
        gate = ctx.get("ma_slope_gate") or {}
        reasons = gate.get("sell_reasons") or []
        if reasons:
            return "MA 기울기 매도 게이트 충족: " + " / ".join(reasons)
        return "MA 기울기 매도 게이트 충족"

    def _module_checks(self, ctx: Dict[str, Any], volume_ok: bool) -> Dict[str, Dict[str, Any]]:
        results: Dict[str, Dict[str, Any]] = {}
        rs_rank = ctx.get("rs_rank_pct")
        rs_pass = rs_rank is not None and rs_rank >= self.rs_rank_min_pct
        results["module_rs_rank"] = {
            "enabled": self.enable_rs_rank,
            "pass": bool(rs_pass),
            "label": "상대강도 랭킹",
            "detail": f"랭킹 {rs_rank:.0%} (최소 {self.rs_rank_min_pct:.0%})" if rs_rank is not None else "랭킹 데이터 없음",
        }
        bb_width_min = ctx.get("bb_width_min")
        bb_upper = ctx.get("bb_upper")
        close = ctx.get("close")
        squeeze_ok = bb_width_min is not None and bb_width_min <= self.bb_squeeze_max_width
        breakout_ok = bb_upper is not None and close is not None and close > bb_upper
        bb_pass = bool(squeeze_ok and breakout_ok and volume_ok)
        results["module_bb_squeeze"] = {
            "enabled": self.enable_bb_squeeze_breakout,
            "pass": bb_pass,
            "label": "BB 수축 후 돌파",
            "detail": (
                f"수축폭 {bb_width_min:.2%} / 임계 {self.bb_squeeze_max_width:.2%}, 상단 돌파={breakout_ok}, 거래량={volume_ok}"
                if bb_width_min is not None
                else "밴드 폭 데이터 없음"
            ),
        }
        return results

    def _count_confirmations(self, modules: Dict[str, Dict[str, Any]], volume_ok: bool) -> int:
        count = 0
        if self.enable_volume_confirm and volume_ok:
            count += 1
        for result in modules.values():
            if result.get("enabled") and result.get("pass"):
                count += 1
        return count

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

    def describe_modules(self, modules: Dict[str, Dict[str, Any]]) -> List[str]:
        lines = []
        for key, info in modules.items():
            label = info.get("label", key)
            if not info.get("enabled"):
                lines.append(f"{label}: OFF")
                continue
            status = "통과" if info.get("pass") else "실패"
            detail = info.get("detail", "")
            lines.append(f"{label}: {status} · {detail}".strip())
        return lines
