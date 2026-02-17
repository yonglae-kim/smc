from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from ..strategy.soft_score import SoftScoreStrategy
from .models import AnalysisContext, EntryPlan, ExitDecision, Position, ScoredContext, TradeSignal
from .policies import EntryPolicy, ExitPolicy, GatingPolicy, RiskPolicy


class TradeRules:
    def __init__(
        self,
        cfg,
        strategy: Optional[SoftScoreStrategy] = None,
        entry_policy: Optional[EntryPolicy] = None,
        risk_policy: Optional[RiskPolicy] = None,
        exit_policy: Optional[ExitPolicy] = None,
        gating_policy: Optional[GatingPolicy] = None,
    ):
        self.cfg = cfg
        self.strategy = strategy or SoftScoreStrategy(cfg)
        self.entry_policy = entry_policy or EntryPolicy(cfg)
        self.risk_policy = risk_policy or RiskPolicy(cfg)
        self.exit_policy = exit_policy or ExitPolicy(cfg)
        self.gating_policy = gating_policy or GatingPolicy(cfg)

        trade = getattr(cfg, "trade", None)
        backtest = getattr(cfg, "backtest", None)
        self.execution_delay_days = int(getattr(trade, "execution_delay_days", 1))
        self.entry_price_mode = str(getattr(trade, "entry_price_mode", "next_open"))
        if backtest is not None and getattr(backtest, "fill_price", None):
            self.entry_price_mode = str(getattr(trade, "entry_price_mode", backtest.fill_price))
        self.force_top_k = int(getattr(trade, "force_top_k", 0))

    @staticmethod
    def _as_ctx(ctx: AnalysisContext | Dict[str, Any]) -> AnalysisContext:
        if isinstance(ctx, AnalysisContext):
            return ctx
        if "score" in ctx or "score_components" in ctx or "soft_score" in ctx:
            return ScoredContext.model_validate(ctx)
        return AnalysisContext.model_validate(ctx)

    def next_trading_day(self, calendar: Iterable[pd.Timestamp], date: str) -> str:
        if not calendar:
            return date
        dt = pd.to_datetime(date)
        cal = list(calendar)
        for c in cal:
            if c > dt:
                return str(c.date())
        return str(cal[-1].date())

    def build_entry_plan(self, ctx: AnalysisContext | Dict[str, Any], entry_price: float) -> EntryPlan:
        return self.entry_policy.build_entry_plan(self._as_ctx(ctx), entry_price)

    def build_signal(
        self,
        date: str,
        ctx: AnalysisContext | Dict[str, Any],
        calendar: Iterable[pd.Timestamp],
        entry_price: float,
    ) -> Tuple[TradeSignal, EntryPlan]:
        ctx = self._as_ctx(ctx)
        eval_result = self.strategy.evaluate(ctx)
        entry_plan = self.entry_policy.build_entry_plan(ctx, entry_price)
        min_score, gates, gate_reasons, breakdown = self.gating_policy.apply(ctx, eval_result, entry_plan)
        score = float(eval_result["score"])
        all_pass = all(gates.values()) if gates else False
        confidence = min(1.0, score / max(min_score * 2.0, 1e-6)) if min_score > 0 else 0.0
        reasons = self._build_buy_reasons(ctx, eval_result, entry_plan, all_pass)
        signal = TradeSignal(
            timestamp=date,
            valid_from=self.next_trading_day(calendar, date),
            symbol=ctx.symbol,
            direction="BUY",
            score=score,
            confidence=float(confidence),
            reasons=reasons,
            gates=gates,
            gate_reasons=gate_reasons,
            score_breakdown=breakdown,
            invalidation=entry_plan.invalidation,
        )
        return signal, entry_plan

    def signal_passes(self, signal: TradeSignal) -> bool:
        return all(signal.gates.values()) if signal.gates else False

    def build_entry_reasons(self, ctx: AnalysisContext | Dict[str, Any], signal: TradeSignal, entry_plan: EntryPlan) -> List[str]:
        return self._build_buy_reasons(self._as_ctx(ctx), {"gate_reasons": signal.gate_reasons}, entry_plan, self.signal_passes(signal))

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
        ctx: AnalysisContext | Dict[str, Any],
    ) -> Position:
        return self.risk_policy.build_position(signal, entry_plan, entry_date, entry_price, size, self._as_ctx(ctx))

    def update_trailing_stop(self, position: Position, ctx: AnalysisContext | Dict[str, Any]) -> None:
        self.exit_policy.update_trailing_stop(position, self._as_ctx(ctx))

    def apply_tp1_risk_reduction(self, position: Position, ctx: Optional[AnalysisContext]) -> None:
        self.exit_policy.apply_tp1_risk_reduction(position, ctx)

    def evaluate_exit(
        self,
        position: Position,
        bar: Dict[str, float],
        ctx: Optional[AnalysisContext],
        date: str,
        eval_ctx: Optional[Dict[str, Any]] = None,
    ) -> List[ExitDecision]:
        _ = date
        return self.exit_policy.evaluate_exit(position, bar, ctx, eval_ctx)

    def _build_buy_reasons(
        self,
        ctx: AnalysisContext | Dict[str, Any],
        eval_result: Dict[str, Any],
        entry_plan: EntryPlan,
        all_pass: bool,
    ) -> List[str]:
        ctx = self._as_ctx(ctx)
        reasons = []
        if not all_pass:
            reasons.append("게이트 조건 일부 미달(상세는 게이트 표 참고).")
            reasons.extend(eval_result.get("gate_reasons", []))
        if ctx.structure_bias == "BULL":
            reasons.append("구조 바이어스: 상승(HH/HL 구조).")
        if ctx.tag_confluence_ob_fvg:
            reasons.append("OB/FVG 컨플루언스 구간으로 신뢰도 가점.")
        if ctx.momentum_60 is not None and ctx.momentum_60 > 0:
            reasons.append("60일 모멘텀 양호.")
        reasons.append(
            "진입 계획: "
            f"{entry_plan.entry_type_label or entry_plan.entry_type} · RR {entry_plan.rr:.2f} · "
            f"손절 {entry_plan.stop_loss:.0f} · 목표 {entry_plan.take_profit:.0f}."
        )
        return reasons

    def build_sell_reasons(self, exit_decisions: List[ExitDecision], position: Position, ctx: Optional[AnalysisContext]) -> List[str]:
        return self.exit_policy.build_sell_reasons(exit_decisions, position, ctx)

    def describe_score_breakdown(self, breakdown: Dict[str, float]) -> List[str]:
        descriptions = {
            "dist_ob": "OB 근접도 가산점",
            "dist_fvg": "FVG 근접도 가산점",
            "confluence": "OB/FVG 컨플루언스",
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
            "vol_adj_return_20": "20일 변동성 대비 수익 가중치",
            "ob_quality": "OB 품질 가중치",
            "ob_age": "OB 노후 패널티",
            "fvg_age": "FVG 노후 패널티",
            "total": "총점",
        }
        return [f"{key}: {val:.2f} · {descriptions.get(key, key)}" for key, val in breakdown.items()]
