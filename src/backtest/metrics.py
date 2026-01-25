from __future__ import annotations
from typing import Dict, Any
import numpy as np

def compute_metrics(result: Dict[str,Any]) -> Dict[str,Any]:
    # Always return a stable schema so templates never crash.
    metrics: Dict[str,Any] = {
        "trades": 0,
        "mdd": None,
        "sharpe": None,
        "winrate": None,
        "avg_win": None,
        "avg_loss": None,
        "expectancy": None,
        "profit_factor": None,
    }

    curve = result.get("equity_curve", []) or []
    trades = result.get("trades", []) or []
    metrics["trades"] = len(trades)

    if len(curve) < 2:
        return metrics

    eq = np.array([x.get("equity", np.nan) for x in curve], dtype=float)
    if np.any(~np.isfinite(eq)):
        return metrics

    peak = np.maximum.accumulate(eq)
    metrics["mdd"] = float(((eq / (peak + 1e-12)) - 1.0).min())

    rets = eq[1:] / (eq[:-1] + 1e-12) - 1.0
    if len(rets) > 10 and float(rets.std()) > 1e-12:
        metrics["sharpe"] = float((rets.mean() / (rets.std()+1e-12)) * (252**0.5))

    if trades:
        pnl = np.array([t.get("pnl", 0.0) for t in trades], dtype=float)
        if len(pnl) > 0:
            wins = pnl[pnl > 0]
            losses = pnl[pnl < 0]
            metrics["winrate"] = float((pnl > 0).mean())
            metrics["avg_win"] = float(wins.mean()) if len(wins) > 0 else 0.0
            metrics["avg_loss"] = float(losses.mean()) if len(losses) > 0 else 0.0
            metrics["expectancy"] = float(metrics["winrate"] * metrics["avg_win"] + (1 - metrics["winrate"]) * metrics["avg_loss"])
            loss_sum = float(losses.sum())
            metrics["profit_factor"] = float(wins.sum() / abs(loss_sum)) if loss_sum != 0 else None

    return metrics
