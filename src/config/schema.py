from __future__ import annotations

from pydantic import BaseModel

from .analysis import AnalysisCfg
from .app import AppCfg, ReportCfg, ScoringCfg, SymbolRegimeCfg, UniverseCfg
from .backtest import BacktestCfg
from .network import NetworkCfg
from .trade import TradeCfg


class Config(BaseModel):
    config_version: int = 2
    app: AppCfg
    network: NetworkCfg
    universe: UniverseCfg
    analysis: AnalysisCfg
    scoring: ScoringCfg
    symbol_regime: SymbolRegimeCfg
    report: ReportCfg
    trade: TradeCfg = TradeCfg()
    backtest: BacktestCfg
