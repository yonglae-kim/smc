from __future__ import annotations

import yaml

from .analysis import AnalysisCfg
from .app import AppCfg, ReportCfg, ScoringCfg, SymbolRegimeCfg, UniverseCfg
from .backtest import BacktestCfg, TpCfg
from .migrations import migrate_config
from .network import NetworkCfg
from .schema import Config
from .trade import TradeCfg


def load_config(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    migrated = migrate_config(raw)
    return Config.model_validate(migrated)


__all__ = [
    "AppCfg",
    "AnalysisCfg",
    "BacktestCfg",
    "Config",
    "NetworkCfg",
    "ReportCfg",
    "ScoringCfg",
    "SymbolRegimeCfg",
    "TpCfg",
    "TradeCfg",
    "UniverseCfg",
    "load_config",
]
