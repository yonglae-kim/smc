from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict

CURRENT_CONFIG_VERSION = 2


def _rename_field(section: Dict[str, Any], old: str, new: str) -> None:
    if old in section and new not in section:
        section[new] = section.pop(old)


def migrate_config(raw: Dict[str, Any]) -> Dict[str, Any]:
    data = deepcopy(raw or {})
    version = int(data.get("config_version", 1))

    if version < 2:
        network = data.setdefault("network", {})
        trade = data.setdefault("trade", {})
        backtest = data.setdefault("backtest", {})

        _rename_field(network, "timeout", "timeout_sec")
        _rename_field(network, "retry_count", "max_retries")
        _rename_field(trade, "tp_conflict_mode", "tp_sl_conflict")
        _rename_field(trade, "min_risk_pct", "min_risk_ratio")
        _rename_field(backtest, "position_limit", "max_positions")

    data["config_version"] = CURRENT_CONFIG_VERSION
    return data
