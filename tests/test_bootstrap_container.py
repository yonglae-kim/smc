from __future__ import annotations

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.bootstrap.container import (
    build_http_client,
    build_market_provider,
    build_storage,
    build_universe_builder,
)
from src.config import (
    AppCfg,
    AnalysisCfg,
    BacktestCfg,
    Config,
    NetworkCfg,
    ReportCfg,
    ScoringCfg,
    SymbolRegimeCfg,
    UniverseCfg,
)
from src.providers.naver import NaverChartProvider, NaverMarketSumFetcher
from src.storage.fs import FSStorage
from src.universe.builder import UniverseBuilder
from src.utils.http import HttpClient


def make_cfg(cache_mode: str = "use", snapshot_id: str = "") -> Config:
    return Config(
        app=AppCfg(out_dir="./out", cache_dir="./cache", log_level="INFO"),
        network=NetworkCfg(cache_mode=cache_mode, cache_snapshot_id=snapshot_id),
        universe=UniverseCfg(),
        analysis=AnalysisCfg(),
        scoring=ScoringCfg(weights={}, top_detail=5),
        symbol_regime=SymbolRegimeCfg(),
        report=ReportCfg(title="Test Report"),
        backtest=BacktestCfg(),
    )


def test_build_http_client_uses_latest_cache_dir_in_non_snapshot_mode():
    cfg = make_cfg(cache_mode="use")

    client = build_http_client(cfg)

    assert isinstance(client, HttpClient)
    assert client.cache is not None
    assert client.cache.base_dir.endswith(os.path.join("http", "latest"))


def test_build_http_client_uses_snapshot_id_when_snapshot_mode():
    cfg = make_cfg(cache_mode="snapshot", snapshot_id="2025-01-31")

    client = build_http_client(cfg)

    assert client.cache is not None
    assert client.cache.base_dir.endswith(os.path.join("http", "2025-01-31"))


def test_build_http_client_generates_snapshot_id_from_today_when_missing(monkeypatch):
    class _FakeDate:
        def strftime(self, fmt: str) -> str:
            assert fmt == "%Y-%m-%d"
            return "2024-02-29"

    import src.bootstrap.container as container

    monkeypatch.setattr(container, "today_kst", lambda: _FakeDate())
    cfg = make_cfg(cache_mode="snapshot", snapshot_id="")

    client = build_http_client(cfg)

    assert client.cache is not None
    assert client.cache.base_dir.endswith(os.path.join("http", "2024-02-29"))


def test_factory_builds_storage_market_and_universe_builder_types():
    cfg = make_cfg()

    storage = build_storage(cfg)
    http = build_http_client(cfg)
    provider, fetcher = build_market_provider(cfg, http)
    ub = build_universe_builder(cfg, storage, provider, fetcher)

    assert isinstance(storage, FSStorage)
    assert isinstance(provider, NaverChartProvider)
    assert isinstance(fetcher, NaverMarketSumFetcher)
    assert isinstance(ub, UniverseBuilder)
    assert ub.storage is storage
    assert ub.provider is provider
    assert ub.fetcher is fetcher
    assert ub.cfg is cfg.universe
