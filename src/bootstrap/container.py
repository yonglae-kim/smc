from __future__ import annotations

import os

from ..providers.naver import NaverChartProvider, NaverMarketSumFetcher
from ..storage.fs import FSStorage
from ..universe.builder import UniverseBuilder
from ..utils.http import HttpClient
from ..utils.http_cache import HttpCache
from ..utils.time import today_kst


def build_http_client(cfg):
    cache_mode = cfg.network.cache_mode
    snapshot_id = cfg.network.cache_snapshot_id or today_kst().strftime("%Y-%m-%d")
    cache_snapshot = snapshot_id if cache_mode == "snapshot" else "latest"
    cache_dir = os.path.join(cfg.app.cache_dir, "http", cache_snapshot)

    http_cache = HttpCache(cache_dir, ttl_sec=cfg.network.cache_ttl_sec, mode=cache_mode)
    return HttpClient(
        timeout_sec=cfg.network.timeout_sec,
        max_retries=cfg.network.max_retries,
        backoff_base_sec=cfg.network.backoff_base_sec,
        jitter_sec=cfg.network.jitter_sec,
        rate_limit_per_sec=cfg.network.rate_limit_per_sec,
        cache=http_cache,
    )


def build_storage(cfg):
    return FSStorage(cfg.app.cache_dir)


def build_market_provider(cfg, http):
    _ = cfg
    return NaverChartProvider(http), NaverMarketSumFetcher(http)


def build_universe_builder(cfg, storage, provider, fetcher):
    return UniverseBuilder(storage, provider, fetcher, cfg.universe)


def build_daily_pipeline_service(config_path: str):
    from ..application.daily_pipeline import DailyPipelineService

    return DailyPipelineService(config_path)
