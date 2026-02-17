from __future__ import annotations

import argparse

from .bootstrap.container import build_daily_pipeline_service


def run(config_path: str) -> None:
    build_daily_pipeline_service(config_path).run()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
