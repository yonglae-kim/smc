from __future__ import annotations

import argparse

from .application.daily_pipeline import DailyPipelineService


def run(config_path: str) -> None:
    DailyPipelineService(config_path).run()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
