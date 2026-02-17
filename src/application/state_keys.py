ANALYSIS_PROGRESS_KEY_TEMPLATE = "state/analysis_progress_{ymd}.json"
POSITIONS_LIVE_KEY = "state/positions_live.json"


def analysis_progress_key(ymd: str) -> str:
    return ANALYSIS_PROGRESS_KEY_TEMPLATE.format(ymd=ymd)
