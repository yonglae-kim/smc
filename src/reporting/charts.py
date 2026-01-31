from __future__ import annotations
import base64
import io
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams
from matplotlib.patches import Rectangle

_KOREAN_FONT_CONFIGURED = False

def _configure_korean_font() -> None:
    global _KOREAN_FONT_CONFIGURED
    if _KOREAN_FONT_CONFIGURED:
        return
    _KOREAN_FONT_CONFIGURED = True

    candidates = (
        "Noto Sans CJK KR",
        "NotoSansCJKkr",
        "Noto Sans KR",
        "NanumGothic",
        "Malgun Gothic",
        "AppleGothic",
    )
    font_files = font_manager.findSystemFonts(fontext="ttf") + font_manager.findSystemFonts(fontext="otf")
    selected_name = None
    for path in font_files:
        lower = path.lower()
        if any(c.replace(" ", "").lower() in lower.replace("-", "").replace("_", "") for c in candidates):
            try:
                font_manager.fontManager.addfont(path)
                selected_name = font_manager.FontProperties(fname=path).get_name()
                break
            except Exception:
                continue

    if selected_name:
        rcParams["font.family"] = selected_name
    else:
        rcParams.setdefault("font.family", "sans-serif")
        rcParams["font.sans-serif"] = list(candidates) + ["DejaVu Sans", "Arial", "sans-serif"]
        warnings.filterwarnings("ignore", message="Glyph .* missing from current font.")

def _candles(ax, df: pd.DataFrame):
    # Basic candlestick without mplfinance.
    for i, r in df.iterrows():
        o,h,l,c = r["open"], r["high"], r["low"], r["close"]
        ax.vlines(i, l, h, linewidth=0.8)
        body_low = min(o,c)
        body_h = max(1e-9, abs(c-o))
        rect = Rectangle((i-0.3, body_low), 0.6, body_h, fill=True, alpha=0.7)
        ax.add_patch(rect)
    ax.set_xlim(-1, len(df))

def plot_symbol_chart(df: pd.DataFrame, ctx: dict, lookback: int=180) -> str:
    """Return base64 PNG for single symbol. Price + RSI subchart, zones overlay."""
    _configure_korean_font()
    full = df.reset_index(drop=True).copy()
    d = full.tail(lookback).reset_index(drop=True).copy()
    if len(d) < 30:
        d = full.copy().reset_index(drop=True)
    start_idx = max(0, len(full) - len(d))

    fig = plt.figure(figsize=(10,6))
    gs = fig.add_gridspec(3,1, height_ratios=[2.2, 0.1, 1.0], hspace=0.05)
    ax = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[2,0], sharex=ax)

    _candles(ax, d)

    # pivot/fractal points
    pivots = ctx.get("pivots") or []
    for p in pivots:
        idx = int(p["idx"]) - start_idx
        if 0 <= idx < len(d):
            color = "tab:green" if p["kind"] == "L" else "tab:red"
            marker = "^" if p["kind"] == "L" else "v"
            ax.scatter(idx, p["price"], color=color, s=18, marker=marker, zorder=3)

    struct_pts = ctx.get("structure_points") or []
    for p in struct_pts[-8:]:
        idx = int(p["idx"]) - start_idx
        if 0 <= idx < len(d) and p.get("cls"):
            ax.text(idx, p["price"], p["cls"], fontsize=7, color="#333", ha="center", va="bottom")

    # overlays
    if "ma20" in d.columns:
        ax.plot(np.arange(len(d)), d["ma20"].values, linewidth=1.0)
    if "ma200" in d.columns:
        ax.plot(np.arange(len(d)), d["ma200"].values, linewidth=1.0)

    # OB/FVG zones as spans
    if ctx.get("ob"):
        lo = ctx["ob"]["lower"]; hi = ctx["ob"]["upper"]
        ax.axhspan(lo, hi, alpha=0.18)
        ax.text(0.01, 0.95, f"OB {ctx['ob']['kind']} [{lo:.0f},{hi:.0f}] inv={ctx['ob']['invalidation']:.0f}",
                transform=ax.transAxes, fontsize=9, va="top")
    if ctx.get("fvg"):
        lo = ctx["fvg"]["lower"]; hi = ctx["fvg"]["upper"]
        ax.axhspan(lo, hi, alpha=0.12)
        ax.text(0.01, 0.88, f"FVG {ctx['fvg']['kind']} [{lo:.0f},{hi:.0f}] {ctx['fvg']['status']} fill={ctx['fvg']['fill_ratio']*100:.0f}%",
                transform=ax.transAxes, fontsize=9, va="top")

    ax.set_title(f"{ctx.get('symbol')} {ctx.get('name')} ({ctx.get('market')})  close={ctx.get('close'):.0f}")
    ax.grid(True, linewidth=0.3)
    ax.tick_params(labelbottom=False)

    # RSI
    if "rsi14" in d.columns:
        ax2.plot(np.arange(len(d)), d["rsi14"].values, linewidth=1.0)
        ax2.axhline(50, linewidth=0.8, linestyle="--")
        ax2.axhline(70, linewidth=0.6, linestyle=":")
        ax2.axhline(30, linewidth=0.6, linestyle=":")
        ax2.set_ylim(0,100)
    ax2.grid(True, linewidth=0.3)

    # entry/SL/TP lines
    entry_plan = ctx.get("entry_plan") or {}
    position = ctx.get("position") or {}
    entry_px = entry_plan.get("entry_price") or position.get("entry_price")
    stop_px = entry_plan.get("stop_loss") or position.get("stop_loss")
    tp_px = entry_plan.get("take_profit") or position.get("take_profit")
    if entry_px:
        ax.axhline(entry_px, color="blue", linestyle="--", linewidth=1.0)
        ax.text(0.01, 0.78, f"Entry {entry_px:.0f}", transform=ax.transAxes, fontsize=8, color="blue")
    if stop_px:
        ax.axhline(stop_px, color="red", linestyle="--", linewidth=1.0)
        ax.text(0.01, 0.72, f"SL {stop_px:.0f}", transform=ax.transAxes, fontsize=8, color="red")
    if tp_px:
        ax.axhline(tp_px, color="green", linestyle="--", linewidth=1.0)
        ax.text(0.01, 0.66, f"TP {tp_px:.0f}", transform=ax.transAxes, fontsize=8, color="green")

    # x ticks by date
    xt = np.linspace(0, len(d)-1, 6).astype(int)
    labels = [str(d["date"].iloc[i].date()) for i in xt]
    ax2.set_xticks(xt)
    ax2.set_xticklabels(labels, rotation=0, fontsize=8)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")
