from __future__ import annotations
import io, base64
from typing import Dict, Any
from jinja2 import Template
import matplotlib.pyplot as plt
import numpy as np

HTML = Template(r"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8"/>
<title>{{ title }}</title>
<style>
body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:20px;color:#111}
.card{border:1px solid #e5e5e5;border-radius:12px;padding:14px;margin:14px 0}
table{border-collapse:collapse;width:100%}
th,td{border-bottom:1px solid #eee;padding:8px 6px;text-align:left;font-size:12px}
th{background:#fafafa}
.small{color:#666;font-size:12px}
</style>
</head>
<body>
<h1>{{ title }}</h1>
<div class="small">Period {{ start }} ~ {{ end }} · Trades {{ metrics.get('trades', 0) }} · MDD {{ "%.2f"|format(metrics.get('mdd')*100) if metrics.get('mdd') is not none else "" }}% · Sharpe {{ "%.2f"|format(metrics.get('sharpe')) if metrics.get('sharpe') is not none else "" }} · WinRate {{ "%.1f"|format(metrics.get('winrate')*100) if metrics.get('winrate') is not none else "" }}%</div>

<div class="card">
  <h2>Equity Curve</h2>
  <img style="width:100%;border-radius:10px;border:1px solid #eee" src="data:image/png;base64,{{ equity_png }}"/>
</div>

<div class="card">
  <h2>Trades</h2>
  <table>
    <thead><tr><th>Symbol / Name</th><th>Entry</th><th>Exit</th><th>EntryPx</th><th>ExitPx</th><th>PnL</th><th>ExitReason</th><th>EntryScore</th><th>EntryReason</th></tr></thead>
    <tbody>
    {% for t in trades %}
      <tr>
        <td>{{ t.symbol }} {{ t.get("name", "") }}</td><td>{{ t.entry_date }}</td><td>{{ t.exit_date }}</td>
        <td>{{ "%.2f"|format(t.entry_px) }}</td><td>{{ "%.2f"|format(t.exit_px) }}</td>
        <td>{{ "%.0f"|format(t.pnl) }}</td><td>{{ t.exit_reason }}</td><td>{{ "%.1f"|format(t.get('entry_score', 0)) }}</td><td>{{ t.entry_reason }}</td>
      </tr>
    {% endfor %}
    </tbody>
  </table>
</div>
</body>
</html>""")

def _equity_png(curve: list[dict]) -> str:
    x = np.arange(len(curve))
    y = np.array([c["equity"] for c in curve], dtype=float)
    fig = plt.figure(figsize=(10,4))
    ax = fig.add_subplot(1,1,1)
    ax.plot(x, y)
    ax.grid(True, linewidth=0.3)
    ax.set_title("Equity")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")

def render_backtest_report(path: str, payload: Dict[str,Any]) -> None:
    payload = dict(payload)
    payload["equity_png"] = _equity_png(payload.get("equity_curve", []))
    html = HTML.render(**payload)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
