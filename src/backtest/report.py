from __future__ import annotations
import io, base64, re
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
th,td{border-bottom:1px solid #eee;padding:8px 6px;text-align:left;font-size:12px;vertical-align:top;line-height:1.4}
th{background:#fafafa}
.small{color:#666;font-size:12px}
tbody tr:nth-child(even){background:#fcfcff}
ul{margin:0;padding-left:16px}
</style>
</head>
<body>
<h1>{{ title }}</h1>
<div class="small">기간 {{ start }} ~ {{ end }} · 거래 {{ metrics.get('trades', 0) }} · MDD {{ "%.2f"|format(metrics.get('mdd')*100) if metrics.get('mdd') is not none else "" }}% · 샤프 {{ "%.2f"|format(metrics.get('sharpe')) if metrics.get('sharpe') is not none else "" }} · 승률 {{ "%.1f"|format(metrics.get('winrate')*100) if metrics.get('winrate') is not none else "" }}%</div>

<div class="card">
  <h2>요약 지표</h2>
  <table>
    <thead>
      <tr>
        <th>Trades</th>
        <th>Winrate</th>
        <th>Avg Win</th>
        <th>Avg Loss</th>
        <th>Expectancy</th>
        <th>Profit Factor</th>
        <th>MDD</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>{{ metrics.get('trades', 0) }}</td>
        <td>{{ "%.1f"|format(metrics.get('winrate')*100) if metrics.get('winrate') is not none else "" }}%</td>
        <td>{{ "%.0f"|format(metrics.get('avg_win')) if metrics.get('avg_win') is not none else "" }}</td>
        <td>{{ "%.0f"|format(metrics.get('avg_loss')) if metrics.get('avg_loss') is not none else "" }}</td>
        <td>{{ "%.0f"|format(metrics.get('expectancy')) if metrics.get('expectancy') is not none else "" }}</td>
        <td>{{ "%.2f"|format(metrics.get('profit_factor')) if metrics.get('profit_factor') is not none else "" }}</td>
        <td>{{ "%.2f"|format(metrics.get('mdd')*100) if metrics.get('mdd') is not none else "" }}%</td>
      </tr>
    </tbody>
  </table>
</div>

<div class="card">
  <h2>에퀴티 커브</h2>
  <img style="width:100%;border-radius:10px;border:1px solid #eee" src="data:image/png;base64,{{ equity_png }}"/>
</div>

<div class="card">
  <h2>거래 내역</h2>
  <table>
    <thead><tr><th>심볼 / 종목명</th><th>진입</th><th>청산</th><th>진입가</th><th>청산가</th><th>PnL</th><th>청산 사유</th><th>진입 점수</th><th>진입 사유</th></tr></thead>
    <tbody>
    {% for t in trades %}
      <tr>
        <td>{{ t.symbol }} {{ t.get("name", "") }}</td><td>{{ t.entry_date }}</td><td>{{ t.exit_date }}</td>
        <td>{{ "%.2f"|format(t.entry_px) }}</td><td>{{ "%.2f"|format(t.exit_px) }}</td>
        <td>{{ "%.0f"|format(t.pnl) }}</td>
        <td>
          {% if t.exit_reason_lines %}
            <ul>{% for r in t.exit_reason_lines %}<li>{{ r }}</li>{% endfor %}</ul>
          {% else %}
            -
          {% endif %}
        </td>
        <td>{{ "%.1f"|format(t.get('entry_score', 0)) }}</td>
        <td>
          {% if t.entry_reason_lines %}
            <ul>{% for r in t.entry_reason_lines %}<li>{{ r }}</li>{% endfor %}</ul>
          {% else %}
            -
          {% endif %}
        </td>
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

def _split_reasons(text: str) -> list[str]:
    if not text:
        return []
    parts = [p.strip() for p in re.split(r"[;\n]+", text) if p.strip()]
    return parts

def render_backtest_report(path: str, payload: Dict[str,Any]) -> None:
    payload = dict(payload)
    trades = []
    for t in payload.get("trades", []):
        trade = dict(t)
        trade["exit_reason_lines"] = _split_reasons(trade.get("exit_reason", ""))
        trade["entry_reason_lines"] = _split_reasons(trade.get("entry_reason", ""))
        trades.append(trade)
    payload["trades"] = trades
    payload["equity_png"] = _equity_png(payload.get("equity_curve", []))
    html = HTML.render(**payload)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
