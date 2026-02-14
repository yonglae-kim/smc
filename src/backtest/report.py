from __future__ import annotations
import io, base64, re
from pathlib import Path
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
details summary{cursor:pointer;color:#333}
.detail-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px}
.detail-card{border:1px solid #eee;border-radius:10px;padding:10px;background:#fff}
.tag{display:inline-block;padding:2px 6px;border-radius:999px;background:#f3f3f3;font-size:11px;margin-right:4px}
.desktop-only{display:block}
.mobile-only{display:none}
.table-wrap{overflow-x:auto}
.trade-mobile-list{display:grid;gap:10px}
.trade-mobile-card{border:1px solid #eee;border-radius:10px;padding:10px;background:#fff}
.trade-mobile-head{display:grid;grid-template-columns:1fr auto;gap:4px 8px;align-items:center}
.trade-mobile-symbol{font-size:16px;font-weight:700}
.trade-mobile-pnl{font-size:16px;font-weight:700}
.progressive-hidden{display:none}
button.more-btn{margin-top:10px;padding:7px 11px;border:1px solid #d4d4d8;border-radius:9px;background:#fff;cursor:pointer}
@media(max-width:720px){
  .desktop-only{display:none}
  .mobile-only{display:block}
}
</style>
<script>
function initProgressiveList(group){
  const nodes=[...document.querySelectorAll(`[data-progressive-group="${group}"]`)];
  if(!nodes.length) return;
  const step=20;
  let visible=Math.min(step,nodes.length);
  nodes.forEach((node,idx)=>node.classList.toggle("progressive-hidden", idx>=visible));
  const btn=document.querySelector(`[data-more-button="${group}"]`);
  if(!btn) return;
  if(nodes.length<=step){btn.style.display="none";return;}
  btn.addEventListener("click",()=>{
    visible=Math.min(visible+step,nodes.length);
    nodes.forEach((node,idx)=>node.classList.toggle("progressive-hidden", idx>=visible));
    if(visible>=nodes.length) btn.style.display="none";
  });
}
document.addEventListener("DOMContentLoaded",()=>{
  initProgressiveList("trade-row");
  initProgressiveList("trade-mobile");
  initProgressiveList("trade-detail-card");
});
</script>
</head>
<body>
<h1>{{ title }}</h1>
<div class="small">기간 {{ start }} ~ {{ end }} · 거래 {{ metrics.get('trades', 0) }} · MDD {{ "%.2f"|format(metrics.get('mdd')*100) if metrics.get('mdd') is not none else "" }}% · 샤프 {{ "%.2f"|format(metrics.get('sharpe')) if metrics.get('sharpe') is not none else "" }} · 승률 {{ "%.1f"|format(metrics.get('winrate')*100) if metrics.get('winrate') is not none else "" }}%</div>

<div class="card">
  <h2>요약 지표</h2>
  <table>
    <thead>
      <tr>
        <th>승률</th>
        <th>평균 PnL</th>
        <th>평균 보유기간</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>{% if metrics.get('winrate') is not none %}{{ "%.1f"|format(metrics.get('winrate')*100) }}%{% else %}-{% endif %}</td>
        <td>{% if metrics.get('avg_pnl') is not none %}{{ "%.0f"|format(metrics.get('avg_pnl')) }}{% else %}-{% endif %}</td>
        <td>{% if metrics.get('avg_hold_days') is not none %}{{ "%.1f"|format(metrics.get('avg_hold_days')) }}일{% else %}-{% endif %}</td>
      </tr>
    </tbody>
  </table>
</div>

<div class="card">
  <h2>에퀴티 커브</h2>
  <img style="width:100%;border-radius:10px;border:1px solid #eee" src="{{ equity_img_src }}"/>
</div>

<div class="card">
  <h2>거래 내역</h2>
  {% if early_exit_summary %}
  <h3>조기 EXIT별 평균 RR/PNL</h3>
  <div class="table-wrap desktop-only">
    <table>
    <thead>
      <tr>
        <th>조기 EXIT 사유</th>
        <th>건수</th>
        <th>평균 RR</th>
        <th>평균 PnL</th>
      </tr>
    </thead>
    <tbody>
    {% for item in early_exit_summary %}
      <tr>
        <td>{{ item.reason }}</td>
        <td>{{ item.count }}</td>
        <td>{% if item.avg_rr is not none %}{{ "%.2f"|format(item.avg_rr) }}{% else %}-{% endif %}</td>
        <td>{% if item.avg_pnl is not none %}{{ "%.0f"|format(item.avg_pnl) }}{% else %}-{% endif %}</td>
      </tr>
    {% endfor %}
    </tbody>
    </table>
  </div>
  {% else %}
  <div class="small">조기 EXIT 기록이 없습니다.</div>
  {% endif %}
  <div class="table-wrap desktop-only">
  <table>
    <thead>
      <tr>
        <th>심볼 / 종목명</th>
        <th>진입</th>
        <th>청산</th>
        <th>진입가</th>
        <th>청산가</th>
        <th>PnL</th>
        <th>보유일</th>
        <th>SL 거리(ATR)</th>
        <th>RR 실현</th>
        <th>MAE</th>
        <th>MFE</th>
        <th>Symbol Regime</th>
        <th>청산 사유</th>
        <th>진입 점수</th>
        <th>점수 구성요소</th>
        <th>진입 사유</th>
      </tr>
    </thead>
    <tbody>
    {% for t in trades %}
      <tr data-progressive-group="trade-row">
        <td>{{ t.symbol }} {{ t.get("name", "") }}</td><td>{{ t.entry_date }}</td><td>{{ t.exit_date }}</td>
        <td>{{ "%.2f"|format(t.entry_px) }}</td><td>{{ "%.2f"|format(t.exit_px) }}</td>
        <td>{{ "%.0f"|format(t.pnl) }}</td>
        <td>{{ t.get("hold_days", 0) }}</td>
        <td>{% if t.stop_distance_atr is not none %}{{ "%.2f"|format(t.stop_distance_atr) }}{% else %}-{% endif %}</td>
        <td>{% if t.rr_realized is not none %}{{ "%.2f"|format(t.rr_realized) }}{% else %}-{% endif %}</td>
        <td>{{ "%.2f"|format(t.get("mae", 0.0)) }}</td>
        <td>{{ "%.2f"|format(t.get("mfe", 0.0)) }}</td>
        <td>{{ t.get("entry_structure_bias", "-") }}</td>
        <td>
          {% if t.exit_reason_lines %}
            <details>
              <summary>보기</summary>
              <ul>{% for r in t.exit_reason_lines %}<li>{{ r }}</li>{% endfor %}</ul>
            </details>
          {% else %}
            -
          {% endif %}
        </td>
        <td>{{ "%.1f"|format(t.get('entry_score', 0)) }}</td>
        <td>
          {% if t.entry_breakdown_items %}
            <details>
              <summary>보기</summary>
              <ul>{% for r in t.entry_breakdown_items %}<li>{{ r }}</li>{% endfor %}</ul>
            </details>
          {% else %}
            -
          {% endif %}
        </td>
        <td>
          {% if t.entry_reason_lines %}
            <details>
              <summary>보기</summary>
              <ul>{% for r in t.entry_reason_lines %}<li>{{ r }}</li>{% endfor %}</ul>
            </details>
          {% else %}
            -
          {% endif %}
        </td>
      </tr>
    {% endfor %}
    </tbody>
  </table>
  </div>
  <div class="mobile-only trade-mobile-list">
    {% for t in trades %}
    <div class="trade-mobile-card" data-progressive-group="trade-mobile">
      <div class="trade-mobile-head">
        <div class="trade-mobile-symbol">{{ t.symbol }}</div>
        <div class="trade-mobile-pnl">{{ "%.0f"|format(t.pnl) }}</div>
        <div class="small">{{ t.get("name", "") }}</div>
        <div class="small">RR {{ "%.2f"|format(t.rr_realized) if t.rr_realized is not none else "-" }}</div>
      </div>
      <div class="small" style="margin-top:6px">{{ t.entry_date }} → {{ t.exit_date }} · {{ t.get("entry_type", "-") }}</div>
      <details>
        <summary>상세 보기</summary>
        <div class="small" style="margin-top:6px">
          진입 {{ "%.2f"|format(t.entry_px) }} · 청산 {{ "%.2f"|format(t.exit_px) }} · 보유 {{ t.get("hold_days", 0) }}일<br/>
          SL 거리(ATR) {{ "%.2f"|format(t.stop_distance_atr) if t.stop_distance_atr is not none else "-" }} · MAE {{ "%.2f"|format(t.get("mae", 0.0)) }} · MFE {{ "%.2f"|format(t.get("mfe", 0.0)) }}<br/>
          Symbol Regime {{ t.get("entry_structure_bias", "-") }} · 진입 점수 {{ "%.1f"|format(t.get('entry_score', 0)) }}
        </div>
        <div class="small" style="margin-top:8px">
          <strong>청산 사유</strong>
          {% if t.exit_reason_lines %}
            <ul>{% for r in t.exit_reason_lines %}<li>{{ r }}</li>{% endfor %}</ul>
          {% else %}
            <div>-</div>
          {% endif %}
        </div>
        <div class="small" style="margin-top:8px">
          <strong>진입 사유</strong>
          {% if t.entry_reason_lines %}
            <ul>{% for r in t.entry_reason_lines %}<li>{{ r }}</li>{% endfor %}</ul>
          {% else %}
            <div>-</div>
          {% endif %}
        </div>
      </details>
    </div>
    {% endfor %}
  </div>
  <button class="more-btn" type="button" data-more-button="trade-row">거래내역 더 보기</button>
  <button class="more-btn mobile-only" type="button" data-more-button="trade-mobile">모바일 거래내역 더 보기</button>
</div>

<div class="card">
  <h2>상세 보기 카드</h2>
  <div class="detail-grid">
    {% for t in trades %}
      <div class="detail-card" data-progressive-group="trade-detail-card">
        <div><strong>{{ t.symbol }}</strong> {{ t.get("name", "") }}</div>
        <div class="small">{{ t.entry_date }} → {{ t.exit_date }}</div>
        <div>진입 {{ "%.2f"|format(t.entry_px) }} · 청산 {{ "%.2f"|format(t.exit_px) }} · PnL {{ "%.0f"|format(t.pnl) }}</div>
        <div>보유 {{ t.get("hold_days", 0) }}일 · RR {{ "%.2f"|format(t.rr_realized) if t.rr_realized is not none else "-" }}</div>
        <div>MAE {{ "%.2f"|format(t.get("mae", 0.0)) }} · MFE {{ "%.2f"|format(t.get("mfe", 0.0)) }}</div>
        <div>
          <span class="tag">Symbol Regime {{ t.get("entry_structure_bias", "-") }}</span>
          <span class="tag">SL/ATR {{ "%.2f"|format(t.stop_distance_atr) if t.stop_distance_atr is not none else "-" }}</span>
        </div>
      </div>
    {% endfor %}
  </div>
  <button class="more-btn" type="button" data-more-button="trade-detail-card">상세 카드 더 보기</button>
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

def _early_exit_summary(trades: list[dict]) -> list[dict]:
    summary: dict[str, dict[str, float]] = {}
    for trade in trades:
        reason_text = trade.get("exit_reason", "") or ""
        if "조기 EXIT" not in reason_text:
            continue
        reason_lines = _split_reasons(reason_text)
        reason_label = next((line for line in reason_lines if "조기 EXIT" in line), "조기 EXIT")
        bucket = summary.setdefault(
            reason_label,
            {"count": 0.0, "total_rr": 0.0, "rr_count": 0.0, "total_pnl": 0.0},
        )
        bucket["count"] += 1
        pnl = trade.get("pnl")
        if isinstance(pnl, (int, float)):
            bucket["total_pnl"] += float(pnl)
        rr_val = trade.get("rr_realized")
        if isinstance(rr_val, (int, float)):
            bucket["total_rr"] += float(rr_val)
            bucket["rr_count"] += 1
    rows = []
    for reason, bucket in summary.items():
        count = int(bucket["count"])
        avg_rr = bucket["total_rr"] / bucket["rr_count"] if bucket["rr_count"] > 0 else None
        avg_pnl = bucket["total_pnl"] / count if count > 0 else None
        rows.append(
            {
                "reason": reason,
                "count": count,
                "avg_rr": avg_rr,
                "avg_pnl": avg_pnl,
            }
        )
    rows.sort(key=lambda item: item["count"], reverse=True)
    return rows

def render_backtest_report(path: str, payload: Dict[str,Any]) -> None:
    payload = dict(payload)
    image_mode = str(payload.get("chart_image_mode", "inline_base64"))
    trades = []
    for t in payload.get("trades", []):
        trade = dict(t)
        trade["exit_reason_lines"] = _split_reasons(trade.get("exit_reason", ""))
        trade["entry_reason_lines"] = _split_reasons(trade.get("entry_reason", ""))
        breakdown = trade.get("entry_breakdown") or {}
        trade["entry_breakdown_items"] = [
            f"{k}: {v:.2f}" if isinstance(v, (int, float)) else f"{k}: {v}"
            for k, v in sorted(breakdown.items())
        ]
        trades.append(trade)
    payload["trades"] = trades
    payload["early_exit_summary"] = _early_exit_summary(trades)
    equity_png = _equity_png(payload.get("equity_curve", []))
    if image_mode == "file_link":
        out_path = Path(path)
        chart_path = out_path.with_name(f"{out_path.stem}_equity.png")
        chart_path.write_bytes(base64.b64decode(equity_png))
        payload["equity_img_src"] = chart_path.name
    else:
        payload["equity_img_src"] = f"data:image/png;base64,{equity_png}"
    html = HTML.render(**payload)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
