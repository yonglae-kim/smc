from __future__ import annotations
from typing import List, Dict, Any
from jinja2 import Template

HTML_TMPL = Template(r"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{{ title }}</title>
<style>
body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:20px;color:#111}
h1{margin:0 0 8px 0}
.small{color:#666;font-size:12px}
.badge{display:inline-block;padding:2px 8px;border-radius:999px;font-size:12px;margin-right:6px;background:#eee}
.card{border:1px solid #e5e5e5;border-radius:12px;padding:14px;margin:14px 0}
.grid{display:grid;grid-template-columns:1fr;gap:10px}
@media(min-width:1000px){.grid{grid-template-columns:1.3fr 1fr}}
table{border-collapse:collapse;width:100%}
th,td{border-bottom:1px solid #eee;padding:8px 6px;text-align:left;font-size:12px;vertical-align:top;line-height:1.4}
th{position:sticky;top:0;background:#fafafa}
tr:hover{background:#fcfcfc}
input{padding:8px 10px;border:1px solid #ddd;border-radius:8px;width:340px}
pre{white-space:pre-wrap;margin:0;font-size:12px;color:#333}
.kpi{display:flex;gap:12px;flex-wrap:wrap;margin:10px 0}
.kpi .card{margin:0;padding:10px 12px}
tbody tr:nth-child(even){background:#fcfcff}
.section-title{margin-top:18px}
</style>
{% if include_js %}
<script>
function sortTable(n){
  const table=document.getElementById("uTable");
  let switching=true, dir="desc", switchcount=0;
  while(switching){
    switching=false;
    const rows=table.rows;
    for(let i=1;i<rows.length-1;i++){
      let shouldSwitch=false;
      const x=rows[i].getElementsByTagName("TD")[n];
      const y=rows[i+1].getElementsByTagName("TD")[n];
      const xv=parseFloat(x.getAttribute("data-sort")||x.innerText)||x.innerText;
      const yv=parseFloat(y.getAttribute("data-sort")||y.innerText)||y.innerText;
      if(dir=="asc" && xv>yv) shouldSwitch=true;
      if(dir=="desc" && xv<yv) shouldSwitch=true;
      if(shouldSwitch){
        rows[i].parentNode.insertBefore(rows[i+1], rows[i]);
        switching=true; switchcount++; break;
      }
    }
    if(!switching && switchcount==0){dir = (dir=="asc")?"desc":"asc"; switching=true;}
  }
}
function filterTable(){
  const q=document.getElementById("q").value.toLowerCase();
  const table=document.getElementById("uTable");
  const rows=table.getElementsByTagName("tr");
  for(let i=1;i<rows.length;i++){
    const txt=rows[i].innerText.toLowerCase();
    rows[i].style.display = txt.indexOf(q)>-1 ? "" : "none";
  }
}
</script>
{% endif %}
</head>
<body>
<h1>{{ title }}</h1>
<div class="small">생성 시각 {{ generated_at }} (KST) · 유니버스: 유동성 상위 {{ universe_n }}개 (중위값, {{ liquidity_window }}일)</div>
<div class="card">
  <div style="font-weight:700">실행 가이드</div>
  <div class="small">{{ execution_guide }}</div>
  <div class="small" style="margin-top:6px">가정: {{ tp_sl_conflict_note }}</div>
</div>

<h2 class="section-title">시장 레짐</h2>
<div>
  <span class="badge">KOSPI: {{ regime_kospi.tag }} · MA200={{ "상단" if regime_kospi.above_ma200 else "하단" }} · RSI50={{ "상단" if regime_kospi.rsi_ge_50 else "하단" }} · ATR spike={{ regime_kospi.atr_spike }}</span>
  <span class="badge">KOSDAQ: {{ regime_kosdaq.tag }} · MA200={{ "상단" if regime_kosdaq.above_ma200 else "하단" }} · RSI50={{ "상단" if regime_kosdaq.rsi_ge_50 else "하단" }} · ATR spike={{ regime_kosdaq.atr_spike }}</span>
</div>

<h2 class="section-title">Top500 요약</h2>
{% if include_js %}
<div style="margin:8px 0 10px 0">
  <input id="q" onkeyup="filterTable()" placeholder="심볼/이름/태그 검색..."/>
</div>
{% endif %}
<table id="uTable">
  <thead>
    <tr>
      <th onclick="sortTable(0)">순위</th>
      <th onclick="sortTable(1)">점수</th>
      <th>심볼</th>
      <th>종목명</th>
      <th>시장</th>
      <th>태그</th>
      <th onclick="sortTable(6)">종가</th>
      <th onclick="sortTable(7)">MA200</th>
      <th onclick="sortTable(8)">RSI</th>
      <th>레벨</th>
    </tr>
  </thead>
  <tbody>
  {% for r in table_rows %}
    <tr>
      <td data-sort="{{ r.rank }}">{{ r.rank }}</td>
      <td data-sort="{{ r.score }}">{{ "%.1f"|format(r.score) }}</td>
      <td>{{ r.symbol }}</td>
      <td>{{ r.name }}</td>
      <td>{{ r.market }}</td>
      <td>{{ r.tags|join(", ") }}</td>
      <td data-sort="{{ r.close }}">{{ "%.0f"|format(r.close) }}</td>
      <td data-sort="{{ r.ma200 or 0 }}">{{ "%.0f"|format(r.ma200) if r.ma200 else "" }}</td>
      <td data-sort="{{ r.rsi14 or 0 }}">{{ "%.1f"|format(r.rsi14) if r.rsi14 else "" }}</td>
      <td>{{ r.levels }}</td>
    </tr>
  {% endfor %}
  </tbody>
</table>

<h2 class="section-title">매수 후보 (다음 세션)</h2>
<div class="small">시그널은 종가 기준 산출, {{ buy_valid_from }}부터 유효.</div>
<table>
  <thead>
    <tr>
      <th>순위</th>
      <th>점수</th>
      <th>심볼</th>
      <th>종목명</th>
      <th>진입가</th>
      <th>손절</th>
      <th>목표</th>
      <th>RR</th>
      <th>게이트</th>
    </tr>
  </thead>
  <tbody>
  {% for b in buy_rows %}
    <tr>
      <td>{{ b.rank }}</td>
      <td>{{ "%.2f"|format(b.signal.score) }}</td>
      <td>{{ b.symbol }}</td>
      <td>{{ b.name }}</td>
      <td>{{ "%.0f"|format(b.entry_plan.entry_price) }}</td>
      <td>{{ "%.0f"|format(b.entry_plan.stop_loss) }}</td>
      <td>{{ "%.0f"|format(b.entry_plan.take_profit) }}</td>
      <td>{{ "%.2f"|format(b.entry_plan.rr) }}</td>
      <td>
        {% for g in b.gates %}
          <span class="badge">{{ g.key }}={{ "통과" if g.pass else "실패" }}</span>
        {% endfor %}
      </td>
    </tr>
  {% endfor %}
  </tbody>
</table>

<h2 class="section-title">매도 후보 (리스크 관리)</h2>
<div class="small">보유 포지션 기준으로만 산출.</div>
<table>
  <thead>
    <tr>
      <th>심볼</th>
      <th>종목명</th>
      <th>진입가</th>
      <th>현재가</th>
      <th>P/L</th>
      <th>청산 사유</th>
      <th>다음 액션</th>
    </tr>
  </thead>
  <tbody>
  {% for s in sell_rows %}
    <tr>
      <td>{{ s.symbol }}</td>
      <td>{{ s.name }}</td>
      <td>{{ "%.0f"|format(s.entry_price) }}</td>
      <td>{{ "%.0f"|format(s.last_price) }}</td>
      <td>{{ "%.2f"|format(s.pnl_pct) }}%</td>
      <td>{{ s.exit_reason }}</td>
      <td>{{ s.next_action }}</td>
    </tr>
  {% endfor %}
  </tbody>
</table>

<h2 class="section-title">포트폴리오 상태</h2>
<table>
  <thead>
    <tr>
      <th>심볼</th>
      <th>종목명</th>
      <th>진입가</th>
      <th>현재가</th>
      <th>P/L</th>
      <th>잔여 리스크</th>
      <th>다음 액션</th>
    </tr>
  </thead>
  <tbody>
  {% for p in portfolio_rows %}
    <tr>
      <td>{{ p.symbol }}</td>
      <td>{{ p.name }}</td>
      <td>{{ "%.0f"|format(p.entry_price) }}</td>
      <td>{{ "%.0f"|format(p.last_price) }}</td>
      <td>{{ "%.2f"|format(p.pnl_pct) }}%</td>
      <td>{{ "%.2f"|format(p.risk_pct) }}%</td>
      <td>{{ p.next_action }}</td>
    </tr>
  {% endfor %}
  </tbody>
</table>

<h2 class="section-title">매수 상세 카드</h2>
{% for c in buy_details %}
<div class="card">
  <div style="display:flex;align-items:center;justify-content:space-between;gap:10px;flex-wrap:wrap">
    <div>
      <div style="font-size:18px;font-weight:700">{{ c.symbol }} · {{ c.name }} <span class="small">({{ c.market }})</span></div>
      <div class="small">점수 {{ "%.2f"|format(c.signal.score) }} · 종가 {{ "%.0f"|format(c.close) }} · ATR {{ "%.1f"|format(c.atr14 or 0) }} · RS {{ c.rs.tag }}</div>
      <div style="margin-top:6px">
        {% for t in c.tags %}
          <span class="badge">{{ t }}</span>
        {% endfor %}
      </div>
    </div>
    <div class="small">
      <div>진입: {{ c.entry_plan.entry_type_label or c.entry_plan.entry_type }} · {{ "%.0f"|format(c.entry_plan.entry_price) }}</div>
      <div>손절: {{ "%.0f"|format(c.entry_plan.stop_loss) }} · 목표: {{ "%.0f"|format(c.entry_plan.take_profit) }}</div>
      <div>RR: {{ "%.2f"|format(c.entry_plan.rr) }} · 기대수익: {{ "%.2f"|format(c.entry_plan.expected_return*100) }}%</div>
    </div>
  </div>

  <div class="grid" style="margin-top:10px">
    <div>
      <img style="width:100%;border-radius:10px;border:1px solid #eee" src="data:image/png;base64,{{ c.chart_b64 }}"/>
    </div>
    <div>
      <div style="font-weight:700;margin-bottom:6px">주요 레벨 / 컨텍스트</div>
      <pre>{{ c.context_text }}</pre>
      <div style="font-weight:700;margin:10px 0 6px 0">게이트 체크</div>
      <pre>{{ c.gate_text }}</pre>
      <div style="font-weight:700;margin:10px 0 6px 0">점수 분해</div>
      <pre>{{ c.score_text }}</pre>
      <div style="font-weight:700;margin:10px 0 6px 0">진입 사유</div>
      <pre>{{ c.reason_text }}</pre>
      <div class="small" style="margin-top:10px">
        {% for n in c.notes %}
          • {{ n }}<br/>
        {% endfor %}
      </div>
    </div>
  </div>
</div>
{% endfor %}

<h2 class="section-title">매도 상세 카드</h2>
{% for c in sell_details %}
<div class="card">
  <div style="display:flex;align-items:center;justify-content:space-between;gap:10px;flex-wrap:wrap">
    <div>
      <div style="font-size:18px;font-weight:700">{{ c.symbol }} · {{ c.name }} <span class="small">({{ c.market }})</span></div>
      <div class="small">진입 {{ "%.0f"|format(c.position.entry_price) }} · 현재 {{ "%.0f"|format(c.close) }} · P/L {{ "%.2f"|format(c.pnl_pct) }}%</div>
      <div style="margin-top:6px">
        {% for t in c.tags %}
          <span class="badge">{{ t }}</span>
        {% endfor %}
      </div>
    </div>
    <div class="small">
      <div>손절: {{ "%.0f"|format(c.position.stop_loss) }} · 목표: {{ "%.0f"|format(c.position.take_profit) }}</div>
      <div>보유: {{ c.position.hold_days }}일 · 다음: {{ c.next_action }}</div>
    </div>
  </div>

  <div class="grid" style="margin-top:10px">
    <div>
      <img style="width:100%;border-radius:10px;border:1px solid #eee" src="data:image/png;base64,{{ c.chart_b64 }}"/>
    </div>
    <div>
      <div style="font-weight:700;margin-bottom:6px">청산 사유</div>
      <pre>{{ c.reason_text }}</pre>
      <div style="font-weight:700;margin:10px 0 6px 0">점수 분해</div>
      <pre>{{ c.score_text }}</pre>
    </div>
  </div>
</div>
{% endfor %}
</body>
</html>""")

def render_report(out_path: str, payload: Dict[str,Any], include_js: bool=True) -> None:
    html = HTML_TMPL.render(**payload, include_js=include_js)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
