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
th,td{border-bottom:1px solid #eee;padding:8px 6px;text-align:left;font-size:12px}
th{position:sticky;top:0;background:#fafafa}
tr:hover{background:#fcfcfc}
input{padding:8px 10px;border:1px solid #ddd;border-radius:8px;width:340px}
pre{white-space:pre-wrap;margin:0;font-size:12px;color:#333}
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
<div class="small">Generated at {{ generated_at }} (KST) · Universe: Top{{ universe_n }} liquidity (median value, {{ liquidity_window }}D) · Detail cards: Top{{ detail_n }}</div>

<h2>Market Regime</h2>
<div>
  <span class="badge">KOSPI: {{ regime_kospi.tag }} · MA200={{ "Above" if regime_kospi.above_ma200 else "Below" }} · RSI50={{ "Above" if regime_kospi.rsi_ge_50 else "Below" }} · ATR spike={{ regime_kospi.atr_spike }}</span>
  <span class="badge">KOSDAQ: {{ regime_kosdaq.tag }} · MA200={{ "Above" if regime_kosdaq.above_ma200 else "Below" }} · RSI50={{ "Above" if regime_kosdaq.rsi_ge_50 else "Below" }} · ATR spike={{ regime_kosdaq.atr_spike }}</span>
</div>

<h2>Top500 Summary</h2>
{% if include_js %}
<div style="margin:8px 0 10px 0">
  <input id="q" onkeyup="filterTable()" placeholder="Search symbol/name/tag..."/>
</div>
{% endif %}
<table id="uTable">
  <thead>
    <tr>
      <th onclick="sortTable(0)">Rank</th>
      <th onclick="sortTable(1)">Score</th>
      <th>Symbol</th>
      <th>Name</th>
      <th>Market</th>
      <th>Tags</th>
      <th onclick="sortTable(6)">Close</th>
      <th onclick="sortTable(7)">MA200</th>
      <th onclick="sortTable(8)">RSI</th>
      <th>Levels</th>
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

<h2>Top{{ detail_n }} Detail Cards</h2>
{% for c in details %}
<div class="card">
  <div style="display:flex;align-items:center;justify-content:space-between;gap:10px;flex-wrap:wrap">
    <div>
      <div style="font-size:18px;font-weight:700">{{ c.symbol }} · {{ c.name }} <span class="small">({{ c.market }})</span></div>
      <div class="small">Score {{ "%.1f"|format(c.score) }} · Close {{ "%.0f"|format(c.close) }} · ATR {{ "%.1f"|format(c.atr14 or 0) }} · RS {{ c.rs.tag }}</div>
      <div style="margin-top:6px">
        {% for t in c.tags %}
          <span class="badge">{{ t }}</span>
        {% endfor %}
      </div>
    </div>
    <div class="small">
      <div>OB dist(ATR): {{ "%.2f"|format(c.dist_to_ob_atr) if c.dist_to_ob_atr is not none else "n/a" }} · status: {{ c.ob.status if c.ob else "n/a" }}</div>
      <div>FVG dist(ATR): {{ "%.2f"|format(c.dist_to_fvg_atr) if c.dist_to_fvg_atr is not none else "n/a" }} · status: {{ c.fvg.status if c.fvg else "n/a" }}</div>
      <div>Invalidation: {{ "%.0f"|format(c.ob.invalidation) if c.ob else "n/a" }}</div>
    </div>
  </div>

  <div class="grid" style="margin-top:10px">
    <div>
      <img style="width:100%;border-radius:10px;border:1px solid #eee" src="data:image/png;base64,{{ c.chart_b64 }}"/>
    </div>
    <div>
      <div style="font-weight:700;margin-bottom:6px">Key Levels / Context</div>
      <pre>{{ c.context_text }}</pre>
      <div style="font-weight:700;margin:10px 0 6px 0">Score breakdown</div>
      <pre>{{ c.score_text }}</pre>
      <div class="small" style="margin-top:10px">
        {% for n in c.notes %}
          • {{ n }}<br/>
        {% endfor %}
      </div>
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
