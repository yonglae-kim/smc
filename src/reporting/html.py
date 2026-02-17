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
body{
  font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
  margin:0;
  color:#0f172a;
  background:#f5f7fb;
}
.container{max-width:1200px;margin:0 auto;padding:28px 20px 60px}
.header{
  background:linear-gradient(135deg,#0f172a,#1d4ed8);
  color:#fff;
  border-radius:18px;
  padding:22px 24px;
  box-shadow:0 14px 30px rgba(15,23,42,0.2);
}
h1{margin:0;font-size:28px;letter-spacing:-0.4px}
.subtitle{margin-top:8px;font-size:13px;color:rgba(255,255,255,0.8)}
.small{color:#64748b;font-size:12px}
.badge{
  display:inline-block;
  padding:2px 10px;
  border-radius:999px;
  font-size:12px;
  margin:0 6px 6px 0;
  background:#e2e8f0;
  color:#334155;
}
.card{
  border:1px solid #e2e8f0;
  border-radius:16px;
  padding:16px;
  margin:16px 0;
  background:#fff;
  box-shadow:0 8px 18px rgba(15,23,42,0.06);
}
.grid{display:grid;grid-template-columns:1fr;gap:12px}
@media(min-width:1000px){.grid{grid-template-columns:1.25fr 1fr}}
table{border-collapse:separate;border-spacing:0;width:100%;min-width:720px}
th,td{
  border-bottom:1px solid #e2e8f0;
  padding:10px 8px;
  text-align:left;
  font-size:12px;
  vertical-align:top;
  line-height:1.4;
}
th{
  position:sticky;
  top:0;
  background:#f8fafc;
  color:#1e293b;
  font-weight:600;
}
tr:hover{background:#f1f5f9}
input{
  padding:10px 12px;
  border:1px solid #cbd5f5;
  border-radius:10px;
  width:360px;
  background:#fff;
}
pre{white-space:pre-wrap;margin:0;font-size:12px;color:#1f2937}
.kpi{display:flex;gap:12px;flex-wrap:wrap;margin:12px 0 0}
.kpi .card{margin:0;padding:12px 14px}
tbody tr:nth-child(even){background:#f8fafc}
.section-title{margin-top:26px;font-size:18px;color:#0f172a}
.table-wrap{
  overflow-x:auto;
  -webkit-overflow-scrolling:touch;
  background:#fff;
  border-radius:14px;
  border:1px solid #e2e8f0;
  padding:6px;
  box-shadow:0 6px 16px rgba(15,23,42,0.04);
}
.meta-row{
  display:flex;
  flex-wrap:wrap;
  gap:8px 14px;
  margin-top:8px;
  color:#e2e8f0;
  font-size:12px;
}
.desktop-only{display:block}
.mobile-only{display:none}
.mobile-candidate-list{display:grid;gap:10px}
.mobile-candidate-card{
  border:1px solid #e2e8f0;
  border-radius:12px;
  padding:12px;
  background:#fff;
}
.mobile-candidate-head{
  display:grid;
  grid-template-columns:auto auto 1fr;
  gap:6px 10px;
  align-items:center;
}
.mobile-candidate-rank{font-size:18px;font-weight:800;color:#0f172a}
.mobile-candidate-score{font-size:18px;font-weight:800;color:#1d4ed8}
.mobile-candidate-main{grid-column:3/4;min-width:0}
.mobile-candidate-symbol-line{display:flex;align-items:baseline;gap:6px;flex-wrap:wrap}
.mobile-candidate-symbol{font-size:16px;font-weight:800;color:#0f172a}
.mobile-candidate-name{font-size:13px;color:#334155}
.mobile-candidate-entry{font-size:11px;color:#64748b}
.mobile-candidate-price{margin-top:4px;font-size:12px;color:#334155}
.mobile-candidate-gate{margin-top:8px;font-size:12px;color:#475569}
.mobile-candidate-card details{margin-top:8px}
.mobile-candidate-card summary{cursor:pointer;font-size:12px;color:#1d4ed8}
.mobile-candidate-detail{margin-top:8px;padding:10px;background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px}
.mobile-candidate-detail img{width:100%;border-radius:10px;border:1px solid #e2e8f0}
.progressive-hidden{display:none !important}
.more-btn{margin-top:10px;border:1px solid #cbd5e1;background:#fff;color:#1e293b;border-radius:10px;padding:8px 12px;font-size:12px;cursor:pointer}
.decision-strip{
  margin-top:18px;
  display:grid;
  grid-template-columns:repeat(3,minmax(0,1fr));
  gap:12px;
}
.story-strip{
  margin-top:16px;
  display:grid;
  grid-template-columns:repeat(3,minmax(0,1fr));
  gap:12px;
}
.story-card{
  background:#ffffff;
  border:1px solid #dbe4f4;
  border-radius:14px;
  padding:14px;
  box-shadow:0 8px 18px rgba(15,23,42,0.06);
}
.story-title{font-size:12px;color:#64748b;font-weight:700}
.story-metric{margin-top:8px;font-size:24px;font-weight:800;color:#0f172a;display:flex;align-items:center;gap:8px}
.story-description{margin-top:8px;font-size:13px;color:#334155;line-height:1.45}
.story-note{margin-top:8px}
.story-note summary{cursor:pointer;font-size:12px;color:#1d4ed8;font-weight:600}
.story-note div{margin-top:6px;font-size:12px;color:#475569;line-height:1.4}
.motion-ready .story-card{
  opacity:0;
  transform:translateY(10px);
  transition:opacity .45s ease, transform .45s ease;
}
.motion-ready .story-card.is-visible{
  opacity:1;
  transform:translateY(0);
}
.decision-block{
  background:#fff;
  border:1px solid #dbe4f4;
  border-radius:14px;
  padding:14px;
  box-shadow:0 8px 18px rgba(15,23,42,0.06);
}
.decision-label{font-size:12px;color:#64748b;font-weight:600}
.decision-value{margin-top:8px;font-size:15px;color:#0f172a;font-weight:700;line-height:1.5}
.token{
  display:inline-flex;
  align-items:center;
  padding:2px 8px;
  border-radius:999px;
  font-size:11px;
  font-weight:700;
  margin-left:6px;
}
.token-low{background:#dcfce7;color:#166534}
.token-mid{background:#fef3c7;color:#92400e}
.token-high{background:#fee2e2;color:#991b1b}

.toolbar{margin:10px 0 12px 0}
.toolbar-row{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.quick-chips{display:flex;gap:8px;flex-wrap:wrap;margin-top:8px}
.chip-btn{
  border:1px solid #cbd5e1;
  background:#fff;
  color:#334155;
  border-radius:999px;
  padding:6px 10px;
  font-size:12px;
  cursor:pointer;
}
.chip-btn.active{background:#dbeafe;border-color:#93c5fd;color:#1e3a8a}
.status-badges{margin-top:8px;display:flex;gap:6px;flex-wrap:wrap}
.status-badge{display:inline-flex;align-items:center;font-size:11px;color:#1e293b;background:#e2e8f0;border-radius:999px;padding:4px 8px}
.sort-btn{border:1px solid #bfdbfe;background:#eff6ff;color:#1d4ed8;border-radius:10px;padding:10px 12px;font-size:12px;font-weight:600;cursor:pointer}
.sort-modal{position:fixed;inset:0;background:rgba(15,23,42,0.48);display:none;align-items:flex-end;z-index:30}
.sort-modal.open{display:flex}
.sort-sheet{width:100%;background:#fff;border-radius:16px 16px 0 0;padding:16px;box-shadow:0 -8px 28px rgba(15,23,42,0.2)}
.sort-option{width:100%;text-align:left;border:1px solid #e2e8f0;background:#fff;border-radius:10px;padding:10px;margin-top:8px;font-size:13px}

@media(min-width:721px){
  .sort-modal.open{display:none}
}

@media(max-width:720px){
  body.mobile-lite{background:#f8fafc}
  body.mobile-lite .header,
  body.mobile-lite .card,
  body.mobile-lite .story-card,
  body.mobile-lite .decision-block,
  body.mobile-lite .table-wrap,
  body.mobile-lite .mobile-candidate-card{box-shadow:none}
  body.mobile-lite .card,
  body.mobile-lite .story-card,
  body.mobile-lite .decision-block,
  body.mobile-lite .table-wrap,
  body.mobile-lite .mobile-candidate-card{border-color:#e2e8f0;border-radius:10px}
}
.inline-detail{padding:10px;background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px}
.desktop-sort{cursor:pointer}
@media(max-width:720px){
  .container{padding:18px 14px 40px}
  h1{font-size:22px}
  input{width:100%}
  table{min-width:640px}
  th,td{font-size:11px;padding:8px 6px}
  .card{padding:14px}
  .desktop-only{display:none}
  .mobile-only{display:block}
  .decision-strip{
    display:flex;
    overflow-x:auto;
    gap:10px;
    padding-bottom:6px;
    scroll-snap-type:x mandatory;
    -webkit-overflow-scrolling:touch;
  }
  .story-strip{grid-template-columns:1fr}
  .decision-block{min-width:82%;scroll-snap-align:start}
  .toolbar-row{align-items:stretch}
  .desktop-sort{pointer-events:none;color:#64748b}
}

@media (prefers-reduced-motion: reduce){
  .motion-ready .story-card,
  .motion-ready .story-card.is-visible{
    opacity:1;
    transform:none;
    transition:none;
  }
}

</style>
{% if include_js %}
<script>
const tableState={query:"",quickFilter:"all",sortColumn:1,sortDir:"desc"};
function initProgressiveRows(group, step=20){
  const rows=Array.from(document.querySelectorAll(`[data-progressive-group="${group}"]`));
  if(!rows.length) return;
  let visible=Math.min(step, rows.length);
  const apply=()=>rows.forEach((row, idx)=>row.classList.toggle("progressive-hidden", idx>=visible));
  apply();
  const btn=document.querySelector(`[data-more-button="${group}"]`);
  if(!btn){return;}
  if(rows.length<=step){btn.style.display="none";return;}
  btn.addEventListener("click", ()=>{
    visible=Math.min(visible+step, rows.length);
    apply();
    if(visible>=rows.length) btn.style.display="none";
  });
}

function sortTable(n, forcedDir){
  const table=document.getElementById("uTable");
  const tbody=table.querySelector("tbody");
  const rows=Array.from(tbody.querySelectorAll("tr.data-row"));
  const dir = forcedDir || ((tableState.sortColumn===n && tableState.sortDir==="desc") ? "asc" : "desc");
  rows.sort((a,b)=>{
    const x=a.getElementsByTagName("TD")[n];
    const y=b.getElementsByTagName("TD")[n];
    const xvRaw=x.getAttribute("data-sort")||x.innerText;
    const yvRaw=y.getAttribute("data-sort")||y.innerText;
    const xvNum=parseFloat(xvRaw);
    const yvNum=parseFloat(yvRaw);
    const bothNum=!Number.isNaN(xvNum)&&!Number.isNaN(yvNum);
    let cmp=0;
    if(bothNum) cmp=xvNum-yvNum;
    else cmp=String(xvRaw).localeCompare(String(yvRaw));
    return dir==="asc" ? cmp : -cmp;
  });
  rows.forEach((row)=>{
    const detail=row.nextElementSibling;
    tbody.appendChild(row);
    if(detail && detail.classList.contains("detail-row")) tbody.appendChild(detail);
  });
  tableState.sortColumn=n;
  tableState.sortDir=dir;
  updateStatusBadges();
  applyAllFilters();
}
function filterTable(){
  tableState.query=document.getElementById("q").value.toLowerCase();
  applyAllFilters();
  updateStatusBadges();
}
function applyQuickFilter(type){
  tableState.quickFilter=type;
  document.querySelectorAll(".chip-btn").forEach((btn)=>btn.classList.toggle("active", btn.dataset.filter===type));
  applyAllFilters();
  updateStatusBadges();
}
function applyAllFilters(){
  const table=document.getElementById("uTable");
  if(!table) return;
  const rows=table.querySelectorAll("tbody tr.data-row");
  rows.forEach((row)=>{
    const txt=row.innerText.toLowerCase();
    const detail=row.nextElementSibling;
    const hitQuery=txt.indexOf(tableState.query)>-1;
    const score=parseFloat(row.dataset.score||"0");
    const rank=parseFloat(row.dataset.rank||"9999");
    const gate=row.dataset.gate||"";
    let hitQuick=true;
    if(tableState.quickFilter==="rr2") hitQuick=score>=2;
    if(tableState.quickFilter==="scoreTop") hitQuick=rank<=50;
    if(tableState.quickFilter==="gatePass") hitQuick=gate==="1";
    const visible=hitQuery && hitQuick;
    row.style.display=visible ? "" : "none";
    if(detail && detail.classList.contains("detail-row")) detail.style.display=(visible && row.dataset.expanded==="1") ? "" : "none";
  });
}
function openSortSheet(){ const el=document.getElementById("sortModal"); if(el) el.classList.add("open"); }
function closeSortSheet(){ const el=document.getElementById("sortModal"); if(el) el.classList.remove("open"); }
function sortByOption(col, dir){ sortTable(col, dir); closeSortSheet(); }
function updateStatusBadges(){
  const queryBadge=document.getElementById("queryBadge");
  const filterBadge=document.getElementById("filterBadge");
  const sortBadge=document.getElementById("sortBadge");
  if(queryBadge) queryBadge.innerText=tableState.query ? `검색: ${tableState.query}` : "검색: 전체";
  const filterMap={all:"필터: 없음",rr2:"필터: RR 2.0+",scoreTop:"필터: 점수 상위",gatePass:"필터: 게이트 통과"};
  if(filterBadge) filterBadge.innerText=filterMap[tableState.quickFilter] || filterMap.all;
  const sortMap={0:"순위",1:"점수",6:"종가",7:"MA20",8:"MA200",9:"Slope20%",10:"RSI"};
  if(sortBadge) sortBadge.innerText=`정렬: ${sortMap[tableState.sortColumn]||"점수"} ${tableState.sortDir==="asc"?"오름차순":"내림차순"}`;
}
function openRowDetail(row){
  const detail=row.nextElementSibling;
  if(!detail || !detail.classList.contains("detail-row")) return;
  const isOpen=row.dataset.expanded==="1";
  row.dataset.expanded=isOpen ? "0" : "1";
  detail.style.display=isOpen ? "none" : "";
}
function initStoryCards(){
  const cards=document.querySelectorAll(".story-card");
  if(!cards.length) return;
  if(window.matchMedia("(prefers-reduced-motion: reduce)").matches){
    cards.forEach((card)=>card.classList.add("is-visible"));
    return;
  }
  document.body.classList.add("motion-ready");
  const observer=new IntersectionObserver((entries)=>{
    entries.forEach((entry)=>{
      if(entry.isIntersecting){
        entry.target.classList.add("is-visible");
        observer.unobserve(entry.target);
      }
    });
  }, {threshold:0.18, rootMargin:"0px 0px -8% 0px"});
  cards.forEach((card)=>observer.observe(card));
}
document.addEventListener("DOMContentLoaded",()=>{
  updateStatusBadges();
  applyQuickFilter("all");
  initStoryCards();
  initProgressiveRows("watch-row-desktop");
  initProgressiveRows("watch-row-mobile");
  initProgressiveRows("table-row");
  initProgressiveRows("trade-row");
});
</script>
{% endif %}
</head>
<body class="{% if mobile_light_mode %}mobile-lite{% endif %}">
<div class="container">
  <div class="header">
    <h1>{{ title }}</h1>
    <div class="subtitle">오늘의 SMC 시그널 요약 리포트</div>
    <div class="meta-row">
      <span>생성 시각 {{ generated_at }} (KST)</span>
      <span>유니버스: 유동성 상위 {{ universe_n }}개</span>
      <span>중위값 기준 {{ liquidity_window }}일</span>
    </div>
  </div>

  {% set top_picks = immediate_buy_rows[:3] %}
  {% set total_rr = namespace(v=0) %}
  {% set pass_total = namespace(v=0) %}
  {% set gate_total = namespace(v=0) %}
  {% for b in immediate_buy_rows %}
    {% set total_rr.v = total_rr.v + (b.entry_plan.rr or 0) %}
    {% set pass_total.v = pass_total.v + (b.gates|selectattr('pass')|list|length) %}
    {% set gate_total.v = gate_total.v + (b.gates|length) %}
  {% endfor %}
  {% set avg_rr = (total_rr.v / (immediate_buy_rows|length)) if immediate_buy_rows else 0 %}
  {% set gate_rate = ((pass_total.v / gate_total.v) * 100) if gate_total.v else 0 %}
  {% set rr_risk = 'token-low' if avg_rr >= 2 else ('token-mid' if avg_rr >= 1.4 else 'token-high') %}
  {% set gate_risk = 'token-low' if gate_rate >= 80 else ('token-mid' if gate_rate >= 60 else 'token-high') %}
  {% set action_head = top_picks[0].symbol if top_picks else 'Top Pick 없음' %}
  {% set trend = namespace(v=0) %}
  {% for r in table_rows %}
    {% if r.ma20 and r.ma200 and r.close and r.close > r.ma20 and r.ma20 > r.ma200 %}
      {% set trend.v = trend.v + 1 %}
    {% endif %}
  {% endfor %}
  {% set trend_ratio = ((trend.v / (table_rows|length)) * 100) if table_rows else 0 %}
  {% set watch_count = pullback_buy_rows|length %}
  {% set immediate_count = immediate_buy_rows|length %}
  {% set watch_surge = watch_count >= (immediate_count + 5) %}

  <div class="story-strip" aria-label="story-card-strip">
    <div class="story-card">
      <div class="story-title">추세 우위 종목 비율</div>
      <div class="story-metric">📈 {{ "%.1f"|format(trend_ratio) }}%</div>
      <div class="story-description">MA20 &gt; MA200 위에서 종가가 유지되는 비율입니다.</div>
      <details class="story-note">
        <summary>왜 중요한가?</summary>
        <div>추세 정렬이 잘 된 종목 비중이 높을수록 돌파·추세 추종 전략의 성공 확률이 개선됩니다.</div>
      </details>
    </div>
    <div class="story-card">
      <div class="story-title">즉시 진입 평균 RR</div>
      <div class="story-metric">⚖️ {{ "%.2f"|format(avg_rr) }}</div>
      <div class="story-description">오늘 즉시 진입 후보의 기대 보상/위험 평균입니다.</div>
      <details class="story-note">
        <summary>왜 중요한가?</summary>
        <div>평균 RR이 높을수록 동일 손실 대비 기대수익 여지가 커져 포트폴리오 효율에 유리합니다.</div>
      </details>
    </div>
    <div class="story-card">
      <div class="story-title">관망 종목 급증 여부</div>
      <div class="story-metric">👀 {{ '급증' if watch_surge else '안정' }} ({{ watch_count }}개)</div>
      <div class="story-description">되돌림 대기 후보 수가 즉시 진입 후보보다 크게 많으면 급증으로 표시합니다.</div>
      <details class="story-note">
        <summary>왜 중요한가?</summary>
        <div>관망 후보 급증은 단기 과열·진입 타이밍 미스매치 신호일 수 있어 추격 매수 리스크 점검이 필요합니다.</div>
      </details>
    </div>
  </div>

  <div class="decision-strip" aria-label="decision-strip">
    <div class="decision-block">
      <div class="decision-label">오늘의 Top Pick 1~3</div>
      <div class="decision-value">
        {% if top_picks %}
          {% for p in top_picks %}
            {{ loop.index }}) {{ p.symbol }}{% if not loop.last %}<br/>{% endif %}
          {% endfor %}
        {% else %}
          즉시 진입 후보 없음
        {% endif %}
      </div>
    </div>
    <div class="decision-block">
      <div class="decision-label">평균 RR / 게이트 통과율</div>
      <div class="decision-value">
        RR {{ "%.2f"|format(avg_rr) }}
        <span class="token {{ rr_risk }}">{{ '낮음' if rr_risk == 'token-low' else ('보통' if rr_risk == 'token-mid' else '높음') }}</span><br/>
        게이트 {{ "%.1f"|format(gate_rate) }}%
        <span class="token {{ gate_risk }}">{{ '낮음' if gate_risk == 'token-low' else ('보통' if gate_risk == 'token-mid' else '높음') }}</span>
      </div>
    </div>
    <div class="decision-block">
      <div class="decision-label">지금 할 일</div>
      <div class="decision-value">{{ buy_valid_from }}부터 {{ action_head }} 우선 점검 · {{ execution_guide }}</div>
    </div>
  </div>

  <div class="card" style="margin-top:18px">
    <div style="font-weight:700;font-size:14px">실행 가이드</div>
    <div class="small" style="margin-top:6px">{{ execution_guide }}</div>
    <div class="small" style="margin-top:8px">가정: {{ tp_sl_conflict_note }}</div>
  </div>

<h2 class="section-title">매수 후보 (다음 세션)</h2>
<div class="small">시그널은 종가 기준 산출, {{ buy_valid_from }}부터 유효.</div>

<h3 class="section-title">즉시 진입 후보</h3>
<div class="table-wrap desktop-only">
  <table>
    <thead>
      <tr>
        <th>순위</th>
        <th>점수</th>
        <th>심볼</th>
        <th>종목명</th>
        <th>진입 타입</th>
        <th>진입가</th>
        <th>손절</th>
        <th>목표</th>
        <th>RR</th>
        <th>게이트</th>
      </tr>
    </thead>
    <tbody>
    {% for b in immediate_buy_rows %}
      <tr>
        <td>{{ b.rank }}</td>
        <td>{{ "%.2f"|format(b.signal.score) }}</td>
        <td>{{ b.symbol }}</td>
        <td>{{ b.name }}</td>
        <td>타입 {{ b.entry_plan.entry_type_label or b.entry_plan.entry_type }}</td>
        <td>{{ "%.0f"|format(b.entry_plan.entry_price) }}</td>
        <td>{{ "%.0f"|format(b.entry_plan.stop_loss) }}</td>
        <td>{{ "%.0f"|format(b.entry_plan.take_profit) }}</td>
        <td>{{ "%.2f"|format(b.entry_plan.rr) }}</td>
        <td>
          {% set total = b.gates|length %}
          {% set passed = b.gates|selectattr('pass')|list|length %}
          {% set failed_keys = b.gates|rejectattr('pass')|map(attribute='key')|list %}
          {{ passed }}/{{ total }}
          {% if failed_keys %}
            ({{ failed_keys[:2]|join(', ') }})
          {% endif %}
        </td>
      </tr>
    {% endfor %}
    </tbody>
  </table>
</div>
<div class="mobile-only mobile-candidate-list">
  {% for b in immediate_buy_rows %}
  <div class="mobile-candidate-card">
    {% set total = b.gates|length %}
    {% set passed = b.gates|selectattr('pass')|list|length %}
    {% set failed_keys = b.gates|rejectattr('pass')|map(attribute='key')|list %}
    {% set detail = (buy_details|selectattr('symbol', 'equalto', b.symbol)|list|first) %}
    <div class="mobile-candidate-head">
      <div class="mobile-candidate-rank">#{{ b.rank }}</div>
      <div class="mobile-candidate-score">{{ "%.2f"|format(b.signal.score) }}</div>
      <div class="mobile-candidate-main">
        <div class="mobile-candidate-symbol-line">
          <span class="mobile-candidate-symbol">{{ b.symbol }}</span>
          <span class="mobile-candidate-name">{{ b.name }}</span>
          <span class="mobile-candidate-entry">타입 {{ b.entry_plan.entry_type_label or b.entry_plan.entry_type }}</span>
        </div>
        <div class="mobile-candidate-price">진입 {{ "%.0f"|format(b.entry_plan.entry_price) }} · 손절 {{ "%.0f"|format(b.entry_plan.stop_loss) }} · 목표 {{ "%.0f"|format(b.entry_plan.take_profit) }}</div>
      </div>
    </div>
    <div class="mobile-candidate-gate">게이트 {{ passed }}/{{ total }}{% if failed_keys %} · 실패 {{ failed_keys[:2]|join(', ') }}{% endif %}</div>
    <details>
      <summary>상세 보기</summary>
      {% if detail %}
      <div class="mobile-candidate-detail">
        <img src="{{ detail.chart_src }}" alt="{{ detail.symbol }} 차트"/>
        <div class="small" style="margin-top:8px">RR {{ "%.2f"|format(detail.entry_plan.rr) }} · 기대수익 {{ "%.2f"|format(detail.entry_plan.expected_return*100) }}%</div>
        <div style="font-weight:700;margin:8px 0 4px 0">진입 사유</div>
        <pre>{{ detail.reason_text }}</pre>
      </div>
      {% else %}
      <div class="small" style="margin-top:6px">상세 데이터가 없습니다.</div>
      {% endif %}
    </details>
  </div>
  {% endfor %}
</div>

<h3 class="section-title">되돌림 대기 후보</h3>
<div class="table-wrap desktop-only">
  <table>
    <thead>
      <tr>
        <th>순위</th>
        <th>점수</th>
        <th>심볼</th>
        <th>종목명</th>
        <th>진입 타입</th>
        <th>진입가</th>
        <th>손절</th>
        <th>목표</th>
        <th>RR</th>
        <th>게이트</th>
      </tr>
    </thead>
    <tbody>
  {% for b in pullback_buy_rows %}
      <tr data-progressive-group="watch-row-desktop">
        <td>{{ b.rank }}</td>
        <td>{{ "%.2f"|format(b.signal.score) }}</td>
        <td>{{ b.symbol }}</td>
        <td>{{ b.name }}</td>
        <td>타입 {{ b.entry_plan.entry_type_label or b.entry_plan.entry_type }}</td>
        <td>{{ "%.0f"|format(b.entry_plan.entry_price) }}</td>
        <td>{{ "%.0f"|format(b.entry_plan.stop_loss) }}</td>
        <td>{{ "%.0f"|format(b.entry_plan.take_profit) }}</td>
        <td>{{ "%.2f"|format(b.entry_plan.rr) }}</td>
        <td>
          {% set total = b.gates|length %}
          {% set passed = b.gates|selectattr('pass')|list|length %}
          {% set failed_keys = b.gates|rejectattr('pass')|map(attribute='key')|list %}
          {{ passed }}/{{ total }}
          {% if failed_keys %}
            ({{ failed_keys[:2]|join(', ') }})
          {% endif %}
        </td>
      </tr>
    {% endfor %}
    </tbody>
  </table>
</div>
<div class="mobile-only mobile-candidate-list">
  {% for b in pullback_buy_rows %}
  <div class="mobile-candidate-card" data-progressive-group="watch-row-mobile">
    {% set total = b.gates|length %}
    {% set passed = b.gates|selectattr('pass')|list|length %}
    {% set failed_keys = b.gates|rejectattr('pass')|map(attribute='key')|list %}
    {% set detail = (buy_details|selectattr('symbol', 'equalto', b.symbol)|list|first) %}
    <div class="mobile-candidate-head">
      <div class="mobile-candidate-rank">#{{ b.rank }}</div>
      <div class="mobile-candidate-score">{{ "%.2f"|format(b.signal.score) }}</div>
      <div class="mobile-candidate-main">
        <div class="mobile-candidate-symbol-line">
          <span class="mobile-candidate-symbol">{{ b.symbol }}</span>
          <span class="mobile-candidate-name">{{ b.name }}</span>
          <span class="mobile-candidate-entry">타입 {{ b.entry_plan.entry_type_label or b.entry_plan.entry_type }}</span>
        </div>
        <div class="mobile-candidate-price">진입 {{ "%.0f"|format(b.entry_plan.entry_price) }} · 손절 {{ "%.0f"|format(b.entry_plan.stop_loss) }} · 목표 {{ "%.0f"|format(b.entry_plan.take_profit) }}</div>
      </div>
    </div>
    <div class="mobile-candidate-gate">게이트 {{ passed }}/{{ total }}{% if failed_keys %} · 실패 {{ failed_keys[:2]|join(', ') }}{% endif %}</div>
    <details>
      <summary>상세 보기</summary>
      {% if detail %}
      <div class="mobile-candidate-detail">
        <img src="{{ detail.chart_src }}" alt="{{ detail.symbol }} 차트"/>
        <div class="small" style="margin-top:8px">RR {{ "%.2f"|format(detail.entry_plan.rr) }} · 기대수익 {{ "%.2f"|format(detail.entry_plan.expected_return*100) }}%</div>
        <div style="font-weight:700;margin:8px 0 4px 0">진입 사유</div>
        <pre>{{ detail.reason_text }}</pre>
      </div>
      {% else %}
      <div class="small" style="margin-top:6px">상세 데이터가 없습니다.</div>
      {% endif %}
    </details>
  </div>
  {% endfor %}
</div>
<button class="more-btn desktop-only" type="button" data-more-button="watch-row-desktop">관망 후보 더 보기</button>
<button class="more-btn mobile-only" type="button" data-more-button="watch-row-mobile">관망 후보 더 보기</button>

<h3 class="section-title">관망 후보</h3>
<div class="small">현재 관망 후보가 없습니다.</div>

<h2 class="section-title">매도 후보 (리스크 관리)</h2>
<div class="small">보유 포지션 기준으로만 산출.</div>
<div class="table-wrap">
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
      <tr data-progressive-group="trade-row">
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
</div>
<button class="more-btn" type="button" data-more-button="trade-row">거래내역 더 보기</button>

<h2 class="section-title">포트폴리오 상태</h2>
<div class="table-wrap">
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
</div>

<h2 class="section-title">유니버스 요약</h2>
{% if include_js %}
<div class="toolbar">
  <div class="toolbar-row">
    <input id="q" onkeyup="filterTable()" placeholder="심볼/이름/태그 검색..."/>
    <button class="sort-btn mobile-only" type="button" onclick="openSortSheet()">정렬</button>
  </div>
  <div class="quick-chips">
    <button class="chip-btn active" type="button" data-filter="all" onclick="applyQuickFilter('all')">전체</button>
    <button class="chip-btn" type="button" data-filter="rr2" onclick="applyQuickFilter('rr2')">RR 2.0+</button>
    <button class="chip-btn" type="button" data-filter="scoreTop" onclick="applyQuickFilter('scoreTop')">점수 상위</button>
    <button class="chip-btn" type="button" data-filter="gatePass" onclick="applyQuickFilter('gatePass')">게이트 통과</button>
  </div>
  <div class="status-badges">
    <span id="queryBadge" class="status-badge">검색: 전체</span>
    <span id="filterBadge" class="status-badge">필터: 없음</span>
    <span id="sortBadge" class="status-badge">정렬: 점수 내림차순</span>
  </div>
</div>
{% endif %}
<div class="table-wrap">
  <table id="uTable">
    <thead>
      <tr>
        <th class="desktop-sort" onclick="sortTable(0)">순위</th>
        <th class="desktop-sort" onclick="sortTable(1)">점수</th>
        <th>심볼</th>
        <th>종목명</th>
        <th>시장</th>
        <th>태그</th>
        <th class="desktop-sort" onclick="sortTable(6)">종가</th>
        <th class="desktop-sort" onclick="sortTable(7)">MA20</th>
        <th class="desktop-sort" onclick="sortTable(8)">MA200</th>
        <th class="desktop-sort" onclick="sortTable(9)">Slope20%</th>
        <th class="desktop-sort" onclick="sortTable(10)">RSI</th>
        <th>레벨</th>
      </tr>
    </thead>
    <tbody>
    {% for r in table_rows %}
      {% set gate_pass = 1 if ('gate_pass' in (r.tags|join(' ')|lower) or 'pass' in (r.tags|join(' ')|lower)) else 0 %}
      <tr class="data-row" data-progressive-group="table-row" data-symbol="{{ r.symbol|lower }}" data-score="{{ r.score }}" data-rank="{{ r.rank }}" data-gate="{{ gate_pass }}" data-expanded="0" onclick="openRowDetail(this)">
        <td data-sort="{{ r.rank }}">{{ r.rank }}</td>
        <td data-sort="{{ r.score }}">{{ "%.1f"|format(r.score) }}</td>
        <td>{{ r.symbol }}</td>
        <td>{{ r.name }}</td>
        <td>{{ r.market }}</td>
        <td>{{ r.tags|join(", ") }}</td>
        <td data-sort="{{ r.close }}">{{ "%.0f"|format(r.close) }}</td>
        <td data-sort="{{ r.ma20 or 0 }}">{{ "%.0f"|format(r.ma20) if r.ma20 else "" }}</td>
        <td data-sort="{{ r.ma200 or 0 }}">{{ "%.0f"|format(r.ma200) if r.ma200 else "" }}</td>
        <td data-sort="{{ r.ma_slope_pct or 0 }}">{{ "%.2f"|format(r.ma_slope_pct * 100) if r.ma_slope_pct is not none else "" }}</td>
        <td data-sort="{{ r.rsi14 or 0 }}">{{ "%.1f"|format(r.rsi14) if r.rsi14 else "" }}</td>
        <td>{{ r.levels }}</td>
      </tr>
      <tr class="detail-row" style="display:none">
        <td colspan="12">
          <div class="inline-detail">
            <div style="font-weight:700">{{ r.symbol }} · {{ r.name }}</div>
            <div class="small" style="margin-top:4px">시장 {{ r.market }} · 점수 {{ "%.1f"|format(r.score) }} · 종가 {{ "%.0f"|format(r.close) }}</div>
            <div class="small" style="margin-top:4px">태그 {{ r.tags|join(', ') }} · 레벨 {{ r.levels }}</div>
          </div>
        </td>
      </tr>
    {% endfor %}
    </tbody>
  </table>
</div>
<button class="more-btn" type="button" data-more-button="table-row">더 보기</button>
<div id="sortModal" class="sort-modal" onclick="if(event.target===this) closeSortSheet()">
  <div class="sort-sheet">
    <div style="font-weight:700">정렬 기준</div>
    <button class="sort-option" type="button" onclick="sortByOption(1,'desc')">점수 높은 순</button>
    <button class="sort-option" type="button" onclick="sortByOption(1,'asc')">점수 낮은 순</button>
    <button class="sort-option" type="button" onclick="sortByOption(0,'asc')">순위 빠른 순</button>
    <button class="sort-option" type="button" onclick="sortByOption(10,'desc')">RSI 높은 순</button>
    <button class="sort-option" type="button" onclick="closeSortSheet()">닫기</button>
  </div>
</div>
</div>
</body>
</html>""")

def render_report(out_path: str, payload: Dict[str,Any], include_js: bool=True) -> None:
    html = HTML_TMPL.render(**payload, include_js=include_js)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
