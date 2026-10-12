"""ボート券種ずれモニター（軽量版：古いスマホのブラウザでも動く1枚の HTML。データを埋め込む）
usage: python scripts/make_dashboard_lite.py COMBOS.json OUT.html
"""
import json, sys

TEMPLATE = r"""<title>券種ずれモニター</title>
<style>
:root{--bg:#f6f7f9;--panel:#ffffff;--fg:#1d2430;--muted:#5d6878;--line:#dfe3ea;--accent:#1f6feb;--accent2:#c2410c;--ok:#15803d;--bad:#b91c1c;--hit:#e7f6ec;--chipok:#dcf3e4;--chipbad:#fbe4e4}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#11151b;--panel:#1a2029;--fg:#e6eaf0;--muted:#9aa5b4;--line:#2c3440;--accent:#5b9dff;--accent2:#f08a4b;--ok:#4ade80;--bad:#f87171;--hit:#183324;--chipok:#1d3a28;--chipbad:#3d1f22;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#11151b;--panel:#1a2029;--fg:#e6eaf0;--muted:#9aa5b4;--line:#2c3440;--accent:#5b9dff;--accent2:#f08a4b;--ok:#4ade80;--bad:#f87171;--hit:#183324;--chipok:#1d3a28;--chipbad:#3d1f22;color-scheme:dark}
body{background:var(--bg);color:var(--fg);font-family:-apple-system,"Hiragino Sans","Hiragino Kaku Gothic ProN",system-ui,sans-serif;font-size:14px;line-height:1.5}
.wrap{max-width:960px;margin:0 auto;padding:16px}
h1{font-size:22px;margin:0 0 4px} h2{font-size:15px;margin:0 0 8px}
.muted{color:var(--muted)} .small{font-size:12px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px;margin-top:12px}
.tiles{display:flex;flex-wrap:wrap;margin:0 -6px}
.tile{flex:1 1 40%;min-width:120px;padding:4px 6px}
.tile .v{font-size:22px;font-variant-numeric:tabular-nums}
.pos{color:var(--ok)} .neg{color:var(--bad)}
select{font-size:14px;padding:4px 6px;margin:0 8px 8px 0;background:var(--bg);color:var(--fg);border:1px solid var(--line);border-radius:6px}
.tbl{overflow-x:auto;-webkit-overflow-scrolling:touch}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:13px}
th,td{padding:5px 6px;text-align:right;white-space:nowrap;border-bottom:1px solid var(--line)}
th{color:var(--muted);font-weight:600;font-size:12px}
th.l,td.l{text-align:left}
tr.hit td{background:var(--hit)}
.chip{padding:0 6px;border-radius:9px}
.chip.good{background:var(--chipok);color:var(--ok)} .chip.bad{background:var(--chipbad);color:var(--bad)}
svg text{fill:var(--muted);font-size:11px}
</style>
<div class="wrap">
  <h1>券種ずれモニター</h1>
  <p class="muted" id="take">—</p>
  <p class="muted small">締切2〜0.5分前のオッズで「3連単の市場から計算した公正なオッズ」と単勝・2連複を比べる。期待値＝3連単から見た確率×その時点のオッズ。仮想売買は期待値1.1超を各100円、払戻は確定オッズ（お金は使っていない）。更新：__UPDATED__</p>
  <div class="card"><h2>2連複の仮想売買</h2><div class="tiles" id="t-q"></div><p class="muted small" id="e-q"></p></div>
  <div class="card"><h2>単勝の仮想売買</h2><div class="tiles" id="t-w"></div><p class="muted small" id="e-w"></p></div>
  <div class="card"><h2>累積損益（円、各100円）</h2><div id="chart"></div></div>
  <div class="card"><h2>大きなずれ（締切2分前の期待値の高い順）</h2>
    <select id="gd" aria-label="日付"></select><select id="gk" aria-label="券種"><option value="">すべて</option><option>2連複</option><option>単勝</option></select>
    <div class="tbl"><table id="gt"></table></div></div>
  <div class="card"><h2>レースを選んで見る</h2>
    <select id="rd" aria-label="日付"></select><select id="rp" aria-label="レース"></select>
    <div class="tbl"><table id="rt"></table></div></div>
</div>
<script type="application/json" id="data">__DATA__</script>
<script>
(function(){
var rows = JSON.parse(document.getElementById('data').textContent);
function fmt(v,d){ return (v===null||v===undefined)?'—':Number(v).toFixed(d); }
function yen(v){ var s=Math.round(v).toString().replace(/\B(?=(\d{3})+(?!\d))/g, ','); return (v>0?'+':'')+s; }
function pct(v){ return (v===null)?'—':(v*100).toFixed(1)+'%'; }
function el(t,c,x){ var e=document.createElement(t); if(c) e.className=c; if(x!==undefined) e.textContent=x; return e; }
function bets(kind){ return rows.filter(function(r){ return r.kind===kind && r.settled && r.ev_t1>1.1 && r.odds_close>0; }); }
function summary(kind){
  var b=bets(kind), ret=0, hits=0, e1=0, ec=0;
  b.forEach(function(r){ if(r.hit){ ret+=r.odds_close*100; hits++; } e1+=r.ev_t1||0; ec+=r.ev_close||0; });
  return {n:b.length, hits:hits, roi:b.length?ret/(b.length*100):null, pnl:ret-b.length*100, e1:b.length?e1/b.length:null, ec:b.length?ec/b.length:null};
}
function tiles(id, s){
  var box=document.getElementById(id); box.innerHTML='';
  [['点数',String(s.n),''],['的中',String(s.hits),''],['回収率',pct(s.roi),s.roi===null?'':(s.roi>=1?'pos':'neg')],['損益（円）',yen(s.pnl),s.pnl>=0?'pos':'neg']].forEach(function(t){
    var d=el('div','tile'); d.appendChild(el('div','muted small',t[0])); d.appendChild(el('div','v '+t[2],t[1])); box.appendChild(d); });
}
var sq=summary('2連複'), sw=summary('単勝');
tiles('t-q',sq); tiles('t-w',sw);
document.getElementById('e-q').textContent='期待値の平均：判定時 '+fmt(sq.e1,2)+' → 確定時 '+fmt(sq.ec,2)+'（確定オッズがそろったレースのみ）';
document.getElementById('e-w').textContent='期待値の平均：判定時 '+fmt(sw.e1,2)+' → 確定時 '+fmt(sw.ec,2)+'（確定オッズがそろったレースのみ）';
document.getElementById('take').textContent = sq.n ? '締切2分前に割安だった2連複を確定オッズで払い戻すと、回収率 '+pct(sq.roi)+'（'+sq.n+'点・'+sq.hits+'的中、損益 '+yen(sq.pnl)+'円）。' : 'まだ確定したレースがありません。';
// 累積損益
(function(){
  var days=[]; rows.forEach(function(r){ if(r.settled && days.indexOf(r.date)<0) days.push(r.date); }); days.sort();
  var box=document.getElementById('chart'); if(!days.length){ box.textContent='確定したレースがまだありません'; return; }
  var series=[['2連複','var(--accent)'],['単勝','var(--accent2)']].map(function(k){
    var cum=0; return {name:k[0], color:k[1], pts:days.map(function(d){ var b=bets(k[0]).filter(function(r){return r.date===d;}); var p=0; b.forEach(function(r){ p+=(r.hit?r.odds_close*100:0)-100; }); cum+=p; return cum; })};
  });
  var W=Math.max(300, box.clientWidth||600), H=200, ml=56, mr=90, mt=10, mb=24;
  var all=[0]; series.forEach(function(s){ all=all.concat(s.pts); });
  var lo=Math.min.apply(null,all), hi=Math.max.apply(null,all); if(hi===lo){ hi+=100; lo-=100; }
  function X(i){ return days.length>1 ? ml+(W-ml-mr)*i/(days.length-1) : ml+(W-ml-mr)/2; }
  function Y(v){ return mt+(H-mt-mb)*(hi-v)/(hi-lo); }
  var s='<svg width="100%" viewBox="0 0 '+W+' '+H+'">';
  s+='<line x1="'+ml+'" x2="'+(W-mr)+'" y1="'+Y(0)+'" y2="'+Y(0)+'" stroke="var(--line)" stroke-dasharray="4 3"/>';
  [lo,0,hi].forEach(function(v){ s+='<text x="'+(ml-6)+'" y="'+(Y(v)+4)+'" text-anchor="end">'+yen(v)+'</text>'; });
  days.forEach(function(d,i){ if(days.length<=7||i%Math.ceil(days.length/7)===0) s+='<text x="'+X(i)+'" y="'+(H-6)+'" text-anchor="middle">'+d.slice(5).replace('-','/')+'</text>'; });
  series.forEach(function(se,j){
    s+='<polyline fill="none" stroke="'+se.color+'" stroke-width="2" stroke-linejoin="round" stroke-linecap="round" points="'+se.pts.map(function(v,i){return X(i)+','+Y(v);}).join(' ')+'"/>';
    se.pts.forEach(function(v,i){ s+='<circle cx="'+X(i)+'" cy="'+Y(v)+'" r="3" fill="'+se.color+'"/>'; });
    var last=se.pts[se.pts.length-1]; s+='<text x="'+(X(se.pts.length-1)+8)+'" y="'+(Y(last)+4+(j?12:-2))+'" style="fill:'+se.color+'">'+se.name+' '+yen(last)+'</text>';
  });
  box.innerHTML=s+'</svg>';
})();
// 表
var COLS=[['race','レース','l'],['deadline','締切',''],['kind','券種','l'],['combo','買い目','l'],['odds_t1','2分前',''],['fair_t1','公正',''],['ev_t1','期待値',''],['odds_close','確定',''],['ev_close','確定時の期待値',''],['hit','的中','']];
function table(id, rs, cols){
  var t=document.getElementById(id); t.innerHTML=''; var tr=el('tr'); cols.forEach(function(c){ tr.appendChild(el('th',c[2],c[1])); }); t.appendChild(tr);
  rs.forEach(function(r){ var tr=el('tr', r.hit?'hit':''); cols.forEach(function(c){ var k=c[0], td=el('td',c[2]);
    if(k==='ev_t1'||k==='ev_close'){ if(r[k]===null) td.textContent='—'; else td.appendChild(el('span','chip '+(r[k]>1.1?'good':(r[k]<0.8?'bad':'')),fmt(r[k],2))); }
    else if(k==='hit') td.textContent = r.hit===null?'—':(r.hit?'◯':'');
    else if(k==='odds_t1'||k==='fair_t1'||k==='odds_close') td.textContent=fmt(r[k],1);
    else td.textContent = r[k]; tr.appendChild(td); }); t.appendChild(tr); });
}
var days=[]; rows.forEach(function(r){ if(days.indexOf(r.date)<0) days.push(r.date); }); days.sort().reverse();
function opts(sel, list, val){ sel.innerHTML=''; list.forEach(function(o){ var op=el('option','',o[1]); op.value=o[0]; sel.appendChild(op); }); if(val!==undefined) sel.value=val; }
var gd=document.getElementById('gd'), gk=document.getElementById('gk'), rd=document.getElementById('rd'), rp=document.getElementById('rp');
opts(gd, days.map(function(d){return [d,d];}), days[0]); opts(rd, days.map(function(d){return [d,d];}), days[0]);
function drawGaps(){ var rs=rows.filter(function(r){ return r.date===gd.value && (!gk.value||r.kind===gk.value) && r.ev_t1!==null; }).sort(function(a,b){return b.ev_t1-a.ev_t1;}).slice(0,40); table('gt', rs, COLS); }
function fillRaces(){ var seen={}, list=[]; rows.forEach(function(r){ if(r.date===rd.value && !seen[r.race]){ seen[r.race]=1; list.push([r.race, r.deadline+' '+r.race, r.deadline]); } });
  list.sort(function(a,b){ return a[2]<b[2]?-1:1; }); opts(rp, list, list.length?list[list.length-1][0]:undefined); }
function drawRace(){ table('rt', rows.filter(function(r){ return r.date===rd.value && r.race===rp.value; }), COLS.slice(2)); }
gd.onchange=drawGaps; gk.onchange=drawGaps; rd.onchange=function(){ fillRaces(); drawRace(); }; rp.onchange=drawRace;
drawGaps(); fillRaces(); drawRace();
})();
</script>
"""

def main():
    rows = json.load(open(sys.argv[1], encoding="utf-8"))
    import datetime
    upd = (datetime.datetime.utcnow() + datetime.timedelta(hours=9)).strftime("%Y-%m-%d %H:%M")
    html = TEMPLATE.replace("__DATA__", json.dumps(rows, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")).replace("__UPDATED__", upd + "（日本時間）")
    open(sys.argv[2], "w", encoding="utf-8").write(html)
    print(sys.argv[2], len(html) // 1024, "KB")

if __name__ == "__main__":
    main()
