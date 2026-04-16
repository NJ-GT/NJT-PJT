import sys, json, numpy as np, os
sys.stdout.reconfigure(encoding='utf-8')

GU = ['강남구','송파구','서초구','영등포구','강서구','성동구','용산구','마포구','중구','종로구']

with open('data/oa_density.json', encoding='utf-8') as f:
    d = json.load(f)

rows = []
for f in d['features']:
    p = f['properties']
    if p.get('gu_name', '') in GU and p.get('count', 0) > 0:
        rows.append({
            'gu':   p['gu_name'],
            'no':   p.get('oa_no', p['id'][8:]),
            'cnt':  p['count'],
            'fire': round(p.get('fire_score') or 0, 1),
            'age':  round(p.get('avg_age') or 0, 1),
            'fl':   round(p.get('avg_floors') or 0, 1),
            'gpye': round(p.get('avg_geonpye') or 0, 1),
            'yong': round(p.get('avg_yongjuk') or 0, 1),
        })

rows.sort(key=lambda x: (GU.index(x['gu']) if x['gu'] in GU else 99, -x['fire']))

def bp_stats(scores):
    arr = np.array(scores, dtype=float)
    q1, q2, q3 = np.percentile(arr, [25, 50, 75])
    iqr = q3 - q1
    lo = max(float(arr.min()), q1 - 1.5 * iqr)
    hi = min(float(arr.max()), q3 + 1.5 * iqr)
    outliers = arr[(arr < lo) | (arr > hi)].tolist()
    return [round(lo,1), round(q1,1), round(q2,1), round(q3,1), round(hi,1)], [round(o,1) for o in outliers]

bp_labels, bp_data, bp_outlier_pts = [], [], []
for gu in GU:
    scores = [r['fire'] for r in rows if r['gu'] == gu]
    if scores:
        bp_labels.append(gu)
        stats, outs = bp_stats(scores)
        bp_data.append(stats)
        idx = len(bp_labels) - 1
        for o in outs:
            bp_outlier_pts.append([idx, o])

rows_json  = json.dumps(rows, ensure_ascii=False)
labels_json = json.dumps(bp_labels, ensure_ascii=False)
bpdata_json = json.dumps(bp_data, ensure_ascii=False)
bpout_json  = json.dumps(bp_outlier_pts, ensure_ascii=False)

no_data_gu = [g for g in GU if g not in bp_labels]
no_data_note = ('<div class="note">※ 숙박시설 공간결합 데이터 부족으로 표시 제외: ' + ', '.join(no_data_gu) + '</div>') if no_data_gu else ''
gu_options = ''.join(f'<option value="{g}">{g}</option>' for g in bp_labels)

HTML = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>10개구 집계구별 화재위험도 분석</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js"></script>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:#0a0a14;color:#ddd;font-family:'Segoe UI',sans-serif;min-height:100vh}}
h1{{color:#ffb432;font-size:18px;padding:20px 24px 0;font-weight:700}}
.sub{{color:#666;font-size:12px;padding:4px 24px 16px}}
#chart{{width:100%;height:360px;padding:0 16px}}
.section{{padding:0 24px 24px}}
.section-title{{color:#ffb432;font-size:14px;font-weight:700;margin-bottom:12px;padding-top:8px;
  border-top:1px solid rgba(255,255,255,.08)}}
.controls{{display:flex;gap:10px;margin-bottom:12px;flex-wrap:wrap;align-items:center}}
.ctrl-label{{font-size:12px;color:#888}}
select{{background:#1a1a2e;color:#ccc;border:1px solid rgba(255,255,255,.2);border-radius:6px;
  padding:5px 10px;font-size:12px;cursor:pointer}}
input[type=text]{{background:#1a1a2e;color:#ccc;border:1px solid rgba(255,255,255,.2);border-radius:6px;
  padding:5px 10px;font-size:12px;width:160px}}
table{{width:100%;border-collapse:collapse;font-size:12px}}
thead th{{background:#111128;color:#ffb432;padding:8px 10px;text-align:left;cursor:pointer;
  border-bottom:1px solid rgba(255,180,50,.3);white-space:nowrap;user-select:none}}
thead th:hover{{color:#fff}}
thead th.sort-asc::after{{content:' ▲'}}
thead th.sort-desc::after{{content:' ▼'}}
tbody tr{{border-bottom:1px solid rgba(255,255,255,.05)}}
tbody tr:hover{{background:rgba(255,180,50,.06)}}
td{{padding:7px 10px}}
.tag-gu{{display:inline-block;padding:2px 7px;border-radius:4px;font-size:11px;font-weight:600}}
.fire-bar{{display:flex;align-items:center;gap:6px}}
.fire-bg{{flex:1;height:8px;background:rgba(255,255,255,.08);border-radius:4px;min-width:60px}}
.fire-fill{{height:100%;border-radius:4px}}
.lv-low{{color:#50c878}}.lv-mid{{color:#fcc419}}.lv-hi{{color:#ff7a00}}.lv-vhi{{color:#e03030}}.lv-ext{{color:#ff2020}}
.note{{font-size:11px;color:#555;margin-top:10px}}
.stat-row{{display:flex;gap:24px;margin-bottom:16px;flex-wrap:wrap}}
.stat-box{{background:#111128;border:1px solid rgba(255,255,255,.1);border-radius:8px;
  padding:10px 16px;min-width:130px}}
.stat-box .sv{{font-size:20px;font-weight:700;color:#fff}}
.stat-box .sk{{font-size:11px;color:#666;margin-top:2px}}
</style>
</head>
<body>
<h1>🔥 10개구 집계구별 화재위험도 분석</h1>
<div class="sub">숙박시설이 있는 집계구 기준 · 화재위험도 = 노후도(30%) + 건폐율(25%) + 용적률(25%) + 층수(20%)</div>

<div id="chart"></div>

<div class="section">
  <div class="section-title">📊 구별 요약 통계</div>
  <div class="stat-row" id="stat-row"></div>

  <div class="section-title">📋 집계구별 상세 테이블</div>
  <div class="controls">
    <span class="ctrl-label">구 필터:</span>
    <select id="gu-sel" onchange="renderTable()">
      <option value="">전체</option>
      {gu_options}
    </select>
    <span class="ctrl-label">위험도:</span>
    <select id="lv-sel" onchange="renderTable()">
      <option value="">전체</option>
      <option value="80">극위험 (80+)</option>
      <option value="65">매우높음 (65+)</option>
      <option value="45">높음 (45+)</option>
      <option value="25">보통 이상 (25+)</option>
    </select>
    <input type="text" id="srch" placeholder="집계구 번호 검색" oninput="renderTable()">
  </div>
  <table id="tbl">
    <thead>
      <tr>
        <th onclick="sortBy('gu')">자치구</th>
        <th onclick="sortBy('no')">집계구 번호</th>
        <th onclick="sortBy('cnt')">숙박시설 수</th>
        <th onclick="sortBy('fire')">화재위험도</th>
        <th onclick="sortBy('age')">평균 건축연령</th>
        <th onclick="sortBy('fl')">평균 층수</th>
        <th onclick="sortBy('gpye')">건폐율(%)</th>
        <th onclick="sortBy('yong')">용적률(%)</th>
      </tr>
    </thead>
    <tbody id="tbody"></tbody>
  </table>
  <div class="note" id="count-note"></div>
  {no_data_note}
</div>

<script>
var DATA = {rows_json};
var GU_COLOR = {{
  '강남구':'#4e9af1','서초구':'#a78bfa','송파구':'#34d399','영등포구':'#fb923c',
  '강동구':'#f472b6','강서구':'#60a5fa','중구':'#fbbf24','종로구':'#e879f9',
  '마포구':'#4ade80','용산구':'#f87171'
}};

function fireColor(v){{
  if(v>=80) return '#ff2020';
  if(v>=65) return '#e03030';
  if(v>=45) return '#ff7a00';
  if(v>=25) return '#fcc419';
  return '#50c878';
}}
function fireLevel(v){{
  if(v>=80) return ['극위험','lv-ext'];
  if(v>=65) return ['매우높음','lv-vhi'];
  if(v>=45) return ['높음','lv-hi'];
  if(v>=25) return ['보통','lv-mid'];
  return ['낮음','lv-low'];
}}

// 박스플롯
var chart = echarts.init(document.getElementById('chart'));
var bpLabels = {labels_json};
var bpData   = {bpdata_json};
var bpOut    = {bpout_json};

chart.setOption({{
  backgroundColor:'transparent',
  title:{{text:'구별 화재위험도 분포 (Boxplot)',textStyle:{{color:'#ffb432',fontSize:13}},left:16,top:8}},
  tooltip:{{trigger:'item',formatter:function(p){{
    if(p.seriesType==='boxplot'){{
      var d=p.data;
      return p.name+'<br/>최대:'+d[4]+' / Q3:'+d[3]+'<br/>중앙:'+d[2]+'<br/>Q1:'+d[1]+' / 최소:'+d[0];
    }}
    return p.seriesName+': '+p.data[1]+'점 (이상치)';
  }}}},
  grid:{{left:40,right:20,top:50,bottom:60}},
  xAxis:{{type:'category',data:bpLabels,axisLabel:{{color:'#aaa',rotate:15}},axisLine:{{lineStyle:{{color:'#333'}}}}}},
  yAxis:{{type:'value',name:'화재위험도',nameTextStyle:{{color:'#888'}},
    axisLabel:{{color:'#aaa'}},splitLine:{{lineStyle:{{color:'rgba(255,255,255,.06)'}}}},
    min:0,max:100}},
  series:[
    {{type:'boxplot',data:bpData,
      itemStyle:{{color:'rgba(255,180,50,.3)',borderColor:'#ffb432',borderWidth:1.5}},
      boxWidth:['30%','50%']}},
    {{type:'scatter',name:'이상치',data:bpOut,symbolSize:5,
      itemStyle:{{color:'#ff6060',opacity:0.7}}}}
  ]
}});
window.addEventListener('resize',function(){{chart.resize();}});

// 통계 요약
var statRow = document.getElementById('stat-row');
bpLabels.forEach(function(gu,i){{
  var scores = DATA.filter(function(r){{return r.gu===gu;}}).map(function(r){{return r.fire;}});
  var avg = (scores.reduce(function(a,b){{return a+b;}},0)/scores.length).toFixed(1);
  var max = Math.max.apply(null,scores).toFixed(1);
  var box = document.createElement('div');
  box.className = 'stat-box';
  var col = GU_COLOR[gu]||'#aaa';
  box.innerHTML = '<div style="font-size:11px;color:'+col+';font-weight:700;margin-bottom:4px">'+gu+'</div>'
    +'<div class="sv">'+avg+'<span style="font-size:13px;color:#888">점</span></div>'
    +'<div class="sk">평균 위험도 (최대 '+max+'점)</div>'
    +'<div class="sk">집계구 '+scores.length+'개</div>';
  statRow.appendChild(box);
}});

// 테이블
var sortKey = 'fire', sortDir = -1;
function sortBy(k){{
  if(sortKey===k) sortDir*=-1; else {{sortKey=k;sortDir=-1;}}
  document.querySelectorAll('thead th').forEach(function(th){{
    th.classList.remove('sort-asc','sort-desc');
  }});
  var ths = document.querySelectorAll('thead th');
  var keys=['gu','no','cnt','fire','age','fl','gpye','yong'];
  var idx=keys.indexOf(k);
  if(idx>=0){{
    ths[idx].classList.add(sortDir>0?'sort-asc':'sort-desc');
  }}
  renderTable();
}}

function renderTable(){{
  var guF  = document.getElementById('gu-sel').value;
  var lvF  = parseInt(document.getElementById('lv-sel').value)||0;
  var srch = document.getElementById('srch').value.trim();
  var filtered = DATA.filter(function(r){{
    if(guF && r.gu!==guF) return false;
    if(lvF && r.fire<lvF) return false;
    if(srch && r.no.indexOf(srch)<0) return false;
    return true;
  }});
  filtered.sort(function(a,b){{
    var av=a[sortKey], bv=b[sortKey];
    if(typeof av==='string') return sortDir*(av<bv?-1:av>bv?1:0);
    return sortDir*(av-bv);
  }});
  var tbody=document.getElementById('tbody');
  var col=GU_COLOR;
  tbody.innerHTML=filtered.map(function(r){{
    var lv=fireLevel(r.fire);
    var fc=fireColor(r.fire);
    var pct=Math.min(r.fire,100);
    return '<tr>'
      +'<td><span class="tag-gu" style="background:'+col[r.gu]+'22;color:'+(col[r.gu]||'#aaa')+'">'+r.gu+'</span></td>'
      +'<td style="color:#999;font-size:11px">'+r.no+'</td>'
      +'<td style="text-align:center;color:#fff;font-weight:600">'+r.cnt+'</td>'
      +'<td><div class="fire-bar"><div class="fire-bg"><div class="fire-fill" style="width:'+pct+'%;background:'+fc+'"></div></div>'
      +'<span class="'+lv[1]+'" style="font-weight:700;min-width:36px">'+r.fire+'</span>'
      +'<span class="'+lv[1]+'" style="font-size:10px">('+lv[0]+')</span></div></td>'
      +'<td style="color:'+(r.age>=50?'#ff4444':r.age>=30?'#ff9900':'#aaa')+'">'+r.age+'년</td>'
      +'<td style="color:#aaa">'+r.fl+'층</td>'
      +'<td style="color:#aaa">'+r.gpye+'%</td>'
      +'<td style="color:#aaa">'+r.yong+'%</td>'
      +'</tr>';
  }}).join('');
  document.getElementById('count-note').textContent='표시: '+filtered.length+'개 집계구';
}}

sortBy('fire');
</script>
</body>
</html>"""

with open('위험도_분석.html', 'w', encoding='utf-8') as f:
    f.write(HTML)
print(f'Done: {os.path.getsize("위험도_분석.html")//1024} KB, 집계구 {len(rows)}개, 구 {len(bp_labels)}개')
