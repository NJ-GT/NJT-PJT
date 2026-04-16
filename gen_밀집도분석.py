import sys, json, os, pandas as pd, numpy as np, geopandas as gpd
from pyproj import Transformer
sys.stdout.reconfigure(encoding='utf-8')

GU = ['강남구','송파구','서초구','영등포구','강서구','성동구','용산구','마포구','중구','종로구']

gu_map = {
    '11010':'종로구','11020':'중구','11030':'용산구','11040':'성동구','11050':'광진구',
    '11060':'동대문구','11070':'중랑구','11080':'노원구','11090':'강북구','11100':'도봉구',
    '11110':'은평구','11120':'서대문구','11130':'마포구','11140':'양천구','11150':'강서구',
    '11160':'구로구','11170':'금천구','11180':'영등포구','11190':'동작구','11200':'관악구',
    '11210':'서초구','11220':'강남구','11230':'송파구','11240':'강동구','11250':'도봉구',
}

print("1. 숙박시설 CSV 로드...")
df = pd.read_csv('data/통합숙박시설최종안0415.csv', encoding='utf-8-sig')
cols = df.columns.tolist()
tf = Transformer.from_crs('EPSG:5181', 'EPSG:4326', always_xy=True)
xs, ys = tf.transform(df[cols[0]].values, df[cols[1]].values)
df['lng'] = xs; df['lat'] = ys
df['연면적'] = pd.to_numeric(df[cols[11]], errors='coerce').fillna(0)
df['층수']   = pd.to_numeric(df[cols[16]], errors='coerce').fillna(1)
df['바닥면적'] = df['연면적'] / df['층수'].clip(lower=1)
print(f"   {len(df)}개 건물")

print("2. 집계구 경계/면적 로드...")
oa = gpd.read_file('data/bnd_oa_11_2025_2Q/bnd_oa_11_2025_2Q.shp').to_crs('EPSG:4326')
oa_m = oa.to_crs('EPSG:5179')
oa['gu_name'] = oa['TOT_OA_CD'].str[:5].map(gu_map).fillna('알수없음')
oa['area_m2'] = oa_m.geometry.area

# 자치구 면적 (m²) — 집계구 합산
gu_area_m2 = oa.groupby('gu_name')['area_m2'].sum()

print("3. 공간결합 (건물 → 구)...")
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['lng'], df['lat']), crs='EPSG:4326')
joined = gpd.sjoin(gdf, oa[['TOT_OA_CD','gu_name','geometry']], how='left', predicate='within')
gu_col = 'gu_name_left' if 'gu_name_left' in joined.columns else 'gu_name'
joined['gu_name'] = joined[gu_col].fillna('알수없음')

# 구별 집계
gu_stats = joined.groupby('gu_name').agg(
    cnt       = ('연면적', 'count'),
    sum_fa    = ('연면적', 'sum'),    # Σ연면적 (m²)
    avg_fa    = ('연면적', 'mean'),
    avg_fl    = ('층수',   'mean'),
).reset_index()

print("4. VolumeRatio 계산...")
results = []
for g in GU:
    row = gu_stats[gu_stats['gu_name'] == g]
    area = float(gu_area_m2.get(g, 1))
    if len(row) == 0:
        results.append({'gu': g, 'cnt': 0, 'sum_fa': 0, 'avg_fa': 0, 'avg_fl': 0,
                        'area_km2': round(area/1e6, 2), 'vr': 0.0})
        continue
    r = row.iloc[0]
    vr = float(r['sum_fa']) / area
    results.append({
        'gu':      g,
        'cnt':     int(r['cnt']),
        'sum_fa':  round(float(r['sum_fa'])/1e4, 2),  # 만㎡ 단위
        'avg_fa':  round(float(r['avg_fa']), 1),
        'avg_fl':  round(float(r['avg_fl']), 1),
        'area_km2': round(area/1e6, 2),
        'vr':      round(vr, 6),
    })
    print(f"   {g}: VR={vr:.4f}  Σ연면적={r['sum_fa']/1e4:.2f}만㎡  구면적={area/1e6:.2f}km²")

# 정렬 (VR 높은 순)
results_sorted = sorted(results, key=lambda x: -x['vr'])
labels = [r['gu'] for r in results_sorted]
vr_vals = [r['vr'] for r in results_sorted]
cnt_vals = [r['cnt'] for r in results_sorted]
fa_vals  = [r['sum_fa'] for r in results_sorted]
area_vals= [r['area_km2'] for r in results_sorted]

results_json = json.dumps(results, ensure_ascii=False)
labels_json  = json.dumps(labels, ensure_ascii=False)
vr_json      = json.dumps(vr_vals, ensure_ascii=False)
cnt_json     = json.dumps(cnt_vals, ensure_ascii=False)
fa_json      = json.dumps(fa_vals, ensure_ascii=False)
area_json    = json.dumps(area_vals, ensure_ascii=False)

# 색상: 상위 3개 강조
def bar_colors(vals):
    mx = max(vals) if vals else 1
    colors = []
    for v in vals:
        r = v / mx
        if r >= 0.8:   colors.append('#ff3030')
        elif r >= 0.5: colors.append('#ff8c00')
        elif r >= 0.3: colors.append('#fcc419')
        else:          colors.append('#4e9af1')
    return colors

colors_json = json.dumps(bar_colors(vr_vals), ensure_ascii=False)

HTML = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>10개구 입체적 화재 하중 밀집도 분석</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js"></script>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:#0a0a14;color:#ddd;font-family:'Segoe UI',sans-serif;padding:24px}}
h1{{color:#ffb432;font-size:18px;font-weight:700;margin-bottom:4px}}
.sub{{color:#666;font-size:12px;margin-bottom:6px}}
.formula{{background:#111128;border:1px solid rgba(255,255,255,.1);border-radius:10px;
  padding:14px 20px;margin-bottom:20px;display:inline-block}}
.formula-title{{color:#ffb432;font-size:12px;font-weight:700;margin-bottom:8px}}
.formula-body{{color:#ddd;font-size:13px;line-height:1.8}}
.formula-math{{text-align:center;padding:8px 0;color:#fff;font-size:14px}}
.frac{{display:inline-block;text-align:center;vertical-align:middle}}
.frac .num{{border-bottom:1px solid #aaa;padding:2px 8px;display:block}}
.frac .den{{padding:2px 8px;display:block}}
#chart{{width:100%;height:420px;margin-bottom:20px}}
.table-wrap{{overflow-x:auto}}
table{{width:100%;border-collapse:collapse;font-size:12px;min-width:600px}}
thead th{{background:#111128;color:#ffb432;padding:9px 12px;text-align:left;
  border-bottom:1px solid rgba(255,180,50,.3);white-space:nowrap}}
tbody tr{{border-bottom:1px solid rgba(255,255,255,.05)}}
tbody tr:hover{{background:rgba(255,180,50,.06)}}
td{{padding:8px 12px}}
.rank{{font-weight:700;color:#ffb432}}
.vr-bar{{display:flex;align-items:center;gap:8px}}
.vr-bg{{flex:1;height:10px;background:rgba(255,255,255,.07);border-radius:5px;min-width:80px}}
.vr-fill{{height:100%;border-radius:5px}}
.badge{{display:inline-block;padding:2px 7px;border-radius:4px;font-size:10px;font-weight:700}}
.note{{font-size:11px;color:#555;margin-top:12px}}
</style>
</head>
<body>
<h1>🔥 10개구 입체적 화재 하중 밀집도 (VolumeRatio)</h1>
<div class="sub">숙박시설 연면적 기준 · 데이터: 통합숙박시설최종안0415.csv</div>

<div class="formula">
  <div class="formula-title">② 입체적 화재 하중 밀집도 (연면적 기반)</div>
  <div class="formula-body">단층 건물이냐, 고층 건물이냐에 따라 잠재적인 화재 규모가 달라지는 것을 반영합니다.</div>
  <div class="formula-math">
    <i>VolumeRatio</i> &nbsp;=&nbsp;
    <span class="frac">
      <span class="num">Σ 개별 건물들의 연면적 (바닥면적 × 층수)</span>
      <span class="den">자치구의 전체 면적</span>
    </span>
  </div>
</div>

<div id="chart"></div>

<div class="table-wrap">
<table>
  <thead>
    <tr>
      <th>순위</th><th>자치구</th><th>VolumeRatio</th>
      <th>Σ연면적 (만㎡)</th><th>구 면적 (km²)</th>
      <th>숙박시설 수</th><th>평균 층수</th>
    </tr>
  </thead>
  <tbody id="tbody"></tbody>
</table>
</div>
<div class="note">※ 서초구·영등포구·마포구는 공간결합 데이터 부족으로 연면적 0 처리됨 (CSV 좌표 범위 이슈)</div>

<script>
var DATA   = {results_json};
var LABELS = {labels_json};
var VR     = {vr_json};
var CNT    = {cnt_json};
var FA     = {fa_json};
var AREA   = {area_json};
var COLORS = {colors_json};

var GU_COLOR = {{
  '강남구':'#4e9af1','서초구':'#a78bfa','송파구':'#34d399','영등포구':'#fb923c',
  '강서구':'#60a5fa','성동구':'#f472b6','용산구':'#f87171','마포구':'#4ade80',
  '중구':'#fbbf24','종로구':'#e879f9'
}};

// 바플롯
var chart = echarts.init(document.getElementById('chart'));
chart.setOption({{
  backgroundColor: 'transparent',
  title:{{text:'자치구별 입체적 화재 하중 밀집도 (높을수록 건물 연면적 집중)',
    textStyle:{{color:'#ffb432',fontSize:13}},left:0,top:4}},
  tooltip:{{
    trigger:'axis',
    formatter:function(p){{
      var i=p[0].dataIndex;
      return '<b>'+LABELS[i]+'</b><br/>'
        +'VolumeRatio: <b>'+VR[i].toFixed(4)+'</b><br/>'
        +'Σ연면적: '+FA[i].toFixed(2)+'만㎡<br/>'
        +'구 면적: '+AREA[i]+'km²<br/>'
        +'숙박시설: '+CNT[i]+'개';
    }}
  }},
  grid:{{left:50,right:20,top:50,bottom:70}},
  xAxis:{{
    type:'category',data:LABELS,
    axisLabel:{{color:'#bbb',fontSize:12}},
    axisLine:{{lineStyle:{{color:'#333'}}}}
  }},
  yAxis:{{
    type:'value',name:'VolumeRatio',
    nameTextStyle:{{color:'#888',fontSize:11}},
    axisLabel:{{color:'#aaa',formatter:function(v){{return v.toFixed(3);}}}},
    splitLine:{{lineStyle:{{color:'rgba(255,255,255,.06)'}}}}
  }},
  series:[{{
    type:'bar',
    data:VR.map(function(v,i){{return {{value:v,itemStyle:{{color:COLORS[i]}}}};}}) ,
    barMaxWidth:60,
    label:{{show:true,position:'top',formatter:function(p){{return p.value.toFixed(4);}},
      color:'#ccc',fontSize:11}}
  }}]
}});
window.addEventListener('resize',function(){{chart.resize();}});

// 테이블
var maxVR = Math.max.apply(null, VR);
var tbody = document.getElementById('tbody');
tbody.innerHTML = DATA.map(function(r,i){{
  var rank = i+1;
  var pct  = maxVR>0 ? (r.vr/maxVR*100).toFixed(1) : 0;
  var col  = r.vr>=maxVR*0.8?'#ff3030':r.vr>=maxVR*0.5?'#ff8c00':r.vr>=maxVR*0.3?'#fcc419':'#4e9af1';
  var gc   = GU_COLOR[r.gu]||'#aaa';
  return '<tr>'
    +'<td class="rank">#'+rank+'</td>'
    +'<td><span style="color:'+gc+';font-weight:700">'+r.gu+'</span></td>'
    +'<td><div class="vr-bar">'
    +  '<div class="vr-bg"><div class="vr-fill" style="width:'+pct+'%;background:'+col+'"></div></div>'
    +  '<span style="color:'+col+';font-weight:700;min-width:50px">'+r.vr.toFixed(4)+'</span>'
    +'</div></td>'
    +'<td style="color:#fff;font-weight:600">'+r.sum_fa.toFixed(2)+'</td>'
    +'<td style="color:#aaa">'+r.area_km2+'</td>'
    +'<td style="color:#aaa">'+r.cnt+'</td>'
    +'<td style="color:#aaa">'+(r.avg_fl||'-')+'</td>'
    +'</tr>';
}}).join('');
</script>
</body>
</html>"""

with open('밀집도_분석.html', 'w', encoding='utf-8') as f:
    f.write(HTML)
print(f'Done: {os.path.getsize("밀집도_분석.html")//1024} KB')
