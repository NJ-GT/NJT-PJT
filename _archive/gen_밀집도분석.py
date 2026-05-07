"""
[파일 설명]
10개 자치구의 입체적 화재 하중 밀집도(VolumeRatio)를 계산하고 바차트로 시각화하는 HTML을 생성하는 스크립트.

VolumeRatio(입체적 화재 하중 밀집도)란?
  숙박시설 연면적(바닥면적×층수)의 합을 자치구 전체 면적으로 나눈 값.
  단층/고층 여부를 반영하여 실질적인 화재 규모 잠재력을 나타낸다.
  VR = Σ(개별 건물 연면적) / 자치구 면적

주요 역할:
  1. 숙박시설 CSV의 좌표를 위도/경도로 변환한다.
  2. 집계구 shapefile로부터 자치구 면적을 계산한다.
  3. 공간결합으로 각 숙박시설을 해당 자치구에 배정한다.
  4. 구별 VolumeRatio를 계산하고 ECharts 바차트 HTML로 출력한다.

입력: data/통합숙박시설최종안0415.csv  (숙박시설 4,246개)
      data/bnd_oa_11_2025_2Q/bnd_oa_11_2025_2Q.shp (집계구 경계)
출력: 밀집도_분석.html                  (VolumeRatio 바차트 + 순위 테이블)
"""

import sys, json, os, pandas as pd, numpy as np, geopandas as gpd
from pyproj import Transformer
sys.stdout.reconfigure(encoding='utf-8')  # 한글 출력 설정

# 분석 대상 10개 자치구 목록 (표시 순서 고정)
GU = ['강남구','송파구','서초구','영등포구','강서구','성동구','용산구','마포구','중구','종로구']

# 집계구 코드 앞 5자리 → 자치구 이름 변환 테이블
gu_map = {
    '11010':'종로구','11020':'중구','11030':'용산구','11040':'성동구','11050':'광진구',
    '11060':'동대문구','11070':'중랑구','11080':'노원구','11090':'강북구','11100':'도봉구',
    '11110':'은평구','11120':'서대문구','11130':'마포구','11140':'양천구','11150':'강서구',
    '11160':'구로구','11170':'금천구','11180':'영등포구','11190':'동작구','11200':'관악구',
    '11210':'서초구','11220':'강남구','11230':'송파구','11240':'강동구','11250':'도봉구',
}

# ─── 1. 숙박시설 CSV 로드 및 좌표 변환 ──────────────────────────
print("1. 숙박시설 CSV 로드...")
df = pd.read_csv('data/통합숙박시설최종안0415.csv', encoding='utf-8-sig')
cols = df.columns.tolist()  # 열 이름 목록

# 한국 좌표계(EPSG:5181) → WGS84 위도/경도(EPSG:4326) 변환
tf = Transformer.from_crs('EPSG:5181', 'EPSG:4326', always_xy=True)
xs, ys = tf.transform(df[cols[0]].values, df[cols[1]].values)
df['lng'] = xs  # 경도
df['lat'] = ys  # 위도

df['연면적'] = pd.to_numeric(df[cols[11]], errors='coerce').fillna(0)  # 결측값은 0으로 처리
df['층수']   = pd.to_numeric(df[cols[16]], errors='coerce').fillna(1)  # 층수 결측 시 1층으로 처리
# 바닥면적 = 연면적 ÷ 층수 (층수가 0이 되지 않도록 clip(lower=1) 적용)
df['바닥면적'] = df['연면적'] / df['층수'].clip(lower=1)
print(f"   {len(df)}개 건물")

# ─── 2. 집계구 경계 및 자치구 면적 로드 ─────────────────────────
print("2. 집계구 경계/면적 로드...")
oa = gpd.read_file('data/bnd_oa_11_2025_2Q/bnd_oa_11_2025_2Q.shp').to_crs('EPSG:4326')
oa_m = oa.to_crs('EPSG:5179')                         # 미터 단위 좌표계로 재투영
oa['gu_name'] = oa['TOT_OA_CD'].str[:5].map(gu_map).fillna('알수없음')
oa['area_m2'] = oa_m.geometry.area                    # 집계구 면적(m²)

# 집계구 면적을 합산하여 자치구 전체 면적 계산
gu_area_m2 = oa.groupby('gu_name')['area_m2'].sum()

# ─── 3. 숙박시설 → 자치구 공간결합 ─────────────────────────────
print("3. 공간결합 (건물 → 구)...")
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['lng'], df['lat']), crs='EPSG:4326')
joined = gpd.sjoin(gdf, oa[['TOT_OA_CD','gu_name','geometry']], how='left', predicate='within')

# 공간결합 후 컬럼명 충돌 처리 (gu_name이 _left/_right 접미사로 분리될 수 있음)
gu_col = 'gu_name_left' if 'gu_name_left' in joined.columns else 'gu_name'
joined['gu_name'] = joined[gu_col].fillna('알수없음')

# 구별 집계 : 시설 수, 연면적 합/평균, 평균 층수
gu_stats = joined.groupby('gu_name').agg(
    cnt       = ('연면적', 'count'),   # 숙박시설 개수
    sum_fa    = ('연면적', 'sum'),     # 연면적 합계 (m²)
    avg_fa    = ('연면적', 'mean'),    # 평균 연면적
    avg_fl    = ('층수',   'mean'),    # 평균 층수
).reset_index()

# ─── 4. VolumeRatio 계산 ────────────────────────────────────────
print("4. VolumeRatio 계산...")
results = []
for g in GU:
    row = gu_stats[gu_stats['gu_name'] == g]
    area = float(gu_area_m2.get(g, 1))  # 구 면적(m²), 없으면 1로 (0 나누기 방지)
    if len(row) == 0:
        # 공간결합 결과 없으면 0으로 처리
        results.append({'gu': g, 'cnt': 0, 'sum_fa': 0, 'avg_fa': 0, 'avg_fl': 0,
                        'area_km2': round(area/1e6, 2), 'vr': 0.0})
        continue
    r = row.iloc[0]
    vr = float(r['sum_fa']) / area  # VolumeRatio = 연면적 합 / 구 면적
    results.append({
        'gu':      g,
        'cnt':     int(r['cnt']),
        'sum_fa':  round(float(r['sum_fa'])/1e4, 2),  # m² → 만㎡ 단위로 변환
        'avg_fa':  round(float(r['avg_fa']), 1),
        'avg_fl':  round(float(r['avg_fl']), 1),
        'area_km2': round(area/1e6, 2),                # m² → km² 변환
        'vr':      round(vr, 6),
    })
    print(f"   {g}: VR={vr:.4f}  Σ연면적={r['sum_fa']/1e4:.2f}만㎡  구면적={area/1e6:.2f}km²")

# VR 높은 순으로 정렬하여 차트에 표시
results_sorted = sorted(results, key=lambda x: -x['vr'])
labels = [r['gu'] for r in results_sorted]
vr_vals = [r['vr'] for r in results_sorted]
cnt_vals = [r['cnt'] for r in results_sorted]
fa_vals  = [r['sum_fa'] for r in results_sorted]
area_vals= [r['area_km2'] for r in results_sorted]

# ─── 5. 데이터를 JavaScript 변수로 직렬화 ────────────────────────
results_json = json.dumps(results, ensure_ascii=False)   # 원래 순서 (테이블용)
labels_json  = json.dumps(labels, ensure_ascii=False)    # VR 정렬 순서 (차트용)
vr_json      = json.dumps(vr_vals, ensure_ascii=False)
cnt_json     = json.dumps(cnt_vals, ensure_ascii=False)
fa_json      = json.dumps(fa_vals, ensure_ascii=False)
area_json    = json.dumps(area_vals, ensure_ascii=False)

# ─── 6. 막대 색상 결정 (최대값 대비 비율에 따라 색상 등급화) ────────
def bar_colors(vals):
    """
    VolumeRatio 값의 상대적 크기에 따라 막대 색상을 결정한다.
    최대값의 80% 이상: 빨강 / 50% 이상: 주황 / 30% 이상: 노랑 / 그 외: 파랑
    """
    mx = max(vals) if vals else 1
    colors = []
    for v in vals:
        r = v / mx
        if r >= 0.8:   colors.append('#ff3030')   # 매우 높음 (빨강)
        elif r >= 0.5: colors.append('#ff8c00')   # 높음 (주황)
        elif r >= 0.3: colors.append('#fcc419')   # 보통 (노랑)
        else:          colors.append('#4e9af1')   # 낮음 (파랑)
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

# ─── 7. HTML 파일 저장 ───────────────────────────────────────────
with open('밀집도_분석.html', 'w', encoding='utf-8') as f:
    f.write(HTML)
print(f'Done: {os.path.getsize("밀집도_분석.html")//1024} KB')
