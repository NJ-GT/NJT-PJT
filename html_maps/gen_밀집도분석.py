# -*- coding: utf-8 -*-
"""
입체적 화재 하중 밀집도(VolumeRatio) 분석 + ECharts HTML 생성 스크립트.

핵심 지표 — VolumeRatio:
    VR = Σ(개별 건물 연면적[m²]) / 자치구 면적[m²]
    → 단층/고층 여부에 관계없이 자치구가 보유한 "숙박 부피"를
      동일 면적 단위로 비교할 수 있게 해주는 밀집도 지표.

처리 흐름:
    1) 숙박시설 CSV 로드 + EPSG:5181 -> WGS84 좌표 변환
    2) 집계구 shapefile 로드 + 자치구별 면적 합산
    3) 시설 포인트와 집계구 폴리곤을 sjoin 으로 결합
    4) 자치구별 합계/평균 + VolumeRatio 계산
    5) ECharts 막대 차트 + 순위 테이블이 들어간 HTML 출력

산출물:
    NJT-PJT/밀집도_분석.html
"""

import sys
import json
import os
import pandas as pd
import geopandas as gpd
# 좌표계 변환기 (EPSG 변환)
from pyproj import Transformer

# Windows 콘솔에서 한글 깨짐 방지
sys.stdout.reconfigure(encoding="utf-8")

# 분석 대상 10개 자치구 (표시 순서)
GU = [
    "강남구",
    "송파구",
    "서초구",
    "영등포구",
    "강서구",
    "성동구",
    "용산구",
    "마포구",
    "중구",
    "종로구",
]

# 집계구 코드 prefix(앞 5자리) -> 자치구명 매핑
# 11220 -> 강남구 와 같이 조회
gu_map = {
    "11010": "종로구",
    "11020": "중구",
    "11030": "용산구",
    "11040": "성동구",
    "11050": "광진구",
    "11060": "동대문구",
    "11070": "중랑구",
    "11080": "노원구",
    "11090": "강북구",
    "11100": "도봉구",
    "11110": "은평구",
    "11120": "서대문구",
    "11130": "마포구",
    "11140": "양천구",
    "11150": "강서구",
    "11160": "구로구",
    "11170": "금천구",
    "11180": "영등포구",
    "11190": "동작구",
    "11200": "관악구",
    "11210": "서초구",
    "11220": "강남구",
    "11230": "송파구",
    "11240": "강동구",
    "11250": "도봉구",
}

# ─── 1. 숙박시설 CSV 로드 및 좌표 변환 ──────────────────────────
print("1. 숙박시설 CSV 로드...")
df = pd.read_csv("data/통합숙박시설최종안0415.csv", encoding="utf-8-sig")
# 컬럼 위치를 인덱스로 접근하기 위해 리스트화 (스키마가 안정적이라는 가정)
cols = df.columns.tolist()

# 한국 중부원점 좌표(EPSG:5181) -> WGS84(EPSG:4326)
# always_xy=True : (x, y) = (경도, 위도) 순으로 일관 처리
tf = Transformer.from_crs("EPSG:5181", "EPSG:4326", always_xy=True)
xs, ys = tf.transform(df[cols[0]].values, df[cols[1]].values)
df["lng"] = xs
df["lat"] = ys

# 연면적/층수 수치 변환 — 결측은 안전 기본값으로 (연면적 0, 층수 1)
df["연면적"] = pd.to_numeric(df[cols[11]], errors="coerce").fillna(0)
df["층수"] = pd.to_numeric(df[cols[16]], errors="coerce").fillna(1)
# 바닥면적 = 연면적 / 층수 (층수 0으로 인한 ZeroDivisionError 방지를 위해 clip)
df["바닥면적"] = df["연면적"] / df["층수"].clip(lower=1)
print(f"   {len(df)}개 건물")

# ─── 2. 집계구 경계 및 자치구 면적 로드 ─────────────────────────
print("2. 집계구 경계/면적 로드...")
oa = gpd.read_file("data/bnd_oa_11_2025_2Q/bnd_oa_11_2025_2Q.shp").to_crs("EPSG:4326")
# 면적 계산은 미터 단위 EPSG:5179에서 수행 (왜곡 최소)
oa_m = oa.to_crs("EPSG:5179")
# 집계구 코드 앞 5자리로 자치구명 부여
oa["gu_name"] = oa["TOT_OA_CD"].str[:5].map(gu_map).fillna("알수없음")
# 집계구 면적(m²)
oa["area_m2"] = oa_m.geometry.area

# 집계구 면적을 자치구 단위로 합산 — 자치구 전체 면적
gu_area_m2 = oa.groupby("gu_name")["area_m2"].sum()

# ─── 3. 숙박시설 → 자치구 공간결합 ─────────────────────────────
print("3. 공간결합 (건물 → 구)...")
# 시설 좌표를 GeoDataFrame으로 — Point geometry
gdf = gpd.GeoDataFrame(
    df, geometry=gpd.points_from_xy(df["lng"], df["lat"]), crs="EPSG:4326"
)
# 시설이 어떤 집계구 폴리곤 내부에 있는지 좌측 조인 (within 술어)
joined = gpd.sjoin(
    gdf, oa[["TOT_OA_CD", "gu_name", "geometry"]], how="left", predicate="within"
)

# sjoin 결과 컬럼명 충돌 방지 — gu_name이 _left/_right로 붙을 수 있음
gu_col = "gu_name_left" if "gu_name_left" in joined.columns else "gu_name"
joined["gu_name"] = joined[gu_col].fillna("알수없음")

# 자치구 단위 집계: 시설수, 연면적 합/평균, 평균 층수
gu_stats = (
    joined.groupby("gu_name")
    .agg(
        cnt=("연면적", "count"),
        sum_fa=("연면적", "sum"),
        avg_fa=("연면적", "mean"),
        avg_fl=("층수", "mean"),
    )
    .reset_index()
)

# ─── 4. VolumeRatio 계산 ────────────────────────────────────────
print("4. VolumeRatio 계산...")
results = []
for g in GU:
    row = gu_stats[gu_stats["gu_name"] == g]
    # 자치구 면적이 없는 경우(예외 케이스) 1로 두어 0 나눗셈 회피
    area = float(gu_area_m2.get(g, 1))
    if len(row) == 0:
        # 공간결합 결과가 비어 있으면 0으로 채워 표시
        results.append(
            {
                "gu": g,
                "cnt": 0,
                "sum_fa": 0,
                "avg_fa": 0,
                "avg_fl": 0,
                "area_km2": round(area / 1e6, 2),
                "vr": 0.0,
            }
        )
        continue
    r = row.iloc[0]
    # 핵심 지표 — 연면적 합 / 자치구 면적
    vr = float(r["sum_fa"]) / area
    results.append(
        {
            "gu": g,
            "cnt": int(r["cnt"]),
            # 보기 좋게 만㎡ 단위로 환산
            "sum_fa": round(float(r["sum_fa"]) / 1e4, 2),
            "avg_fa": round(float(r["avg_fa"]), 1),
            "avg_fl": round(float(r["avg_fl"]), 1),
            # km² 환산
            "area_km2": round(area / 1e6, 2),
            "vr": round(vr, 6),
        }
    )
    print(
        f"   {g}: VR={vr:.4f}  Σ연면적={r['sum_fa'] / 1e4:.2f}만㎡  구면적={area / 1e6:.2f}km²"
    )

# 차트는 VR 큰 순으로 정렬 표시
results_sorted = sorted(results, key=lambda x: -x["vr"])
labels = [r["gu"] for r in results_sorted]
vr_vals = [r["vr"] for r in results_sorted]
cnt_vals = [r["cnt"] for r in results_sorted]
fa_vals = [r["sum_fa"] for r in results_sorted]
area_vals = [r["area_km2"] for r in results_sorted]

# ─── 5. JS 변수 직렬화 ────────────────────────────────────────────
# DATA는 원래 GU 정렬 (테이블용), LABELS/VR는 VR 정렬 (차트용)
results_json = json.dumps(results, ensure_ascii=False)
labels_json = json.dumps(labels, ensure_ascii=False)
vr_json = json.dumps(vr_vals, ensure_ascii=False)
cnt_json = json.dumps(cnt_vals, ensure_ascii=False)
fa_json = json.dumps(fa_vals, ensure_ascii=False)
area_json = json.dumps(area_vals, ensure_ascii=False)


# ─── 6. 막대 색상 등급화 ──────────────────────────────────────
def bar_colors(vals):
    """최댓값 대비 비율로 4단계 색상 부여 (빨강/주황/노랑/파랑)."""
    mx = max(vals) if vals else 1
    colors = []
    for v in vals:
        r = v / mx
        if r >= 0.8:
            colors.append("#ff3030")  # 매우 높음
        elif r >= 0.5:
            colors.append("#ff8c00")  # 높음
        elif r >= 0.3:
            colors.append("#fcc419")  # 보통
        else:
            colors.append("#4e9af1")  # 낮음
    return colors


colors_json = json.dumps(bar_colors(vr_vals), ensure_ascii=False)

# ─── 7. HTML 빌드 (f-string으로 데이터 주입) ─────────────────────
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

// 테이블 — DATA(원래 GU 순서) 기준 렌더
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

# ─── 8. HTML 파일 저장 ───────────────────────────────────────────
with open("밀집도_분석.html", "w", encoding="utf-8") as f:
    f.write(HTML)
print(f"Done: {os.path.getsize('밀집도_분석.html') // 1024} KB")
