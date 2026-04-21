# -*- coding: utf-8 -*-
"""
[파일 설명]
버퍼 분석(반경 50m 건물 수) 결과에 커널 밀도 추정(KDE)을 적용하여
숙박시설 주변 건물 밀집도를 연속 밀도 곡면으로 시각화하는 스크립트.

[기존 격자 방식과의 차이]
  - 격자 방식: 공간을 50,000m² 격자로 균등 분할 후 건물 수 집계 (공간 중심)
  - 버퍼+KDE : 각 숙박시설을 중심으로 반경 50m 버퍼 안 건물 수를 가중치로
               scipy 가우시안 KDE를 적용 → 연속 밀도 곡면 (시설 중심)

[처리 흐름]
  1. XY_GIS_Analysis_Summary.csv 로드 (gis_analysis.py 출력, 버퍼 결과 포함)
  2. EPSG:5186 → WGS84 좌표 변환
  3. scipy.stats.gaussian_kde로 가중 KDE 계산 (가중치 = 반경_50m_건물수)
  4. 서울 전역 200×200 격자에서 KDE 값 평가
  5. matplotlib으로 KDE 컨투어 이미지 생성 → base64 인코딩
  6. Folium 지도에 이미지 오버레이 + 개별 포인트 마커 레이어 추가
  7. HTML 저장

[입력]  data/XY_GIS_Analysis_Summary.csv  (버퍼 분석 결과)
[출력]  data/Map_Buffer_KDE.html           (KDE 시각화 지도)
"""

import os
import base64
from io import BytesIO

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 화면 없이 이미지 생성 (서버/스크립트 환경)
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import gaussian_kde
from pyproj import Transformer
import folium
from folium.plugins import HeatMap

# ── 경로 설정 ─────────────────────────────────────────────────────────
BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
INPUT_PATH  = os.path.join(BASE, 'data', 'XY_GIS_Analysis_Summary.csv')
OUTPUT_PATH = os.path.join(BASE, 'data', 'Map_Buffer_KDE.html')

# ── 1. 버퍼 분석 결과 로드 ────────────────────────────────────────────
print("Loading buffer analysis results...")
df = pd.read_csv(INPUT_PATH, encoding='utf-8-sig')
# 컬럼명을 직접 지정 (터미널 인코딩 문제 방지)
df.columns = ['인덱스', '보정_X', '보정_Y', '반경_50m_건물수',
              '주택_수', '상업_수', '숙박_수', '사무_수', '기타_수',
              '밀집도', '주요_시설군', '집중도(%)']
print(f"  총 {len(df)}개 숙박시설 로드")

# ── 2. 좌표 변환: EPSG:5186 → WGS84 ──────────────────────────────────
print("Converting coordinates EPSG:5186 → WGS84...")
transformer = Transformer.from_crs("EPSG:5186", "EPSG:4326", always_xy=True)
lons, lats = transformer.transform(df['보정_X'].values, df['보정_Y'].values)
df['lat'] = lats
df['lon'] = lons

# 서울 범위 외 이상 좌표 제거
df = df[(df['lat'] > 37.4) & (df['lat'] < 37.7) &
        (df['lon'] > 126.7) & (df['lon'] < 127.3)]
print(f"  유효 좌표: {len(df)}개")

# ── 3. 가중 KDE 계산 ──────────────────────────────────────────────────
# 가중치: 반경 50m 버퍼 안의 건물 수
# 건물이 많을수록 그 숙박시설 위치의 밀도 기여가 커짐
print("Computing weighted Kernel Density Estimation...")
weights = df['반경_50m_건물수'].values.astype(float)

# 가중치가 모두 0인 경우 균등 가중치로 대체
if weights.sum() == 0:
    weights = np.ones(len(df))
weights = weights / weights.sum()  # 정규화 (합 = 1)

# gaussian_kde: lon/lat 2차원 포인트에 가우시안 커널 적용
xy = np.vstack([df['lon'].values, df['lat'].values])
# bw_method: 대역폭 (클수록 더 매끄럽게, 작을수록 더 날카롭게)
kde = gaussian_kde(xy, weights=weights, bw_method=0.04)

# ── 4. 서울 전역 격자에서 KDE 값 평가 ────────────────────────────────
print("Evaluating KDE on 200x200 grid...")
# 데이터 범위보다 약간 넓게 격자 생성
lat_min = df['lat'].min() - 0.02
lat_max = df['lat'].max() + 0.02
lon_min = df['lon'].min() - 0.02
lon_max = df['lon'].max() + 0.02

# 200×200 격자 (해상도와 속도의 균형)
GRID_N = 200
grid_lon, grid_lat = np.mgrid[lon_min:lon_max:GRID_N*1j,
                               lat_min:lat_max:GRID_N*1j]
grid_points = np.vstack([grid_lon.ravel(), grid_lat.ravel()])
kde_values  = kde(grid_points).reshape(GRID_N, GRID_N)

# 시각화를 위해 0~1 정규화
kde_norm = (kde_values - kde_values.min()) / (kde_values.max() - kde_values.min())

# ── 5. KDE 곡면을 투명 PNG로 렌더링 ──────────────────────────────────
print("Rendering KDE surface as transparent PNG...")
fig, ax = plt.subplots(figsize=(10, 10), dpi=150)

# YlOrRd 컬러맵: 낮은 밀도(노랑) → 높은 밀도(진빨강)
cmap = plt.get_cmap('YlOrRd')
# 낮은 값(0~10%)은 투명하게 처리하여 배경 지도가 보이도록
cmap_alpha = cmap(np.linspace(0, 1, 256))
cmap_alpha[:25, 3] = 0    # 하위 10% 완전 투명
cmap_alpha[25:50, 3] = np.linspace(0, 0.4, 25)  # 10~20% 점진 투명
custom_cmap = mcolors.LinearSegmentedColormap.from_list('YlOrRd_alpha', cmap_alpha)

# 컨투어 채우기 (20개 등고선 단계)
ax.contourf(grid_lon, grid_lat, kde_norm,
            levels=20, cmap=custom_cmap, alpha=0.85)
ax.set_xlim(lon_min, lon_max)
ax.set_ylim(lat_min, lat_max)
ax.axis('off')

# 배경 투명하게
fig.patch.set_alpha(0)
ax.patch.set_alpha(0)

# PNG를 메모리에 저장 후 base64 인코딩 (HTML에 내장)
buf = BytesIO()
fig.savefig(buf, format='png', bbox_inches='tight',
            pad_inches=0, transparent=True, dpi=150)
buf.seek(0)
img_b64 = base64.b64encode(buf.read()).decode('utf-8')
plt.close(fig)
print("  PNG 렌더링 완료")

# ── 6. Folium 지도 생성 ───────────────────────────────────────────────
print("Building Folium map...")
center_lat = (lat_min + lat_max) / 2
center_lon = (lon_min + lon_max) / 2
m = folium.Map(location=[center_lat, center_lon],
               zoom_start=12,
               tiles='CartoDB positron')

# ── 레이어 1: KDE 연속 밀도 곡면 (이미지 오버레이) ─────────────────
fg_kde = folium.FeatureGroup(name='🌡️ KDE 건물 밀집도 곡면', show=True)
folium.raster_layers.ImageOverlay(
    image=f'data:image/png;base64,{img_b64}',
    bounds=[[lat_min, lon_min], [lat_max, lon_max]],
    opacity=0.75,
    name='KDE 곡면'
).add_to(fg_kde)
fg_kde.add_to(m)

# ── 레이어 2: HeatMap (Leaflet.heat KDE - 인터랙티브) ────────────────
# 버퍼 건물 수를 가중치로 사용한 히트맵 (zoom에 따라 동적으로 반응)
fg_heat = folium.FeatureGroup(name='🔥 HeatMap (동적 KDE)', show=False)
heat_data = [
    [row['lat'], row['lon'], float(row['반경_50m_건물수'])]
    for _, row in df.iterrows()
    if row['반경_50m_건물수'] > 0
]
HeatMap(heat_data, radius=20, blur=25, max_zoom=15,
        gradient={0.2: 'blue', 0.4: 'lime', 0.6: 'yellow',
                  0.8: 'orange', 1.0: 'red'}).add_to(fg_heat)
fg_heat.add_to(m)

# ── 레이어 3: 개별 숙박시설 마커 (버퍼 결과) ─────────────────────────
fg_pts = folium.FeatureGroup(name='📍 숙박시설 개별 포인트 (버퍼 결과)', show=False)
for _, row in df.iterrows():
    cnt = int(row['반경_50m_건물수'])
    # 색상: 30개 초과=빨강, 15~30=주황, 15미만=초록
    if cnt > 30:
        color = '#c0392b'
    elif cnt >= 15:
        color = '#e67e22'
    else:
        color = '#27ae60'

    popup_html = f"""
    <div style="font-family:'Malgun Gothic',sans-serif;width:180px">
      <b>ID: {int(row['인덱스'])}</b><br>
      <hr style="margin:4px 0">
      반경 50m 건물 수: <b>{cnt}개</b><br>
      주요 시설군: {row['주요_시설군']}<br>
      집중도: {row['집중도(%)']:.1f}%<br>
      <small>주택:{int(row['주택_수'])} 상업:{int(row['상업_수'])}
             숙박:{int(row['숙박_수'])} 사무:{int(row['사무_수'])}
             기타:{int(row['기타_수'])}</small>
    </div>"""

    folium.CircleMarker(
        location=[row['lat'], row['lon']],
        radius=4,
        color=color, fill=True, fill_color=color,
        fill_opacity=0.85, weight=0.8,
        popup=folium.Popup(popup_html, max_width=200),
        tooltip=f"ID:{int(row['인덱스'])} | 주변건물 {cnt}개"
    ).add_to(fg_pts)
fg_pts.add_to(m)

# ── 범례 & 레이어 컨트롤 ─────────────────────────────────────────────
folium.LayerControl(collapsed=False).add_to(m)

legend_html = f"""
<div style="position:fixed;bottom:30px;left:30px;z-index:1000;
            background:white;padding:14px 18px;border-radius:10px;
            box-shadow:2px 2px 8px rgba(0,0,0,0.3);font-size:12px;
            font-family:'Malgun Gothic',sans-serif;min-width:230px">
  <b>🏨 버퍼(50m) + KDE 건물 밀집도</b>
  <hr style="margin:8px 0">
  <b>KDE 밀도 (곡면 레이어)</b>
  <div style="width:200px;height:14px;
              background:linear-gradient(to right,
                rgba(255,255,204,0.2),#fecc5c,#fd8d3c,#e31a1c);
              border-radius:3px;margin:5px 0 2px"></div>
  <div style="display:flex;justify-content:space-between;font-size:10px;width:200px">
    <span>낮은 밀도</span><span>높은 밀도</span>
  </div>
  <hr style="margin:8px 0">
  <b>개별 포인트 (버퍼 분석)</b><br>
  <span style="color:#c0392b">●</span> 위험 &gt;30개 &nbsp;
  <span style="color:#e67e22">●</span> 주의 15~30개 &nbsp;
  <span style="color:#27ae60">●</span> 보통 &lt;15개<br>
  <hr style="margin:8px 0">
  <small>총 {len(df):,}개 숙박시설 | 대역폭 bw=0.04</small>
</div>"""
m.get_root().html.add_child(folium.Element(legend_html))

# ── 7. 저장 ───────────────────────────────────────────────────────────
m.save(OUTPUT_PATH)
print(f"\n[완료] 저장: {OUTPUT_PATH}")
print(f"   - Layer 1: scipy KDE 연속 곡면 (이미지 오버레이)")
print(f"   - Layer 2: Leaflet HeatMap (동적 KDE)")
print(f"   - Layer 3: 개별 포인트 (버퍼 50m 건물수)")
