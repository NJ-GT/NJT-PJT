# -*- coding: utf-8 -*-
import pandas as pd, geopandas as gpd, matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import contextily as ctx
from shapely.geometry import Point

SAFE_DIST   = 1000
GOLDEN_DIST = 2000
DANGER_DIST = 3000

BASE    = 'c:/Users/USER/Documents/GitHub/기말공모전/NJT-PJT/data'
IN_CSV  = f'{BASE}/서울10구_숙소_소방거리_유클리드.csv'
FIRE_CSV= f'{BASE}/소방서_안전센터_구조대_위치정보_2025_wgs84.csv'
OUT_PNG = f'{BASE}/Map_Seoul10_Firestation.png'

df   = pd.read_csv(IN_CSV,   encoding='utf-8-sig')
fire = pd.read_csv(FIRE_CSV, encoding='utf-8-sig')

def get_color(d):
    if d <= SAFE_DIST:   return '#27ae60'
    if d <= GOLDEN_DIST: return '#e67e22'
    if d <= DANGER_DIST: return '#c0392b'
    return '#7f0000'

df['color'] = df['최근접_거리m'].apply(get_color)

gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['경도'], df['위도']), crs='EPSG:4326').to_crs('EPSG:3857')
fire_gdf = gpd.GeoDataFrame(fire, geometry=gpd.points_from_xy(fire['경도'], fire['위도']), crs='EPSG:4326').to_crs('EPSG:3857')
fs_gdf = fire_gdf[fire_gdf['시설유형'] == '소방서']
sc_gdf = fire_gdf[fire_gdf['시설유형'] == '안전센터/구조대']

fig, ax = plt.subplots(figsize=(16, 13))

# 숙박시설 마커
ax.scatter(gdf.geometry.x, gdf.geometry.y,
           c=gdf['color'], s=18, alpha=0.8, zorder=3, linewidths=0.3, edgecolors='white')

# 소방서
ax.scatter(fs_gdf.geometry.x, fs_gdf.geometry.y,
           c='red', s=120, marker='*', zorder=5, linewidths=0.5, edgecolors='darkred', label='소방서')
# 안전센터
ax.scatter(sc_gdf.geometry.x, sc_gdf.geometry.y,
           c='orange', s=50, marker='^', zorder=4, linewidths=0.4, edgecolors='darkorange', label='안전센터')

# 배경 지도
ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zoom=12)

# 범례
patches = [
    mpatches.Patch(color='#27ae60', label=f'~{SAFE_DIST}m · 총3분 이내 (안전): {(df["최근접_거리m"]<=SAFE_DIST).sum()}개'),
    mpatches.Patch(color='#e67e22', label=f'~{GOLDEN_DIST}m · 골든타임 경계: {((df["최근접_거리m"]>SAFE_DIST)&(df["최근접_거리m"]<=GOLDEN_DIST)).sum()}개'),
    mpatches.Patch(color='#c0392b', label=f'~{DANGER_DIST}m · 위험: {((df["최근접_거리m"]>GOLDEN_DIST)&(df["최근접_거리m"]<=DANGER_DIST)).sum()}개'),
    mpatches.Patch(color='#7f0000', label=f'{DANGER_DIST}m+ · 매우위험: {(df["최근접_거리m"]>DANGER_DIST).sum()}개'),
    plt.Line2D([0],[0], marker='*', color='w', markerfacecolor='red', markersize=12, label='소방서'),
    plt.Line2D([0],[0], marker='^', color='w', markerfacecolor='orange', markersize=8, label='안전센터'),
]
ax.legend(handles=patches, loc='lower left', fontsize=9,
          framealpha=0.95, edgecolor='gray',
          prop={'family': 'Malgun Gothic', 'size': 9})

ax.set_title('서울 10개구 숙박시설 ↔ 소방시설 유클리드 거리\n(소방차 30km/h · 골든타임 5분 기준)',
             fontsize=14, fontfamily='Malgun Gothic', pad=12)
ax.axis('off')

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150, bbox_inches='tight', facecolor='white')
print(f'[저장] {OUT_PNG}')
