# -*- coding: utf-8 -*-
import sys, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import gaussian_kde
sys.stdout.reconfigure(encoding='utf-8')

BASE = 'c:/Users/USER/Documents/GitHub/기말공모전/NJT-PJT'
FIRE_PATH = f'{BASE}/data/화재출동/화재출동_2021_2024.csv'
ACC_PATH  = f'{BASE}/data/서울10구_숙소_소방거리_유클리드.csv'
OUT_PATH  = f'{BASE}/data/fire_kde_map.png'

# ── 데이터 로드 ───────────────────────────────────────────────────────
fire = pd.read_csv(FIRE_PATH, encoding='utf-8-sig', low_memory=False)
acc  = pd.read_csv(ACC_PATH,  encoding='utf-8-sig', on_bad_lines='skip')

target_gu = ['종로구','중구','용산구','성동구','광진구','마포구',
             '서대문구','은평구','서초구','강남구','송파구','강서구','영등포구']
fire = fire[(fire['위도']>37.4)&(fire['위도']<37.7)&
            (fire['경도']>126.7)&(fire['경도']<127.3)&
            fire['발생시군구'].isin(target_gu)].copy()
print(f'화재: {len(fire)}건 | 숙박시설: {len(acc)}개')

# ── KDE ──────────────────────────────────────────────────────────────
xy  = np.vstack([fire['경도'].values, fire['위도'].values])
kde = gaussian_kde(xy, bw_method=0.03)

lon_min, lon_max = 126.82, 127.18
lat_min, lat_max = 37.44, 37.65

GRID_N = 300
grid_lon, grid_lat = np.mgrid[lon_min:lon_max:GRID_N*1j, lat_min:lat_max:GRID_N*1j]
kde_values = kde(np.vstack([grid_lon.ravel(), grid_lat.ravel()])).reshape(GRID_N, GRID_N)
kde_norm   = (kde_values - kde_values.min()) / (kde_values.max() - kde_values.min())
print('KDE 완료')

# ── 시각화 ────────────────────────────────────────────────────────────
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(14, 12), dpi=150)

# KDE 컨투어
cmap       = plt.get_cmap('YlOrRd')
cmap_alpha = cmap(np.linspace(0, 1, 256))
cmap_alpha[:30, 3] = 0
cmap_alpha[30:60, 3] = np.linspace(0, 0.5, 30)
custom_cmap = mcolors.LinearSegmentedColormap.from_list('fire_alpha', cmap_alpha)
cf = ax.contourf(grid_lon, grid_lat, kde_norm, levels=25, cmap=custom_cmap, alpha=0.85)

# 숙박시설
ax.scatter(acc['경도'], acc['위도'], s=6, c='#2980b9', alpha=0.5, zorder=3, label=f'숙박시설 ({len(acc):,}개)')

# 컬러바
cbar = fig.colorbar(cf, ax=ax, shrink=0.5, pad=0.02)
cbar.set_label('화재 발생 밀도 (KDE)', fontsize=11)

ax.set_xlim(lon_min, lon_max)
ax.set_ylim(lat_min, lat_max)
ax.set_xlabel('경도', fontsize=11)
ax.set_ylabel('위도', fontsize=11)
ax.set_title(f'서울 10구 화재출동 KDE 밀도 + 숙박시설 분포\n(화재출동 {len(fire):,}건, 2017~2024)', fontsize=14, pad=15)
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150, bbox_inches='tight')
plt.close()
print(f'[저장 완료] {OUT_PATH}')
