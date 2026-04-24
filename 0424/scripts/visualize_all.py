# -*- coding: utf-8 -*-
"""통합숙박시설 전체 시각화 스크립트"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch
import seaborn as sns
import folium
from folium.plugins import HeatMap
import sys, os

sys.stdout.reconfigure(encoding='utf-8')

for font in fm.findSystemFonts():
    if 'malgun' in font.lower():
        plt.rcParams['font.family'] = fm.FontProperties(fname=font).get_name()
        break
plt.rcParams['axes.unicode_minus'] = False

BASE = 'c:/Users/USER/Documents/GitHub/기말공모전/NJT-PJT/data'
OUT  = f'{BASE}/viz_all'
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(f'{BASE}/통합숙박시설_최종안0421.csv', encoding='utf-8-sig')
GU_LIST = sorted(df['구'].unique())
GU_COLORS = dict(zip(GU_LIST, plt.cm.tab10.colors[:len(GU_LIST)]))
print(f'로드: {len(df)}행')

# ══════════════════════════════════════════════════════════════
# 1. 위치 산포도 — 구별 색상
# ══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 10))
for gu in GU_LIST:
    sub = df[df['구'] == gu]
    ax.scatter(sub['경도'], sub['위도'], c=[GU_COLORS[gu]], s=8, alpha=0.6, label=gu)
ax.set_title('서울 10개 구 숙박시설 위치 분포', fontsize=14, fontweight='bold')
ax.set_xlabel('경도'); ax.set_ylabel('위도')
ax.legend(loc='lower right', fontsize=8, ncol=2, markerscale=2)
ax.set_facecolor('#f0f0f0')
plt.tight_layout()
plt.savefig(f'{OUT}/01_위치_산포도.png', dpi=150, bbox_inches='tight')
plt.close(); print('01 완료')

# ══════════════════════════════════════════════════════════════
# 2. 건물나이 바이올린 플롯 — 구별
# ══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 6))
data_by_gu = [df[df['구']==gu]['건물나이'].dropna().values for gu in GU_LIST]
vp = ax.violinplot(data_by_gu, positions=range(len(GU_LIST)), showmedians=True, showextrema=True)
for i, body in enumerate(vp['bodies']):
    body.set_facecolor(list(GU_COLORS.values())[i])
    body.set_alpha(0.75)
vp['cmedians'].set_color('white'); vp['cmedians'].set_linewidth(2)
ax.set_xticks(range(len(GU_LIST))); ax.set_xticklabels(GU_LIST, rotation=30)
ax.set_ylabel('건물나이 (년)')
ax.set_title('구별 숙박시설 건물나이 분포', fontsize=14, fontweight='bold')
ax.set_facecolor('#fafafa')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT}/02_건물나이_바이올린.png', dpi=150, bbox_inches='tight')
plt.close(); print('02 완료')

# ══════════════════════════════════════════════════════════════
# 3. 연도별 신규 숙박시설 — 구별 스택 영역 차트
# ══════════════════════════════════════════════════════════════
yearly = df[df['승인연도'] >= 2000].groupby(['승인연도','구']).size().unstack(fill_value=0)
fig, ax = plt.subplots(figsize=(14, 6))
yearly.plot.area(ax=ax, alpha=0.8, colormap='tab10')
ax.set_title('연도별 신규 숙박시설 수 (2000~)', fontsize=14, fontweight='bold')
ax.set_xlabel('승인연도'); ax.set_ylabel('신규 숙박시설 수')
ax.legend(loc='upper left', fontsize=8, ncol=2)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT}/03_연도별_신규_스택.png', dpi=150, bbox_inches='tight')
plt.close(); print('03 완료')

# ══════════════════════════════════════════════════════════════
# 4. 소방 골든타임 분석 — 도로폭보정 예상도착시간
# ══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle('소방 골든타임 분석 (5분=300초 기준)', fontsize=14, fontweight='bold')

golden = 300
gu_over = df.groupby('구').apply(lambda x: (x['도로폭_보정예상도착초'] > golden).mean()*100).sort_values(ascending=False)
colors = ['#e74c3c' if v > 50 else '#e67e22' if v > 30 else '#2ecc71' for v in gu_over.values]
bars = axes[0].bar(gu_over.index, gu_over.values, color=colors)
for bar, v in zip(bars, gu_over.values):
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5, f'{v:.0f}%', ha='center', fontsize=9)
axes[0].axhline(50, color='red', lw=1.5, linestyle='--', label='50% 기준선')
axes[0].set_xlabel('구'); axes[0].set_ylabel('골든타임 초과율 (%)')
axes[0].set_title('구별 골든타임(5분) 초과 비율'); axes[0].legend()
axes[0].tick_params(axis='x', rotation=30)

data_box = [df[df['구']==gu]['도로폭_보정예상도착초'].dropna().values for gu in GU_LIST]
bp = axes[1].boxplot(data_box, labels=GU_LIST, patch_artist=True, notch=True)
for patch, gu in zip(bp['boxes'], GU_LIST):
    patch.set_facecolor(GU_COLORS[gu]); patch.set_alpha(0.7)
axes[1].axhline(golden, color='red', lw=1.5, linestyle='--', label='골든타임 300초')
axes[1].set_ylabel('예상도착시간 (초)'); axes[1].set_title('구별 예상도착시간 분포')
axes[1].legend(); axes[1].tick_params(axis='x', rotation=30)
plt.tight_layout()
plt.savefig(f'{OUT}/04_소방_골든타임.png', dpi=150, bbox_inches='tight')
plt.close(); print('04 완료')

# ══════════════════════════════════════════════════════════════
# 5. AHP vs PCA 산점도 — 군집 색상
# ══════════════════════════════════════════════════════════════
df_s = df.dropna(subset=['위험점수_PCA','위험점수_AHP'])
fig, ax = plt.subplots(figsize=(10, 8))
clusters = sorted(df_s['군집'].unique())
cmap = plt.cm.Set1
for cl in clusters:
    sub = df_s[df_s['군집']==cl]
    ax.scatter(sub['위험점수_PCA'], sub['위험점수_AHP'],
               c=[cmap(cl/max(clusters))], s=15, alpha=0.5, label=f'군집 {cl}')
ax.set_xlabel('위험점수_PCA', fontsize=11)
ax.set_ylabel('위험점수_AHP', fontsize=11)
ax.set_title('PCA vs AHP 위험점수 — 군집별 분포', fontsize=14, fontweight='bold')
ax.legend(fontsize=10, markerscale=2)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT}/05_PCA_AHP_산점도.png', dpi=150, bbox_inches='tight')
plt.close(); print('05 완료')

# ══════════════════════════════════════════════════════════════
# 6. 군집별 레이더 차트
# ══════════════════════════════════════════════════════════════
radar_vars = ['건물나이','반경_50m_건물수','집중도(%)','구조_노후_통합점수','도로폭_위험도','소방위험도_점수']
radar_labels = ['건물나이','주변건물수','집중도','구조노후도','도로폭위험도','소방위험도']
N = len(radar_vars)
angles = [n/float(N)*2*np.pi for n in range(N)] + [0]

fig, axes = plt.subplots(1, len(clusters), figsize=(5*len(clusters), 5), subplot_kw=dict(polar=True))
fig.suptitle('군집별 특성 레이더 차트', fontsize=14, fontweight='bold')
if len(clusters) == 1: axes = [axes]

# 정규화
norm_df = df.copy()
for v in radar_vars:
    mn, mx = df[v].min(), df[v].max()
    norm_df[v] = (df[v]-mn)/(mx-mn+1e-9)

for ax, cl in zip(axes, clusters):
    vals = norm_df[norm_df['군집']==cl][radar_vars].mean().tolist() + \
           [norm_df[norm_df['군집']==cl][radar_vars].mean().tolist()[0]]
    ax.plot(angles, vals, 'o-', linewidth=2, color=cmap(cl/max(clusters)))
    ax.fill(angles, vals, alpha=0.25, color=cmap(cl/max(clusters)))
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(radar_labels, fontsize=9)
    ax.set_title(f'군집 {cl}\n(N={len(df[df["군집"]==cl])})', fontsize=11, fontweight='bold', pad=15)
    ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig(f'{OUT}/06_군집_레이더.png', dpi=150, bbox_inches='tight')
plt.close(); print('06 완료')

# ══════════════════════════════════════════════════════════════
# 7. 상관 히트맵
# ══════════════════════════════════════════════════════════════
corr_vars = ['건물나이','반경_50m_건물수','집중도(%)','구조_노후_통합점수',
             '도로폭_위험도','소방위험도_점수','위험점수_AHP','안전센터_유클리드m','상업비율(%)']
corr_labels = ['건물나이','주변건물수','집중도','구조노후도','도로폭위험도','소방위험도','AHP위험점수','안전센터거리','상업비율']
corr = df[corr_vars].corr()
corr.index = corr.columns = corr_labels

fig, ax = plt.subplots(figsize=(11, 9))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, ax=ax, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, linewidths=0.5,
            annot_kws={'size': 10}, cbar_kws={'shrink': 0.8})
ax.set_title('주요 변수 상관 히트맵', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/07_상관_히트맵.png', dpi=150, bbox_inches='tight')
plt.close(); print('07 완료')

# ══════════════════════════════════════════════════════════════
# 8. 건물나이 vs 구조노후도 — Hexbin
# ══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 7))
hb = ax.hexbin(df['건물나이'], df['구조_노후_통합점수'], gridsize=40, cmap='YlOrRd', mincnt=1)
plt.colorbar(hb, ax=ax, label='숙소 수')
ax.set_xlabel('건물나이 (년)', fontsize=11)
ax.set_ylabel('구조_노후_통합점수', fontsize=11)
ax.set_title('건물나이 vs 구조노후도 (Hexbin)', fontsize=14, fontweight='bold')
ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(f'{OUT}/08_건물나이_구조노후도_hexbin.png', dpi=150, bbox_inches='tight')
plt.close(); print('08 완료')

# ══════════════════════════════════════════════════════════════
# 9. 구별 평균 위험점수_AHP — 수평 barplot
# ══════════════════════════════════════════════════════════════
gu_ahp = df.groupby('구')['위험점수_AHP'].mean().sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(10, 6))
norm = mcolors.Normalize(vmin=gu_ahp.min(), vmax=gu_ahp.max())
colors = [cm.RdYlGn_r(norm(v)) for v in gu_ahp.values]
bars = ax.barh(gu_ahp.index, gu_ahp.values, color=colors)
for bar, v in zip(bars, gu_ahp.values):
    ax.text(bar.get_width()+0.3, bar.get_y()+bar.get_height()/2, f'{v:.1f}', va='center', fontsize=9)
ax.set_xlabel('평균 AHP 위험점수')
ax.set_title('구별 평균 AHP 위험점수\n(높을수록 위험)', fontsize=14, fontweight='bold')
sm = cm.ScalarMappable(cmap='RdYlGn_r', norm=norm)
plt.colorbar(sm, ax=ax, label='위험점수')
plt.tight_layout()
plt.savefig(f'{OUT}/09_구별_AHP위험점수.png', dpi=150, bbox_inches='tight')
plt.close(); print('09 완료')

# ══════════════════════════════════════════════════════════════
# 10. 주요_시설군 구별 누적 barplot
# ══════════════════════════════════════════════════════════════
sil_pivot = df.groupby(['구','주요_시설군']).size().unstack(fill_value=0)
fig, ax = plt.subplots(figsize=(13, 6))
sil_pivot.plot.bar(ax=ax, stacked=True, colormap='Set2', alpha=0.9)
ax.set_title('구별 숙박시설 주변 시설군 구성', fontsize=14, fontweight='bold')
ax.set_xlabel('구'); ax.set_ylabel('숙소 수')
ax.legend(title='주요시설군', bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=9)
ax.tick_params(axis='x', rotation=30)
plt.tight_layout()
plt.savefig(f'{OUT}/10_시설군_구성.png', dpi=150, bbox_inches='tight')
plt.close(); print('10 완료')

# ══════════════════════════════════════════════════════════════
# 11. 층수 분포 — 구별 boxplot
# ══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(13, 6))
df_floor = df[df['총층수'] <= 30]
data_floor = [df_floor[df_floor['구']==gu]['총층수'].dropna().values for gu in GU_LIST]
bp = ax.boxplot(data_floor, labels=GU_LIST, patch_artist=True, notch=False)
for patch, gu in zip(bp['boxes'], GU_LIST):
    patch.set_facecolor(GU_COLORS[gu]); patch.set_alpha(0.8)
ax.set_ylabel('총층수'); ax.set_title('구별 숙박시설 층수 분포 (30층 이하)', fontsize=14, fontweight='bold')
ax.tick_params(axis='x', rotation=30); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT}/11_층수_분포.png', dpi=150, bbox_inches='tight')
plt.close(); print('11 완료')

# ══════════════════════════════════════════════════════════════
# 12. Folium 히트맵 — 위험점수_AHP
# ══════════════════════════════════════════════════════════════
m = folium.Map(location=[37.555, 126.985], zoom_start=12, tiles='CartoDB dark_matter')
heat_data = df[['위도','경도','위험점수_AHP']].dropna().values.tolist()
HeatMap(heat_data, radius=12, blur=15, max_zoom=14,
        gradient={0.2:'blue', 0.5:'lime', 0.8:'orange', 1.0:'red'}).add_to(m)
folium.LayerControl().add_to(m)
m.save(f'{OUT}/12_위험점수_히트맵.html')
print('12 완료 (HTML)')

# ══════════════════════════════════════════════════════════════
# 13. 연면적 분포 — 로그스케일 히스토그램
# ══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(11, 5))
df_area = df[df['연면적(㎡)'] > 0]['연면적(㎡)']
ax.hist(np.log10(df_area), bins=60, color='#3498db', edgecolor='white', alpha=0.85)
ax.set_xlabel('log₁₀(연면적, ㎡)')
ax.set_ylabel('빈도')
ax.set_title('숙박시설 연면적 분포 (로그스케일)', fontsize=14, fontweight='bold')
ticks = [1,2,3,4,5]
ax.set_xticks(ticks)
ax.set_xticklabels([f'10^{t}\n({10**t:,}㎡)' for t in ticks])
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT}/13_연면적_분포.png', dpi=150, bbox_inches='tight')
plt.close(); print('13 완료')

print(f'\n모든 시각화 완료 → {OUT}')
