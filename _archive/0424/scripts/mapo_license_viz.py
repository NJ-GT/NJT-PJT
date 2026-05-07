# -*- coding: utf-8 -*-
import pandas as pd, numpy as np, sys
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.gridspec import GridSpec
sys.stdout.reconfigure(encoding='utf-8')

for font in fm.findSystemFonts():
    if 'malgun' in font.lower():
        plt.rcParams['font.family'] = fm.FontProperties(fname=font).get_name()
        break
plt.rcParams['axes.unicode_minus'] = False

BASE = 'c:/Users/USER/Documents/GitHub/기말공모전/NJT-PJT'

def load_mapo(path):
    df = pd.read_csv(path, encoding='utf-8-sig', low_memory=False)
    df = df[df['지번주소'].str.contains('마포구', na=False)].copy()
    df['인허가연도'] = pd.to_datetime(df['인허가일자'], errors='coerce').dt.year
    return df

f1 = load_mapo(f'{BASE}/원본데이터/서울시 외국인관광도시민박업 인허가 정보.csv')
f2 = load_mapo(f'{BASE}/원본데이터/서울시 관광숙박업 인허가 정보.csv')
f3 = load_mapo(f'{BASE}/원본데이터/서울시 숙박업 인허가 정보.csv')

YEARS = list(range(2020, 2025))

민박_yr    = f1[f1['인허가연도'].between(2020,2024)].groupby('인허가연도').size().reindex(YEARS, fill_value=0)
민박_영업  = f1[f1['인허가연도'].between(2020,2024) & (f1['영업상태명']=='영업/정상')].groupby('인허가연도').size().reindex(YEARS, fill_value=0)

f1['동'] = f1['지번주소'].str.extract(r'마포구\s+(\S+동|\S+가)')
동_pivot  = f1[f1['인허가연도'].between(2020,2024)].groupby(['동','인허가연도']).size().unstack(fill_value=0).reindex(columns=YEARS, fill_value=0)
top_dong  = 동_pivot.sum(axis=1).nlargest(8).index
동_pivot  = 동_pivot.loc[top_dong]

f3['업태'] = f3['위생업태명']
숙박_yr   = f3[f3['인허가연도'].between(2020,2024)].groupby(['인허가연도','업태']).size().unstack(fill_value=0).reindex(index=YEARS, fill_value=0)

f2['업태'] = f2['관광숙박업상세명']
관광_yr   = f2[f2['인허가연도'].between(2020,2024)].groupby(['인허가연도','업태']).size().unstack(fill_value=0).reindex(index=YEARS, fill_value=0)

C_MAIN  = '#2D5BE3'
C_LIVE  = '#00C49F'
PALETTE = ['#2D5BE3','#00C49F','#FF6B6B','#FFD166','#A855F7','#F97316','#14B8A6','#EC4899']
BG      = '#F8F9FC'

fig = plt.figure(figsize=(18, 12), facecolor=BG)
fig.suptitle('마포구 숙박업 신규 인허가 트렌드 (2020–2024)',
             fontsize=18, fontweight='bold', y=0.98, color='#1A1A2E')

gs = GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35,
              left=0.07, right=0.96, top=0.92, bottom=0.08)

# [1] 외국인민박 연도별 막대+선
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor('white')
bars = ax1.bar(YEARS, 민박_yr.values, color=C_MAIN, alpha=0.85, width=0.55, zorder=3, label='신규인허가')
ax1.plot(YEARS, 민박_영업.values, 'o-', color=C_LIVE, lw=2.5, ms=7, zorder=4, label='영업중')
for bar, val in zip(bars, 민박_yr.values):
    ax1.text(bar.get_x()+bar.get_width()/2, val+5, str(int(val)),
             ha='center', va='bottom', fontsize=10, fontweight='bold', color=C_MAIN)
ax1.set_title('외국인관광 도시민박업\n연도별 신규 인허가', fontsize=11, fontweight='bold', pad=8)
ax1.set_ylabel('신규 인허가 수', fontsize=9)
ax1.set_xticks(YEARS)
ax1.legend(fontsize=8, loc='upper left')
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.spines[['top','right']].set_visible(False)
ax1.set_ylim(0, max(민박_yr.values)*1.2)

# [2] 동별 히트맵
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_facecolor('white')
im = ax2.imshow(동_pivot.values, aspect='auto', cmap='Blues', vmin=0)
ax2.set_xticks(range(len(YEARS))); ax2.set_xticklabels(YEARS, fontsize=9)
ax2.set_yticks(range(len(top_dong))); ax2.set_yticklabels(top_dong, fontsize=9)
for i in range(len(top_dong)):
    for j in range(len(YEARS)):
        v = int(동_pivot.values[i, j])
        ax2.text(j, i, str(v) if v>0 else '', ha='center', va='center',
                 fontsize=9, fontweight='bold',
                 color='white' if v > 동_pivot.values.max()*0.6 else '#333')
ax2.set_title('동별 외국인민박 신규 인허가\n(상위 8개 동)', fontsize=11, fontweight='bold', pad=8)
plt.colorbar(im, ax=ax2, shrink=0.8, label='건수')
ax2.spines[['top','right','left','bottom']].set_visible(False)

# [3] 핵심 3개 동 선그래프
ax3 = fig.add_subplot(gs[0, 2])
ax3.set_facecolor('white')
top3 = ['연남동','서교동','동교동']
for i, dong in enumerate(top3):
    if dong in 동_pivot.index:
        vals = 동_pivot.loc[dong].values
        ax3.plot(YEARS, vals, 'o-', color=PALETTE[i], lw=2.5, ms=7, label=dong, zorder=3)
        ax3.fill_between(YEARS, vals, alpha=0.08, color=PALETTE[i])
        ax3.text(YEARS[-1]+0.05, vals[-1], dong, va='center', fontsize=9, color=PALETTE[i], fontweight='bold')
ax3.set_title('핵심 3개 동 연도별 추이', fontsize=11, fontweight='bold', pad=8)
ax3.set_ylabel('신규 인허가 수', fontsize=9)
ax3.set_xticks(YEARS)
ax3.legend(fontsize=8)
ax3.grid(axis='y', alpha=0.3, linestyle='--')
ax3.spines[['top','right']].set_visible(False)

# [4] 숙박업 업태별 누적 막대
ax4 = fig.add_subplot(gs[1, 0])
ax4.set_facecolor('white')
if not 숙박_yr.empty and 숙박_yr.sum().sum() > 0:
    cols = 숙박_yr.sum().sort_values(ascending=False).index
    bottom = np.zeros(len(YEARS))
    for i, col in enumerate(cols):
        vals = 숙박_yr[col].values if col in 숙박_yr.columns else np.zeros(len(YEARS))
        ax4.bar(YEARS, vals, bottom=bottom, color=PALETTE[i], alpha=0.85, width=0.55, label=col, zorder=3)
        bottom += vals
ax4.set_title('숙박업 업태별 신규 인허가\n(위생업태 기준)', fontsize=11, fontweight='bold', pad=8)
ax4.set_ylabel('신규 인허가 수', fontsize=9)
ax4.set_xticks(YEARS)
ax4.legend(fontsize=8, loc='upper right')
ax4.grid(axis='y', alpha=0.3, linestyle='--')
ax4.spines[['top','right']].set_visible(False)

# [5] 관광숙박업
ax5 = fig.add_subplot(gs[1, 1])
ax5.set_facecolor('white')
if not 관광_yr.empty and 관광_yr.sum().sum() > 0:
    cols2 = 관광_yr.sum().sort_values(ascending=False).index
    bottom2 = np.zeros(len(YEARS))
    for i, col in enumerate(cols2):
        vals = 관광_yr[col].values if col in 관광_yr.columns else np.zeros(len(YEARS))
        ax5.bar(YEARS, vals, bottom=bottom2, color=PALETTE[i+3], alpha=0.85, width=0.55, label=col, zorder=3)
        bottom2 += vals
ax5.set_title('관광숙박업 업태별 신규 인허가', fontsize=11, fontweight='bold', pad=8)
ax5.set_ylabel('신규 인허가 수', fontsize=9)
ax5.set_xticks(YEARS)
ax5.legend(fontsize=8)
ax5.grid(axis='y', alpha=0.3, linestyle='--')
ax5.spines[['top','right']].set_visible(False)

# [6] 2020 vs 2024 업종별 비교
ax6 = fig.add_subplot(gs[1, 2])
ax6.set_facecolor('white')
summary = {
    '외국인관광\n도시민박업': (int(민박_yr[2020]), int(민박_yr[2024])),
    '숙박업\n(합계)':        (int(숙박_yr.sum(axis=1).get(2020, 0)), int(숙박_yr.sum(axis=1).get(2024, 0))),
    '관광숙박업\n(합계)':    (int(관광_yr.sum(axis=1).get(2020, 0)), int(관광_yr.sum(axis=1).get(2024, 0))),
}
labels = list(summary.keys())
v2020 = [summary[k][0] for k in labels]
v2024 = [summary[k][1] for k in labels]
x = np.arange(len(labels))
w = 0.32
ax6.bar(x-w/2, v2020, width=w, color='#94A3B8', alpha=0.85, label='2020년', zorder=3)
ax6.bar(x+w/2, v2024, width=w, color=C_MAIN,   alpha=0.85, label='2024년', zorder=3)
for i,a,b) in enumerate(zip(v2020, v2024)):
    ax6.text(i-w/2, a+1, str(a), ha='center', va='bottom', fontsize=9, color='#64748B', fontweight='bold')
    ax6.text(i+w/2, b+1, str(b), ha='center', va='bottom', fontsize=9, color=C_MAIN,   fontweight='bold')
ax6.set_title('2020 vs 2024\n업종별 신규 인허가 비교', fontsize=11, fontweight='bold', pad=8)
ax6.set_xticks(x); ax6.set_xticklabels(labels, fontsize=9)
ax6.legend(fontsize=9)
ax6.grid(axis='y', alpha=0.3, linestyle='--')
ax6.spines[['top','right']].set_visible(False)

plt.savefig(f'{BASE}/data/mapo_license_trend.png', dpi=150, bbox_inches='tight', facecolor=BG)
print('저장 완료: data/mapo_license_trend.png')
