# -*- coding: utf-8 -*-
import pandas as pd, numpy as np, sys, os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.gridspec import GridSpec
sys.stdout.reconfigure(encoding='utf-8')

for font in fm.findSystemFonts():
    if 'malgun' in font.lower():
        plt.rcParams['font.family'] = fm.FontProperties(fname=font).get_name()
        break
plt.rcParams['axes.unicode_minus'] = False

BASE  = 'c:/Users/USER/Documents/GitHub/기말공모전/NJT-PJT'
GU_10 = ['강남구','강서구','마포구','서초구','성동구','송파구','영등포구','용산구','종로구','중구']
YEARS = list(range(2020, 2026))
BG    = '#F8F9FC'
C_MAIN = '#2D5BE3'; C_LIVE = '#00C49F'; C_WARN = '#FF6B6B'
PAL   = ['#2D5BE3','#00C49F','#FF6B6B','#FFD166','#A855F7','#F97316','#14B8A6','#EC4899','#64748B','#EF4444']

OUT_DIR = f'{BASE}/data/viz_each_gu'
os.makedirs(OUT_DIR, exist_ok=True)

def load_gu(path):
    df = pd.read_csv(path, encoding='utf-8-sig', low_memory=False)
    df['인허가연도'] = pd.to_datetime(df['인허가일자'], errors='coerce').dt.year
    df['구'] = df['지번주소'].str.extract(r'서울특별시\s+(\S+구)')
    return df[df['구'].isin(GU_10)].copy()

f1 = load_gu(f'{BASE}/원본데이터/서울시 외국인관광도시민박업 인허가 정보.csv')
f2 = load_gu(f'{BASE}/원본데이터/서울시 관광숙박업 인허가 정보.csv')
f3 = load_gu(f'{BASE}/원본데이터/서울시 숙박업 인허가 정보.csv')

for GU in GU_10:
    g1 = f1[f1['구'] == GU].copy()
    g1['동'] = g1['지번주소'].str.extract(r'%s\s+(\S+동|\S+가)' % GU)
    g2 = f2[f2['구'] == GU]
    g3 = f3[f3['구'] == GU]

    fig = plt.figure(figsize=(18, 10), facecolor=BG)
    fig.suptitle(f'{GU} 숙박업 인허가 심층 분석 (2020–2025)',
                 fontsize=18, fontweight='bold', y=0.98, color='#1A1A2E')
    gs = GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.38,
                  left=0.06, right=0.97, top=0.91, bottom=0.09)

    # [A] 누적 꺾은선
    ax = fig.add_subplot(gs[0, :2])
    ax.set_facecolor('white')
    누적_영업, 누적_폐업 = [], []
    for yr in YEARS:
        sub = g1[g1['인허가연도'] <= yr]
        누적_영업.append((sub['영업상태명'] == '영업/정상').sum())
        누적_폐업.append((sub['영업상태명'] == '폐업').sum())
    ax.fill_between(YEARS, 누적_영업, alpha=0.15, color=C_MAIN)
    ax.fill_between(YEARS, 누적_폐업, alpha=0.15, color=C_WARN)
    ax.plot(YEARS, 누적_영업, 'o-', color=C_MAIN, lw=2.8, ms=8, label='영업중 누적', zorder=4)
    ax.plot(YEARS, 누적_폐업, 's-', color=C_WARN, lw=2.8, ms=8, label='폐업 누적',   zorder=4)
    for x, v in zip(YEARS, 누적_영업):
        ax.text(x, v + max(누적_영업)*0.02 + 0.5, str(v), ha='center', fontsize=9, fontweight='bold', color=C_MAIN)
    for x, v in zip(YEARS, 누적_폐업):
        offset = -max(누적_영업)*0.05 - 0.5
        ax.text(x, v + offset, str(v), ha='center', fontsize=8, color=C_WARN)
    ax.set_title('외국인관광도시민박업 누적 현황 (영업중 vs 폐업)', fontsize=11, fontweight='bold', pad=8)
    ax.set_ylabel('누적 업소 수', fontsize=9); ax.set_xticks(YEARS)
    ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.25, linestyle='--')
    ax.spines[['top','right']].set_visible(False)

    # [B] 파이차트 (최신연도)
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_facecolor('white')
    yr_data = g1[g1['인허가연도'].between(2020, 2025)]
    latest_yr = int(yr_data['인허가연도'].max()) if len(yr_data) > 0 else 2025
    dong_cnt = g1[g1['인허가연도'] == latest_yr]['동'].value_counts()

    if len(dong_cnt) == 0:
        ax2.text(0, 0, f'{latest_yr}년\n데이터 없음', ha='center', va='center', fontsize=12, color='#888')
        ax2.set_title(f'{latest_yr}년 신규 인허가\n동별 비중', fontsize=11, fontweight='bold', pad=8)
        ax2.axis('off')
    else:
        TOP_N = 7
        top_d    = dong_cnt.head(TOP_N)
        other_d  = dong_cnt.iloc[TOP_N:]
        other_cnt = other_d.sum()
        labels_pie = list(top_d.index) + (['기타'] if other_cnt > 0 else [])
        sizes_pie  = list(top_d.values)  + ([other_cnt] if other_cnt > 0 else [])
        wedges, texts, autotexts = ax2.pie(
            sizes_pie, labels=labels_pie, autopct='%1.0f%%',
            colors=PAL[:len(labels_pie)], startangle=140,
            pctdistance=0.75, textprops={'fontsize': 8}
        )
        for at in autotexts:
            at.set_fontweight('bold'); at.set_fontsize(8)
        ax2.set_title(f'{latest_yr}년 신규 인허가\n동별 비중', fontsize=11, fontweight='bold', pad=8)
        if other_cnt > 0:
            other_names = ', '.join(other_d.index.tolist())
            ax2.text(0, -1.45, f'기타({int(other_cnt)}건): {other_names}',
                     ha='center', va='top', fontsize=7, color='#555',
                     bbox=dict(boxstyle='round,pad=0.3', fc='#F1F5F9', alpha=0.8))

    # [C] 업종별 연도별 막대
    ax3 = fig.add_subplot(gs[0, 3])
    ax3.set_facecolor('white')
    m1 = g1[g1['인허가연도'].between(2020,2025)].groupby('인허가연도').size().reindex(YEARS, fill_value=0)
    m2 = g2[g2['인허가연도'].between(2020,2025)].groupby('인허가연도').size().reindex(YEARS, fill_value=0)
    m3 = g3[g3['인허가연도'].between(2020,2025)].groupby('인허가연도').size().reindex(YEARS, fill_value=0)
    x = np.arange(len(YEARS)); w = 0.25
    ax3.bar(x-w, m1.values, width=w, color=C_MAIN, alpha=0.85, label='외국인민박')
    ax3.bar(x,   m3.values, width=w, color=C_LIVE, alpha=0.85, label='숙박업')
    ax3.bar(x+w, m2.values, width=w, color=C_WARN, alpha=0.85, label='관광숙박업')
    ax3.set_title('업종별 신규 인허가\n연도 비교', fontsize=11, fontweight='bold', pad=8)
    ax3.set_xticks(x); ax3.set_xticklabels(YEARS, fontsize=7, rotation=30)
    ax3.legend(fontsize=7); ax3.grid(axis='y', alpha=0.25, linestyle='--')
    ax3.spines[['top','right']].set_visible(False)

    # [D] 동별 히트맵
    ax4 = fig.add_subplot(gs[1, :])
    ax4.set_facecolor('white')
    g1_filt = g1[g1['인허가연도'].between(2020, 2025)]
    if len(g1_filt) > 0 and g1_filt['동'].notna().any():
        dong_piv = (g1_filt.groupby(['동','인허가연도']).size()
                    .unstack(fill_value=0)
                    .reindex(columns=YEARS, fill_value=0))
        top_n = min(10, len(dong_piv))
        top_d_idx = dong_piv.sum(axis=1).nlargest(top_n).index
        dong_piv  = dong_piv.loc[top_d_idx]
        im = ax4.imshow(dong_piv.values, aspect='auto', cmap='YlOrRd', vmin=0)
        ax4.set_xticks(range(len(YEARS))); ax4.set_xticklabels(YEARS, fontsize=10)
        ax4.set_yticks(range(len(top_d_idx))); ax4.set_yticklabels(top_d_idx, fontsize=10)
        for i in range(len(top_d_idx)):
            for j in range(len(YEARS)):
                v = int(dong_piv.values[i, j])
                ax4.text(j, i, str(v) if v > 0 else '·', ha='center', va='center', fontsize=11,
                         fontweight='bold',
                         color='white' if v > dong_piv.values.max()*0.55 else '#333')
        plt.colorbar(im, ax=ax4, orientation='vertical', shrink=0.9, label='건수', pad=0.01)
    else:
        ax4.text(0.5, 0.5, '동별 데이터 없음', ha='center', va='center',
                 fontsize=13, color='#888', transform=ax4.transAxes)
        ax4.axis('off')
    ax4.set_title(f'동별 외국인관광도시민박업 신규 인허가 히트맵 (상위 10개 동)', fontsize=11, fontweight='bold', pad=8)
    ax4.spines[['top','right','left','bottom']].set_visible(False)

    out = f'{OUT_DIR}/{GU}_인허가분석.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f'저장: {GU}_인허가분석.png')

print('\n전체 완료!')
