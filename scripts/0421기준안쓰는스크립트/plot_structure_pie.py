# -*- coding: utf-8 -*-
import pandas as pd, numpy as np, sys, re
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
sys.stdout.reconfigure(encoding='utf-8')

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

BASE = 'c:/Users/USER/Documents/GitHub/기말공모전/NJT-PJT'
df = pd.read_csv(f'{BASE}/data/통합숙박시설최종안0415.csv',
                 encoding='utf-8-sig', on_bad_lines='skip')

def classify(s):
    """기타구조 문자열 → 7개 화재취약성 카테고리 (복합재는 가장 취약한 소재로 귀속)"""
    if pd.isna(s):
        return '기타/미상'
    s = str(s).replace(' ', '').upper()
    # 취약도 높은 순으로 우선 검사
    if re.search(r'목조|목구조|WOOD', s):
        return '목조'
    if re.search(r'샌드위치|판넬|패널', s):
        return '샌드위치판넬'
    if re.search(r'경량철골|가스라이팅', s):
        return '경량철골'
    if re.search(r'연와|벽돌|조적|시멘트벽돌|세멘벽돌|세멘부럭|세멘부록|세멘연와|시멘트연와|BRICK', s):
        return '조적/연와/벽돌'
    if re.search(r'SRC|철골철근콘크리트|STEELREINFORCED', s):
        return 'SRC(철골철근콘크리트)'
    if re.search(r'일반철골|철골구조|스틸', s):
        return '일반철골'
    if re.search(r'철근콘크리트|R\.C|RC조|라멘|벽식|콘크리트', s):
        return '철근콘크리트(RC)'
    return '기타/미상'

df['구조카테고리'] = df['기타구조'].apply(classify)
counts = df['구조카테고리'].value_counts()
print(counts.to_string())
print(f'\n전체: {counts.sum()}개')

# ── 파이그래프 ──────────────────────────────────────────────────────────
# 화재취약성 순서 (취약 → 안전)
ORDER = ['목조', '샌드위치판넬', '경량철골', '조적/연와/벽돌',
         '일반철골', '철근콘크리트(RC)', 'SRC(철골철근콘크리트)', '기타/미상']
labels_ordered = [c for c in ORDER if c in counts.index]
sizes = [counts[c] for c in labels_ordered]

# 색상: 빨강계(취약) → 초록계(안전) → 회색(기타)
COLORS = {
    '목조':              '#c0392b',
    '샌드위치판넬':       '#e74c3c',
    '경량철골':           '#e67e22',
    '조적/연와/벽돌':     '#f39c12',
    '일반철골':           '#2ecc71',
    '철근콘크리트(RC)':   '#27ae60',
    'SRC(철골철근콘크리트)': '#1abc9c',
    '기타/미상':          '#bdc3c7',
}
colors = [COLORS[c] for c in labels_ordered]

fig, ax = plt.subplots(figsize=(9, 7))
wedges, texts, autotexts = ax.pie(
    sizes,
    labels=labels_ordered,
    colors=colors,
    autopct=lambda p: f'{p:.1f}%\n({int(round(p*sum(sizes)/100)):,}개)',
    startangle=140,
    pctdistance=0.75,
    wedgeprops=dict(linewidth=0.8, edgecolor='white')
)
for t in texts:
    t.set_fontsize(10)
for at in autotexts:
    at.set_fontsize(8.5)

ax.set_title('숙박시설 건물구조 분포\n(화재취약성 순 정렬)', fontsize=14, pad=16)

# 범례: 취약↑ → 안전↓
legend_labels = [f'{c}  ({counts.get(c,0):,}개, {counts.get(c,0)/counts.sum()*100:.1f}%)'
                 for c in labels_ordered]
ax.legend(wedges, legend_labels, title='구조 카테고리',
          loc='lower left', bbox_to_anchor=(-0.15, -0.05),
          fontsize=9, title_fontsize=10)

plt.tight_layout()
out = f'{BASE}/data/structure_pie.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f'\n[저장] {out}')
