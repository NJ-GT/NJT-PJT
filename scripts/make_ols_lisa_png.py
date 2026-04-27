# -*- coding: utf-8 -*-
"""
최신 6변수 기준 OLS 결과 + LISA 지도 PNG 생성
  변수: 구조노후도, 단속위험도, 도로폭위험도, 집중도, 주변건물수, 소방위험도_점수
  Y  : 위험점수_AHP
"""
import sys, glob, warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rc
rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

# ── 데이터 로드 ──────────────────────────────────────
f    = glob.glob('data/*0423*.csv')[0]
main = pd.read_csv(f, encoding='utf-8-sig')
core = pd.read_csv('data/data_with_fire_targets.csv', encoding='utf-8-sig')
c    = core.columns
core_key = core[[c[4], c[5], c[17], c[22], c[45]]].copy()
core_key.columns = ['위도', '경도', '소방위험도_점수', '위험점수_AHP', '반경100m_화재수']

df = pd.merge(main, core_key, on=['위도', '경도'], how='left')

VARS6 = ['구조노후도', '단속위험도', '도로폭위험도', '집중도', '주변건물수', '소방위험도_점수']
for v in VARS6 + ['위험점수_AHP', '반경100m_화재수']:
    df[v] = pd.to_numeric(df[v], errors='coerce')

df = df.dropna(subset=VARS6 + ['반경100m_화재수', '위도', '경도']).reset_index(drop=True)
N  = len(df)
print(f"샘플: {N:,}개")

# ── OLS ──────────────────────────────────────────────
from sklearn.preprocessing import StandardScaler

X_raw = df[VARS6].values
Xs    = StandardScaler().fit_transform(X_raw)

# Y = log(화재수+1) — 실제 발생 기록 기준 (체인 전체 일관성)
Y = np.log1p(df['반경100m_화재수'].fillna(0)).values

from numpy.linalg import lstsq, inv
Xb   = np.column_stack([np.ones(N), Xs])
coef, _, _, _ = lstsq(Xb, Y, rcond=None)
Yhat = Xb @ coef
resid = Y - Yhat
ss_res = (resid**2).sum()
ss_tot = ((Y - Y.mean())**2).sum()
r2 = 1 - ss_res / ss_tot

# HC3 표준오차
hat  = Xb @ inv(Xb.T @ Xb) @ Xb.T
hii  = np.diag(hat)
e_sq = (resid / (1 - hii))**2
meat = (Xb.T * e_sq) @ Xb
vcov = inv(Xb.T @ Xb) @ meat @ inv(Xb.T @ Xb)
se   = np.sqrt(np.diag(vcov))
tval = coef / se
ci95_lo = coef - 1.96 * se
ci95_hi = coef + 1.96 * se

# 계수 (절편 제외)
coefs  = coef[1:]
lo95   = ci95_lo[1:]
hi95   = ci95_hi[1:]
tvals  = tval[1:]

def sig_star(t):
    a = abs(t)
    if a > 3.29: return '***'
    if a > 2.58: return '**'
    if a > 1.96: return '*'
    return 'n.s.'

# ── 잔차 Moran's I ───────────────────────────────────
from libpysal.weights import KNN
from esda.moran import Moran, Moran_Local

coords = df[['위도', '경도']].values
W = KNN.from_array(coords, k=8)
W.transform = 'r'

mi_resid = Moran(resid, W)
print(f"잔차 Moran's I = {mi_resid.I:.4f}  p={mi_resid.p_sim:.4f}")

# ── 화재수 Y 자체 Moran's I + LISA ───────────────────
mi_y = Moran(Y, W)
print(f"Y(화재수) Moran's I = {mi_y.I:.4f}  p={mi_y.p_sim:.4f}")

lisa = Moran_Local(Y, W, seed=42)
sig  = lisa.p_sim < 0.05
cats = np.full(N, 'Not Sig', dtype=object)
cats[(lisa.q == 1) & sig] = 'HH'
cats[(lisa.q == 3) & sig] = 'LL'
cats[(lisa.q == 2) & sig] = 'LH'
cats[(lisa.q == 4) & sig] = 'HL'

# ── Spatial Lag R² (근사) ────────────────────────────
lag_Y = W.sparse.dot(Y)
Xb2   = np.column_stack([np.ones(N), Xs, lag_Y])
coef2, _, _, _ = lstsq(Xb2, Y, rcond=None)
Yhat2 = Xb2 @ coef2
r2_slm = 1 - ((Y - Yhat2)**2).sum() / ss_tot

print(f"OLS R²={r2:.3f}  SLM(근사) R²={r2_slm:.3f}")

# ════════════════════════════════════════════════════
# PNG 1 — OLS 결과 (계수 + 잔차 + 모델 비교)
# ════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle(
    f'OLS 결과 — Y=log(반경100m 화재수+1)  (6변수 최종)\n'
    f'N={N:,}  R²={r2:.3f}  잔차 Moran\'s I={mi_resid.I:.3f} p<0.001',
    fontsize=13, fontweight='bold'
)

# 패널1: 계수 플롯
ax = axes[0]
colors = ['#E53E3E' if c > 0 else '#3182CE' for c in coefs]
y_pos  = np.arange(len(VARS6))
ax.barh(y_pos, coefs, color=colors, alpha=0.75, height=0.5)
ax.errorbar(coefs, y_pos, xerr=[coefs - lo95, hi95 - coefs],
            fmt='none', color='black', capsize=4, linewidth=1.5)
for i, (c_, t) in enumerate(zip(coefs, tvals)):
    ax.text(hi95[i] + 0.01 * (hi95.max() - lo95.min()),
            i, sig_star(t), va='center', fontsize=10)
ax.set_yticks(y_pos)
ax.set_yticklabels(VARS6, fontsize=10)
ax.axvline(0, color='black', linestyle='--', linewidth=0.8)
ax.set_title('계수 플롯 (95% CI, HC3)', fontsize=11)
ax.set_xlabel('계수값 (표준화)')

# 패널2: 잔차 vs 예측값
ax2 = axes[1]
ax2.scatter(Yhat, resid, alpha=0.15, s=8, color='#4A90D9')
ax2.axhline(0, color='red', linewidth=1.2)
ax2.set_xlabel('예측값')
ax2.set_ylabel('잔차')
ax2.set_title(f"잔차 Moran's I={mi_resid.I:.3f}  p<0.001", fontsize=11)

# 패널3: 모델 비교 바
ax3 = axes[2]
models = ['OLS', 'SLM\n(근사)']
r2s    = [r2, r2_slm]
bar_colors = ['#A0AEC0', '#38A169']
bars = ax3.bar(models, r2s, color=bar_colors, width=0.4)
for bar, v in zip(bars, r2s):
    ax3.text(bar.get_x() + bar.get_width()/2, v + 0.01,
             f'{v:.3f}', ha='center', fontsize=11, fontweight='bold')
ax3.set_ylim(0, 1.05)
ax3.set_ylabel('R²')
ax3.set_title('모델 설명력 비교', fontsize=11)

plt.tight_layout()
plt.savefig('data/ols_6var_result.png', dpi=150, bbox_inches='tight')
plt.close()
print("저장: data/ols_6var_result.png")

# ════════════════════════════════════════════════════
# PNG 2 — LISA 지도
# ════════════════════════════════════════════════════
df['lisa_cat'] = cats
df['lat']      = df['위도']
df['lon']      = df['경도']

cat_color = {'HH': '#D63031', 'LL': '#0984E3', 'LH': '#74B9FF', 'HL': '#FD79A8', 'Not Sig': '#DFE6E9'}
cat_order  = ['HH', 'LL', 'LH', 'HL', 'Not Sig']
cat_label  = {'HH':'HH 핫스팟', 'LL':'LL 저위험', 'LH':'LH (저-고)', 'HL':'HL (고-저)', 'Not Sig':'비유의'}

counts = df['lisa_cat'].value_counts()

fig2, axes2 = plt.subplots(1, 2, figsize=(16, 7))
fig2.suptitle(
    f"LISA 분석 — 숙박시설 화재수 공간 군집\n(Global Moran's I = {mi_y.I:.4f}, p < 0.01  |  Y=log(화재수+1))",
    fontsize=14, fontweight='bold'
)

# 패널1: 지도
ax_m = axes2[0]
for cat in cat_order[::-1]:
    sub = df[df['lisa_cat'] == cat]
    ax_m.scatter(sub['lon'], sub['lat'],
                 c=cat_color[cat], s=5, alpha=0.7,
                 label=f"{cat_label[cat]} ({counts.get(cat,0):,})", zorder=3)
ax_m.set_xlabel('경도', fontsize=10)
ax_m.set_ylabel('위도', fontsize=10)
ax_m.set_title('LISA 군집 지도', fontsize=12)
ax_m.set_facecolor('#F0F4F8')
ax_m.legend(loc='lower left', fontsize=9, framealpha=0.9)

# 패널2: 구별 HH 비율
gu_hh = df[df['lisa_cat']=='HH'].groupby('구').size()
gu_total = df.groupby('구').size()
gu_ratio = (gu_hh / gu_total * 100).dropna().sort_values(ascending=True)

bar_vals = gu_ratio.values
bar_labs = gu_ratio.index.tolist()
bar_cols  = ['#E53E3E' if v >= 30 else '#FC8181' if v >= 15 else '#FEB2B2' for v in bar_vals]
ax_g = axes2[1]
ax_g.barh(bar_labs, bar_vals, color=bar_cols, height=0.6)
ax_g.axvline(30, color='gray', linestyle='--', linewidth=1, alpha=0.7)
for i, v in enumerate(bar_vals):
    ax_g.text(v + 0.5, i, f'{v:.1f}%', va='center', fontsize=10)
ax_g.set_xlabel('핫스팟(HH) 비율 (%)', fontsize=10)
ax_g.set_title('구별 핫스팟 비율', fontsize=12)

plt.tight_layout()
plt.savefig('data/lisa_6var_map.png', dpi=150, bbox_inches='tight')
plt.close()
print("저장: data/lisa_6var_map.png")
