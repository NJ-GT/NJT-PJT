# -*- coding: utf-8 -*-
"""
OLS 개선 버전
  Step 1: 변수 추가 (소방접근성, 상업비율, 시설군 더미)
  Step 2: 비선형 항 (건물나이²)
  Step 3: Moran's I → Spatial Lag Model
"""
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
import libpysal.weights as lw
import esda
import spreg
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import sys

sys.stdout.reconfigure(encoding='utf-8')

for font in fm.findSystemFonts():
    if 'malgun' in font.lower():
        plt.rcParams['font.family'] = fm.FontProperties(fname=font).get_name()
        break
plt.rcParams['axes.unicode_minus'] = False

BASE = 'c:/Users/USER/Documents/GitHub/기말공모전/NJT-PJT/data'

# ── 데이터 준비 ───────────────────────────────────────────────────────
lodging = pd.read_csv(f'{BASE}/핵심서울0424.csv', encoding='utf-8-sig')
fire_raw = pd.read_csv(f'{BASE}/화재출동/화재출동_2021_2024.csv', encoding='utf-8-sig', low_memory=False)

LODGING_TYPES = ['호텔','모텔','여관','여인숙','기타 숙박시설','숙박공유업']
m1 = fire_raw['발화장소_소분류'].str.strip().isin(LODGING_TYPES)
fire = fire_raw[m1 & fire_raw['위도'].notna() & fire_raw['경도'].notna()].copy()

def hav(lat1, lon1, lat2, lon2):
    R = 6371000
    lat1 = np.radians(lat1)[:,None]; lon1 = np.radians(lon1)[:,None]
    lat2 = np.radians(lat2)[None,:]; lon2 = np.radians(lon2)[None,:]
    a = np.sin((lat2-lat1)/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin((lon2-lon1)/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

dist = hav(lodging['위도'].values, lodging['경도'].values,
           fire['위도'].values, fire['경도'].values)
lodging['반경500m_화재수'] = (dist <= 500).sum(axis=1)
lodging['Y'] = np.log1p(lodging['반경500m_화재수'])

# ── Step 1+2: 변수 구성 ───────────────────────────────────────────────
lodging['건물나이']    = 2026 - lodging['승인연도']
lodging['건물나이2']   = lodging['건물나이'] ** 2          # 비선형항
lodging['주변건물수']  = lodging['반경_50m_건물수']
lodging['집중도']      = lodging['집중도(%)']
lodging['단속위험도']  = lodging['로그_주변대비_상대위험도_고유단속지점_50m'].fillna(
                            lodging['로그_주변대비_상대위험도_고유단속지점_50m'].median())
lodging['구조노후도']  = lodging['구조_노후_통합점수']
lodging['도로폭위험도'] = lodging['도로폭_위험도']
lodging['소방거리']    = lodging['최근접_거리m']
lodging['상업비율']    = lodging['상업비율(%)'].fillna(lodging['상업비율(%)'].median())

df = pd.get_dummies(lodging, columns=['구', '주요_시설군'], drop_first=True, dtype=int)
gu_cols  = [c for c in df.columns if c.startswith('구_')]
sil_cols = [c for c in df.columns if c.startswith('주요_시설군_')]

X_base = ['건물나이','주변건물수','집중도','단속위험도','구조노후도','도로폭위험도','위도','경도']
X_new  = ['건물나이2','소방거리','상업비율']
X_all  = X_base + X_new + sil_cols + gu_cols

df_clean = df[['Y','위도','경도'] + X_all].dropna()
print(f'분석 행수: {len(df_clean)}')

# ── OLS 기본 vs 개선 비교 ─────────────────────────────────────────────
m_base = smf.ols('Y ~ ' + ' + '.join(X_base + gu_cols), data=df_clean).fit(cov_type='HC3')
m_impr = smf.ols('Y ~ ' + ' + '.join(X_all),            data=df_clean).fit(cov_type='HC3')

print(f'\n[기본 OLS]    R²={m_base.rsquared:.3f}  Adj.R²={m_base.rsquared_adj:.3f}')
print(f'[개선 OLS]    R²={m_impr.rsquared:.3f}  Adj.R²={m_impr.rsquared_adj:.3f}')

print('\n[개선 모델 핵심 계수]')
key_vars = ['건물나이','건물나이2','주변건물수','집중도','단속위험도',
            '구조노후도','도로폭위험도','소방거리','상업비율']
for v in key_vars:
    c = m_impr.params[v]; p = m_impr.pvalues[v]
    sig = '***' if p<0.001 else ('**' if p<0.01 else ('*' if p<0.05 else 'n.s.'))
    print(f'  {v:<14} coef={c:+.6f}  p={p:.4f}  {sig}')

# ── Step 3: Moran's I ─────────────────────────────────────────────────
print('\n=== Moran\'s I (공간 자기상관 검정) ===')
coords = np.column_stack([df_clean['경도'].values, df_clean['위도'].values])
W = lw.KNN.from_array(coords, k=8)
W.transform = 'r'

mi = esda.Moran(df_clean['Y'].values, W)
print(f'  Moran\'s I = {mi.I:.4f}  p = {mi.p_sim:.4f}',
      '→ 공간 자기상관 유의' if mi.p_sim < 0.05 else '→ 공간 자기상관 없음')

# ── Spatial Lag Model ─────────────────────────────────────────────────
print('\n=== Spatial Lag Model (SLM) ===')
y_arr = df_clean['Y'].values.reshape(-1, 1)
X_arr = df_clean[X_all].values

# WY: 이웃 Y의 가중평균을 X로 직접 추가 (Spatial Lag OLS)
Wfull = W.full()[0]
wy = Wfull @ df_clean['Y'].values
df_clean = df_clean.copy()
df_clean['WY'] = wy

m_slag = smf.ols('Y ~ WY + ' + ' + '.join(X_all), data=df_clean).fit(cov_type='HC3')
pseudo_r2 = m_slag.rsquared
rho = m_slag.params['WY']
rho_p = m_slag.pvalues['WY']
print(f'  Pseudo R² ≈ {pseudo_r2:.3f}')
print(f'  Rho (공간래그계수) = {rho:.4f}  (p={rho_p:.4f})')

# ── 시각화 ────────────────────────────────────────────────────────────
key_vars_plot = ['건물나이','건물나이2','집중도','단속위험도',
                 '구조노후도','도로폭위험도','소방거리','상업비율']
coefs  = [m_impr.params[v] for v in key_vars_plot]
ci_lo  = [m_impr.conf_int().loc[v,0] for v in key_vars_plot]
ci_hi  = [m_impr.conf_int().loc[v,1] for v in key_vars_plot]
pvals  = [m_impr.pvalues[v] for v in key_vars_plot]
colors = ['#e74c3c' if c>0 else '#3498db' for c in coefs]
alphas = [1.0 if p<0.05 else 0.3 for p in pvals]

fig, axes = plt.subplots(1, 3, figsize=(17, 6))
fig.suptitle(
    f'개선 OLS — Y=log(1+반경500m 숙박화재수)\n'
    f'기본 R²={m_base.rsquared:.3f} → 개선 R²={m_impr.rsquared:.3f} '
    f'/ Moran\'s I={mi.I:.3f}(p={mi.p_sim:.3f}) / SLM Pseudo R²≈{pseudo_r2:.3f}',
    fontsize=11, fontweight='bold'
)

ax = axes[0]
for i,(v,c,lo,hi,p,col,al) in enumerate(zip(key_vars_plot,coefs,ci_lo,ci_hi,pvals,colors,alphas)):
    ax.barh(i, c, color=col, alpha=al, height=0.55)
    ax.plot([lo,hi],[i,i], color='black', lw=1.5)
    sig = '***' if p<0.001 else ('**' if p<0.01 else ('*' if p<0.05 else 'n.s.'))
    x_txt = hi+abs(c)*0.05 if c>=0 else lo-abs(c)*0.05
    ax.text(x_txt, i, sig, va='center', fontsize=9, ha='left' if c>=0 else 'right')
ax.axvline(0, color='black', lw=0.8, linestyle='--')
ax.set_yticks(range(len(key_vars_plot)))
ax.set_yticklabels(key_vars_plot, fontsize=9)
ax.set_title('계수 플롯 (HC3 95% CI)\n빨강=양(+) 파랑=음(−) 투명=n.s.')

resid = m_impr.resid
mi_resid = esda.Moran(resid.values, W)
axes[1].scatter(m_impr.fittedvalues, resid, alpha=0.15, s=5, color='steelblue')
axes[1].axhline(0, color='red', lw=1)
axes[1].set_xlabel('예측값'); axes[1].set_ylabel('잔차')
axes[1].set_title(f'잔차 vs 예측값\n잔차 Moran\'s I={mi_resid.I:.3f} p={mi_resid.p_sim:.3f}')

r2_compare = {'기본\nOLS': m_base.rsquared,
              '개선\nOLS': m_impr.rsquared,
              '공간래그\nOLS': pseudo_r2}
bars = axes[2].bar(r2_compare.keys(), r2_compare.values(),
                   color=['#95a5a6','#e67e22','#2ecc71'], alpha=0.85, width=0.5)
for bar, val in zip(bars, r2_compare.values()):
    axes[2].text(bar.get_x()+bar.get_width()/2, val+0.005, f'{val:.3f}',
                 ha='center', va='bottom', fontweight='bold', fontsize=12)
axes[2].set_ylim(0, max(r2_compare.values())*1.2)
axes[2].set_ylabel('R²')
axes[2].set_title('모델별 설명력 비교')

plt.tight_layout()
plt.savefig(f'{BASE}/ols_improved_result.png', dpi=150, bbox_inches='tight')
print(f'\n[저장] ols_improved_result.png')
