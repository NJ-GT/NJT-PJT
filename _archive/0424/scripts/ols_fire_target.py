# -*- coding: utf-8 -*-
"""OLS: log(1+반경500m 숙박화재수) ~ 입지·구조 변수 + 구 고정효과"""
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
import sys

sys.stdout.reconfigure(encoding='utf-8')

BASE = 'c:/Users/USER/Documents/GitHub/기말공모전/NJT-PJT/data'
df = pd.read_csv(f'{BASE}/data_with_fire_targets.csv', encoding='utf-8-sig')
print(f'로드: {len(df)}행')

# ── 1. 변수 준비 ──────────────────────────────────────────────────────
df['건물나이']   = 2026 - df['승인연도']
df['주변건물수'] = df['반경_50m_건물수']
df['집중도']     = df['집중도(%)']
df['단속위험도'] = df['로그_주변대비_상대위험도_고유단속지점_50m'].fillna(
                       df['로그_주변대비_상대위험도_고유단속지점_50m'].median())
df['구조노후도'] = df['구조_노후_통합점수']
df['도로폭위험도'] = df['도로폭_위험도']
df['Y'] = df['log1p_반경500m']

# 구 더미 (drop_first=True → 강남구 기준)
df = pd.get_dummies(df, columns=['구'], drop_first=True, dtype=int)
gu_cols = [c for c in df.columns if c.startswith('구_')]

X_cols = ['건물나이', '주변건물수', '집중도', '단속위험도', '구조노후도', '도로폭위험도',
          '위도', '경도'] + gu_cols

df_clean = df[['Y'] + X_cols].dropna()
print(f'분석 행수: {len(df_clean)} (결측 제거 후)')

# ── 2. OLS ────────────────────────────────────────────────────────────
formula = 'Y ~ ' + ' + '.join(X_cols)
model = smf.ols(formula, data=df_clean).fit(cov_type='HC3')   # robust SE

print('\n' + '='*60)
print('OLS 결과 — Y = log(1+반경500m 숙박화재수)')
print('='*60)
print(f'  N = {int(model.nobs):,}   R² = {model.rsquared:.3f}   Adj.R² = {model.rsquared_adj:.3f}')
print(f'  F-stat = {model.fvalue:.2f}   p = {model.f_pvalue:.4f}')

print('\n[핵심 변수 계수]')
key_vars = ['건물나이', '주변건물수', '집중도', '단속위험도', '구조노후도', '도로폭위험도']
for v in key_vars:
    coef = model.params[v]
    pval = model.pvalues[v]
    sig  = '***' if pval < 0.001 else ('**' if pval < 0.01 else ('*' if pval < 0.05 else ''))
    print(f'  {v:<12}  coef={coef:+.4f}  p={pval:.4f}  {sig}')

# ── 3. VIF ────────────────────────────────────────────────────────────
print('\n[VIF — 다중공선성]')
X_mat = df_clean[X_cols].assign(const=1)
for i, col in enumerate(X_cols):
    vif = variance_inflation_factor(X_mat.values, i)
    flag = ' ← 주의' if vif > 10 else ''
    print(f'  {col:<14}  VIF={vif:.2f}{flag}')

# ── 4. 이분산 검정 ────────────────────────────────────────────────────
bp_stat, bp_p, _, _ = het_breuschpagan(model.resid, model.model.exog)
print(f'\n[Breusch-Pagan] stat={bp_stat:.2f}  p={bp_p:.4f}',
      '→ 이분산 있음 (HC3 SE 적용 중)' if bp_p < 0.05 else '→ 등분산')

# ── 5. 잔차 저장 ──────────────────────────────────────────────────────
df_clean = df_clean.copy()
df_clean['잔차'] = model.resid
df_clean['예측값'] = model.fittedvalues
df_clean.to_csv(f'{BASE}/ols_residuals.csv', index=False, encoding='utf-8-sig')
print(f'\n[저장] ols_residuals.csv (잔차·예측값 포함)')
print('\n전체 회귀 요약:')
print(model.summary().tables[1])
