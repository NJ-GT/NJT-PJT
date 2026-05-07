# -*- coding: utf-8 -*-
import pandas as pd, numpy as np, sys
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler
sys.stdout.reconfigure(encoding='utf-8')

import glob, os
matches = glob.glob('c:/Users/USER/Documents/GitHub/*/NJT-PJT/data/*0423*.csv')
fpath = [m for m in matches if '분석변수' in m or '분析변수' in m][0]
df = pd.read_csv(fpath, encoding='utf-8-sig')

# 업종 더미 (관광숙박업 기준)
df2 = pd.get_dummies(df, columns=['업종'], drop_first=False, dtype=int)
df2 = df2.rename(columns={
    '업종_숙박업':           '숙박업',
    '업종_외국인관광도시민박업': '외국인민박',
    '업종_관광숙박업':        '관광숙박업'
})

risk_vars = ['구조노후도','단속위험도','도로폭위험도','집중도','주변건물수']
sc = StandardScaler()
df2['위험지수'] = sc.fit_transform(df2[risk_vars]).mean(axis=1)

formula = '위험지수 ~ 구조노후도 + 단속위험도 + 도로폭위험도 + 집중도 + 주변건물수 + 숙박업 + 외국인민박'
model = smf.ols(formula, data=df2).fit(cov_type='HC3')

print(f'R² = {model.rsquared:.3f}  N = {int(model.nobs):,}')
print()
print('=== 업종 효과 (기준: 관광숙박업) ===')
for v in ['숙박업', '외국인민박']:
    c    = model.params[v]
    p    = model.pvalues[v]
    ci_lo = model.conf_int().loc[v, 0]
    ci_hi = model.conf_int().loc[v, 1]
    sig  = '***' if p<0.001 else ('**' if p<0.01 else ('*' if p<0.05 else 'n.s.'))
    print(f'  {v:<12}  coef={c:+.4f}  95%CI=[{ci_lo:+.4f},{ci_hi:+.4f}]  p={p:.4f}  {sig}')
print()
print('=== 다른 변수 계수 ===')
for v in risk_vars:
    c = model.params[v]; p = model.pvalues[v]
    sig = '***' if p<0.001 else ('**' if p<0.01 else ('*' if p<0.05 else 'n.s.'))
    print(f'  {v:<12}  coef={c:+.4f}  p={p:.4f}  {sig}')
