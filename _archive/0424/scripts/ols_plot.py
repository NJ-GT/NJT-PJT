# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import statsmodels.formula.api as smf
import sys

sys.stdout.reconfigure(encoding='utf-8')

# 한글 폰트
for font in fm.findSystemFonts():
    if 'malgun' in font.lower() or 'NanumGothic' in font.lower():
        plt.rcParams['font.family'] = fm.FontProperties(fname=font).get_name()
        break
plt.rcParams['axes.unicode_minus'] = False

BASE = 'c:/Users/USER/Documents/GitHub/기말공모전/NJT-PJT/data'
df = pd.read_csv(f'{BASE}/data_with_fire_targets.csv', encoding='utf-8-sig')

df['건물나이']    = 2026 - df['승인연도']
df['주변건물수']  = df['반경_50m_건물수']
df['집중도']      = df['집중도(%)']
df['단속위험도']  = df['로그_주변대비_상대위험도_고유단속지점_50m'].fillna(
                        df['로그_주변대비_상대위험도_고유단속지점_50m'].median())
df['구조노후도']  = df['구조_노후_통합점수']
df['도로폭위험도'] = df['도로폭_위험도']
df['Y'] = df['log1p_반경500m']

df = pd.get_dummies(df, columns=['구'], drop_first=True, dtype=int)
gu_cols = [c for c in df.columns if c.startswith('구_')]
X_cols = ['건물나이', '주변건물수', '집중도', '단속위험도', '구조노후도', '도로폭위험도',
          '위도', '경도'] + gu_cols

df_clean = df[['Y'] + X_cols].dropna()
model = smf.ols('Y ~ ' + ' + '.join(X_cols), data=df_clean).fit(cov_type='HC3')

key_vars  = ['건물나이', '주변건물수', '집중도', '단속위험도', '구조노후도', '도로폭위험도']
coefs     = [model.params[v] for v in key_vars]
ci_low    = [model.conf_int().loc[v, 0] for v in key_vars]
ci_high   = [model.conf_int().loc[v, 1] for v in key_vars]
pvals     = [model.pvalues[v] for v in key_vars]
colors    = ['#e74c3c' if c > 0 else '#3498db' for c in coefs]
alphas    = [1.0 if p < 0.05 else 0.35 for p in pvals]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle(f'OLS 결과 — Y = log(1+반경500m 숙박화재수)\nN={int(model.nobs):,}  R²={model.rsquared:.3f}  Adj.R²={model.rsquared_adj:.3f}',
             fontsize=13, fontweight='bold', y=1.01)

# ── 그래프1: 계수 플롯 ────────────────────────────────────────────────
ax = axes[0]
y_pos = range(len(key_vars))
for i, (v, c, lo, hi, p, col) in enumerate(zip(key_vars, coefs, ci_low, ci_high, pvals, colors)):
    ax.barh(i, c, color=col, alpha=alphas[i], height=0.5)
    ax.plot([lo, hi], [i, i], color='black', lw=1.5)
    sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'n.s.'))
    ax.text(max(hi, c) + 0.005, i, sig, va='center', fontsize=10)
ax.axvline(0, color='black', lw=0.8, linestyle='--')
ax.set_yticks(list(y_pos))
ax.set_yticklabels(key_vars, fontsize=10)
ax.set_xlabel('회귀계수 (95% CI, HC3)')
ax.set_title('핵심 변수 계수\n빨강=양(+) 파랑=음(−) 투명=비유의')

# ── 그래프2: 잔차 vs 예측값 ───────────────────────────────────────────
ax = axes[1]
ax.scatter(model.fittedvalues, model.resid, alpha=0.15, s=5, color='steelblue')
ax.axhline(0, color='red', lw=1)
ax.set_xlabel('예측값')
ax.set_ylabel('잔차')
ax.set_title('잔차 vs 예측값\n(이분산 확인)')

# ── 그래프3: 실제 vs 예측 ─────────────────────────────────────────────
ax = axes[2]
ax.scatter(df_clean['Y'], model.fittedvalues, alpha=0.15, s=5, color='darkorange')
mn, mx = df_clean['Y'].min(), df_clean['Y'].max()
ax.plot([mn, mx], [mn, mx], 'r--', lw=1)
ax.set_xlabel('실제값 (Y)')
ax.set_ylabel('예측값')
ax.set_title(f'실제 vs 예측\nR²={model.rsquared:.3f}')

plt.tight_layout()
out = f'{BASE}/ols_result.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'[저장] {out}')
plt.show()
