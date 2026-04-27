# -*- coding: utf-8 -*-
"""6변수 기준 Spatial Lag / Spatial Error 계수 추출"""
import sys, glob, warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from libpysal.weights import KNN
from esda.moran import Moran

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
df = df.dropna(subset=VARS6 + ['위험점수_AHP', '반경100m_화재수', '위도', '경도']).reset_index(drop=True)

# Y = log(화재수+1) — 공간분석 체인용
df['Y'] = np.log1p(df['반경100m_화재수'])
N = len(df)
print(f"샘플: {N:,}")

coords = df[['위도', '경도']].values
W = KNN.from_array(coords, k=8)
W.transform = 'r'

Xs = StandardScaler().fit_transform(df[VARS6].values)
Y  = df['Y'].values

# OLS 잔차
Xc = np.column_stack([np.ones(N), Xs])
b, *_ = np.linalg.lstsq(Xc, Y, rcond=None)
Yhat  = Xc @ b
resid = Y - Yhat
r2_ols = 1 - ((resid**2).sum() / ((Y - Y.mean())**2).sum())
mi_res = Moran(resid, W, permutations=499)
print(f"\nOLS  R²={r2_ols:.4f}  잔차 Moran's I={mi_res.I:.4f} p={mi_res.p_sim:.4f}")

# ── Spatial Lag (OLS + lagged Y) ──
lag_Y = W.sparse.dot(Y)
Xc2   = np.column_stack([np.ones(N), Xs, lag_Y])
b2, *_ = np.linalg.lstsq(Xc2, Y, rcond=None)
Yhat2  = Xc2 @ b2
r2_slm = 1 - ((Y - Yhat2)**2).sum() / ((Y - Y.mean())**2).sum()
rho    = b2[-1]
print(f"SLM  R²={r2_slm:.4f}  ρ(lag_Y)={rho:.4f}")

# ── Spatial Error (spreg GM_Error) ──
try:
    from spreg import GM_Error
    gm = GM_Error(Y.reshape(-1,1), Xs, w=W, name_y='Y', name_x=VARS6)
    lam_row = gm.output[gm.output['var_names']=='lambda']
    lam_val = lam_row['coefficients'].values[0]
    print(f"GM_Error  λ={lam_val:.4f}")
except Exception as e:
    print(f"GM_Error 오류: {e}")

# ── Y=AHP 기준도 추가 ──
print("\n--- AHP 기준 (GWR page와 동일) ---")
Ya = df['위험점수_AHP'].values
Xca = np.column_stack([np.ones(N), Xs])
ba, *_ = np.linalg.lstsq(Xca, Ya, rcond=None)
resid_a = Ya - Xca @ ba
r2_a = 1 - ((resid_a**2).sum() / ((Ya - Ya.mean())**2).sum())
mi_a = Moran(resid_a, W, permutations=499)
lag_Ya = W.sparse.dot(Ya)
Xca2 = np.column_stack([np.ones(N), Xs, lag_Ya])
ba2, *_ = np.linalg.lstsq(Xca2, Ya, rcond=None)
r2_slm_a = 1 - ((Ya - Xca2@ba2)**2).sum() / ((Ya - Ya.mean())**2).sum()
print(f"AHP OLS  R²={r2_a:.4f}  잔차 Moran's I={mi_a.I:.4f} p={mi_a.p_sim:.4f}")
print(f"AHP SLM  R²={r2_slm_a:.4f}  ρ={ba2[-1]:.4f}")
