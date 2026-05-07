# -*- coding: utf-8 -*-
"""
튜닝 파이프라인 v2
  변경사항:
    1. Y 변수  : AHP 위험점수 → 반경100m_화재수 (log1p 변환, 실제 발생 기반)
    2. Moran's I 이웃 k: 4 / 8 / 12 / 16 민감도 비교
    3. GWR 커널 : bisquare + gaussian 비교
    4. Spatial Error: ML_Error 외에 GM_Error (GMM 기반, 더 강건) 추가
"""
import sys, glob, warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LassoCV

from libpysal.weights import KNN
from esda.moran import Moran
from spreg import ML_Lag, ML_Error, GM_Error
from mgwr.gwr   import GWR
from mgwr.sel_bw import Sel_BW

# ──────────────────────────────────────────────
# 데이터 로드 & 병합
# ──────────────────────────────────────────────
f    = glob.glob('data/*0423*.csv')[0]
main = pd.read_csv(f, encoding='utf-8-sig')
core = pd.read_csv('data/data_with_fire_targets.csv', encoding='utf-8-sig')

df = pd.merge(main, core[['위도','경도','위험점수_AHP','반경100m_화재수','이동시간초']],
              on=['위도','경도'], how='left')
df = df.dropna(subset=['반경100m_화재수','구조노후도','단속위험도',
                        '도로폭위험도','집중도','주변건물수']).reset_index(drop=True)

RISK_VARS = ['구조노후도','단속위험도','도로폭위험도','집중도','주변건물수']
GEO_VARS  = ['위도','경도']

# Y 변수: log1p(반경100m_화재수) — 카운트 데이터 스큐 보정
df['Y'] = np.log1p(df['반경100m_화재수'])

print(f"로드 완료: {len(df):,}개 시설")
print(f"Y(log1p 화재수) 통계: mean={df['Y'].mean():.4f}, std={df['Y'].std():.4f}, "
      f"max={df['Y'].max():.4f}, 비零비율={( df['Y']>0).mean()*100:.1f}%\n")

# ══════════════════════════════════════════════════════════════
# STEP 1 │ Ridge / Lasso 회귀  (Y = log1p 화재수)
# ══════════════════════════════════════════════════════════════
print("━"*60)
print("STEP 1 │ Ridge / Lasso 회귀  (Y = log1p(반경100m_화재수))")
print("━"*60)

dummies  = pd.get_dummies(df['업종'], drop_first=True, dtype=int)
df_reg   = pd.concat([df[RISK_VARS + ['Y']], dummies], axis=1).dropna()
REG_FEAT = RISK_VARS + list(dummies.columns)

Xr = StandardScaler().fit_transform(df_reg[REG_FEAT])
yr = df_reg['Y'].values

ridge = RidgeCV(alphas=[0.001, 0.01, 0.1, 1, 10, 100]).fit(Xr, yr)
lasso = LassoCV(cv=5, max_iter=10000, random_state=42).fit(Xr, yr)

print(f"\n  Ridge  α={ridge.alpha_:.4f}   R²={ridge.score(Xr,yr):.4f}")
print(f"  Lasso  α={lasso.alpha_:.6f}  R²={lasso.score(Xr,yr):.4f}")

print("\n  ── Ridge 계수 (표준화, 절댓값 순) ──")
for feat, coef in sorted(zip(REG_FEAT, ridge.coef_), key=lambda x: -abs(x[1])):
    bar  = '█' * max(1, int(abs(coef) * 30))
    sign = '+' if coef >= 0 else '-'
    print(f"    {feat:<25} {coef:+.4f}  {sign}{bar}")

# OLS 잔차 (Moran's I 용)
Xr_const = np.column_stack([np.ones(len(Xr)), Xr])
coef_ols, *_ = np.linalg.lstsq(Xr_const, yr, rcond=None)
resid = yr - Xr_const @ coef_ols

# ══════════════════════════════════════════════════════════════
# STEP 2 │ Moran's I 민감도 — k = 4, 8, 12, 16
# ══════════════════════════════════════════════════════════════
print("\n" + "━"*60)
print("STEP 2 │ Moran's I 민감도  (k = 4 / 8 / 12 / 16)")
print("━"*60)

coords_r = df.loc[df_reg.index, GEO_VARS].values
print(f"\n  {'k':>4}  {'Moran I':>9}  {'z-score':>9}  {'p-value':>9}  판정")
best_k = 8
for k in [4, 8, 12, 16]:
    w = KNN.from_array(coords_r, k=k)
    w.transform = 'R'
    mi = Moran(resid, w, permutations=999)
    sig = "유의 ✓" if mi.p_sim < 0.05 else "비유의"
    print(f"  {k:>4}  {mi.I:>+9.4f}  {mi.z_sim:>9.4f}  {mi.p_sim:>9.4f}  {sig}")

# 이후 k=8 사용 (기본값 유지 — 민감도 확인 목적)
w_r = KNN.from_array(coords_r, k=8)
w_r.transform = 'R'

# ══════════════════════════════════════════════════════════════
# STEP 3 │ Spatial Lag · ML_Error · GM_Error
# ══════════════════════════════════════════════════════════════
print("\n" + "━"*60)
print("STEP 3 │ Spatial 모델 비교  (Lag / ML_Error / GM_Error)")
print("━"*60)

coords_sp = df.loc[df_reg.index, GEO_VARS].values
y_sp = yr.reshape(-1, 1)
X_sp = np.column_stack([np.ones(len(Xr)), Xr])
w_sp = KNN.from_array(coords_sp, k=8)
w_sp.transform = 'R'
var_names = ['CONST'] + REG_FEAT

try:
    lag = ML_Lag(y_sp, X_sp, w=w_sp, name_y='log1p_화재수', name_x=var_names)
    print(f"\n  [Spatial Lag  — ML]")
    print(f"    ρ (공간시차)  = {lag.rho:+.4f}")
    print(f"    Pseudo-R²    = {lag.pr2:.4f}")
    print(f"    AIC          = {lag.aic:.2f}")
except Exception as e:
    print(f"  Spatial Lag 오류: {e}")

try:
    err = ML_Error(y_sp, X_sp, w=w_sp, name_y='log1p_화재수', name_x=var_names)
    print(f"\n  [Spatial Error — ML]")
    print(f"    λ (공간오차)  = {err.lam:+.4f}")
    print(f"    Pseudo-R²    = {err.pr2:.4f}")
    print(f"    AIC          = {err.aic:.2f}")
except Exception as e:
    print(f"  ML_Error 오류: {e}")

try:
    gm = GM_Error(y_sp, X_sp, w=w_sp, name_y='log1p_화재수', name_x=var_names)
    print(f"\n  [Spatial Error — GM (GMM 기반, 더 강건)]")
    lam_row = gm.output[gm.output['var_names'] == 'lambda']
    lam_val = lam_row['coefficients'].values[0] if len(lam_row) else float('nan')
    print(f"    λ (공간오차)  = {lam_val:+.4f}")
    print(f"    Pseudo-R²    = {gm.pr2:.4f}")
    print(f"    계수 (상위5, GM):")
    out = gm.output[gm.output['var_names'] != 'lambda'].head(6)
    for _, row in out.iterrows():
        se_str = f"se={row['std_err']:.4f}" if row['std_err'] is not None else ""
        print(f"      {row['var_names']:<25} {row['coefficients']:+.4f}  ({se_str})")
except Exception as e:
    print(f"  GM_Error 오류: {e}")

# ══════════════════════════════════════════════════════════════
# STEP 4 │ GWR 커널 비교  (bisquare vs gaussian)
# ══════════════════════════════════════════════════════════════
print("\n" + "━"*60)
print("STEP 4 │ GWR 커널 비교  (bisquare vs gaussian)")
print("━"*60)

N_GWR   = 2000
gwr_idx = np.random.RandomState(42).choice(len(df_reg), min(N_GWR, len(df_reg)), replace=False)
coords_gw = coords_sp[gwr_idx]
y_gw      = yr[gwr_idx].reshape(-1, 1)
X_gw      = StandardScaler().fit_transform(df_reg.iloc[gwr_idx][RISK_VARS].values)

print(f"\n  샘플: {len(gwr_idx):,}  변수: {RISK_VARS}")

gwr_results = {}
for kernel in ['bisquare', 'gaussian']:
    print(f"\n  ── 커널: {kernel} ──")
    try:
        bw_sel = Sel_BW(coords_gw, y_gw, X_gw, kernel=kernel, fixed=False)
        bw     = bw_sel.search(search_method='golden_section')
        print(f"  최적 bandwidth (adaptive NN): {int(bw)}")

        res = GWR(coords_gw, y_gw, X_gw, bw=bw, kernel=kernel, fixed=False).fit()
        print(f"  R²     = {res.R2:.4f}")
        print(f"  adj.R² = {res.adj_R2:.4f}")
        aic_val = getattr(res, 'AIC', None) or getattr(res, 'aicc', None)
        if aic_val: print(f"  AICc   = {aic_val:.2f}")

        print("  계수 변동 (평균 ± std):")
        for i, var in enumerate(RISK_VARS):
            p = res.params[:, i+1]
            print(f"    {var:<20} mean={p.mean():+.4f}  std={p.std():.4f}  [{p.min():+.4f}~{p.max():+.4f}]")

        gwr_results[kernel] = (res, bw, coords_gw)
    except Exception as e:
        print(f"  GWR({kernel}) 오류: {e}")

# 최종 GWR 결과 저장 (gaussian 우선, 없으면 bisquare)
for preferred in ['gaussian', 'bisquare']:
    if preferred in gwr_results:
        res, bw, cg = gwr_results[preferred]
        gwr_out = pd.DataFrame({'위도': cg[:, 0], '경도': cg[:, 1],
                                'local_R2': res.localR2.flatten(),
                                'kernel': preferred, 'bandwidth': int(bw)})
        for i, var in enumerate(RISK_VARS):
            gwr_out[f'coef_{var}'] = res.params[:, i+1]
            gwr_out[f'tval_{var}'] = res.tvalues[:, i+1]
        out_path = f'data/gwr_results_v2_{preferred}.csv'
        gwr_out.to_csv(out_path, index=False, encoding='utf-8-sig')
        print(f"\n  GWR v2 결과 저장: {out_path}  ({len(gwr_out)}행)")
        # 대시보드용 덮어쓰기 (gaussian 우선)
        gwr_out.to_csv('data/gwr_results.csv', index=False, encoding='utf-8-sig')
        print(f"  대시보드용 gwr_results.csv 갱신 완료 (커널={preferred}, Y=log1p_화재수)")
        break

print("\n" + "━"*60)
print("튜닝 파이프라인 v2 완료")
print("━"*60)
