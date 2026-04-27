# -*- coding: utf-8 -*-
"""
앙상블 + 공간회귀 전체 파이프라인
  Step 1 : 업종별 K-Means (외국인민박 K=2 / 숙박업 K=3 / 관광숙박업 K=3)
  Step 2 : LightGBM + XGBoost + CatBoost + Stacking 분류
  Step 3 : Ridge / Lasso 회귀  (Y = AHP 위험점수)
  Step 4 : Moran's I (OLS 잔차 공간자기상관)
  Step 5 : Spatial Lag / Spatial Error
  Step 6 : GWR (Geographically Weighted Regression)
"""
import sys, glob, warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import Ridge, Lasso, LogisticRegression, RidgeCV, LassoCV
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score, classification_report
from lightgbm  import LGBMClassifier
from xgboost   import XGBClassifier
from catboost  import CatBoostClassifier

from libpysal.weights import KNN
from esda.moran import Moran
from spreg import ML_Lag, ML_Error
from mgwr.gwr   import GWR
from mgwr.sel_bw import Sel_BW

# ──────────────────────────────────────────────
# 데이터 로드
# ──────────────────────────────────────────────
f    = glob.glob('data/*0423*.csv')[0]
main = pd.read_csv(f, encoding='utf-8-sig')
core = pd.read_csv('data/data_with_fire_targets.csv', encoding='utf-8-sig')

df = pd.merge(main, core[['위도','경도','위험점수_AHP','반경100m_화재수','이동시간초']],
              on=['위도','경도'], how='left')
df = df.dropna(subset=['위험점수_AHP','구조노후도','단속위험도','도로폭위험도','집중도','주변건물수']).reset_index(drop=True)

RISK_VARS = ['구조노후도','단속위험도','도로폭위험도','집중도','주변건물수']
GEO_VARS  = ['위도','경도']
FEAT_COLS = RISK_VARS + GEO_VARS
BEST_K    = {'외국인관광도시민박업': 2, '숙박업': 3, '관광숙박업': 3}

print(f"로드 완료: {len(df):,}개 시설\n")

# ══════════════════════════════════════════════════════════════
# STEP 1 │ 업종별 K-Means
# ══════════════════════════════════════════════════════════════
print("━"*60)
print("STEP 1 │ 업종별 K-Means 클러스터링 (Method 1)")
print("━"*60)

df['업종내_군집'] = -1
for upjong, k in BEST_K.items():
    idx  = df[df['업종'] == upjong].index
    X    = StandardScaler().fit_transform(df.loc[idx, RISK_VARS])
    lbl  = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)
    df.loc[idx, '업종내_군집'] = lbl
    sizes = dict(zip(*np.unique(lbl, return_counts=True)))
    print(f"  [{upjong}]  K={k}  분포: { {f'군집{k}': v for k,v in sizes.items()} }")

# ══════════════════════════════════════════════════════════════
# STEP 2 │ 앙상블 분류
# ══════════════════════════════════════════════════════════════
print("\n" + "━"*60)
print("STEP 2 │ 앙상블 분류  LightGBM · XGBoost · CatBoost · Stacking")
print("━"*60)

def run_ensemble(sub, feat_cols, target_col):
    X  = StandardScaler().fit_transform(sub[feat_cols])
    y  = sub[target_col].astype(int).values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    lgbm = LGBMClassifier(n_estimators=300, learning_rate=0.05,
                          num_leaves=31, random_state=42, verbose=-1)
    xgb  = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=5,
                         random_state=42, eval_metric='mlogloss', verbosity=0)
    cat  = CatBoostClassifier(iterations=300, learning_rate=0.05,
                              depth=5, random_state=42, verbose=0)
    stack = StackingClassifier(
        estimators=[('lgbm', LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=42, verbose=-1)),
                    ('xgb',  XGBClassifier(n_estimators=300, learning_rate=0.05, random_state=42, eval_metric='mlogloss', verbosity=0)),
                    ('cat',  CatBoostClassifier(iterations=300, learning_rate=0.05, random_state=42, verbose=0))],
        final_estimator=LogisticRegression(max_iter=1000),
        cv=5, passthrough=False,
    )

    results = {}
    for name, clf in [('LightGBM', lgbm), ('XGBoost', xgb), ('CatBoost', cat), ('Stacking', stack)]:
        clf.fit(Xtr, ytr)
        acc = accuracy_score(yte, clf.predict(Xte))
        results[name] = (clf, acc)

    # Feature importance (LGBM)
    fi = dict(zip(feat_cols, lgbm.feature_importances_))
    top3 = sorted(fi.items(), key=lambda x: -x[1])[:3]
    return results, top3

for upjong, k in BEST_K.items():
    sub = df[df['업종'] == upjong].dropna(subset=FEAT_COLS+['업종내_군집'])
    print(f"\n  [{upjong}]  K={k}  N={len(sub)}")
    res, top3 = run_ensemble(sub, FEAT_COLS, '업종내_군집')
    for name, (clf, acc) in res.items():
        print(f"    {name:<12}  test_acc = {acc:.4f}")
    print(f"    LGBM 중요변수 top3: {', '.join([f'{k}({v:.0f})' for k,v in top3])}")

# ══════════════════════════════════════════════════════════════
# STEP 3 │ Ridge / Lasso 회귀
# ══════════════════════════════════════════════════════════════
print("\n" + "━"*60)
print("STEP 3 │ Ridge / Lasso 회귀  (Y = AHP 위험점수)")
print("━"*60)

dummies  = pd.get_dummies(df['업종'], drop_first=True, dtype=int)
df_reg   = pd.concat([df[RISK_VARS + ['위험점수_AHP']], dummies], axis=1).dropna()
REG_FEAT = RISK_VARS + list(dummies.columns)

Xr = StandardScaler().fit_transform(df_reg[REG_FEAT])
yr = df_reg['위험점수_AHP'].values

ridge = RidgeCV(alphas=[0.001, 0.01, 0.1, 1, 10, 100]).fit(Xr, yr)
lasso = LassoCV(cv=5, max_iter=10000, random_state=42).fit(Xr, yr)

print(f"\n  Ridge  α={ridge.alpha_:.4f}   R²={ridge.score(Xr,yr):.4f}")
print(f"  Lasso  α={lasso.alpha_:.6f}  R²={lasso.score(Xr,yr):.4f}")

print("\n  ── Ridge 계수 (표준화) ──")
for feat, coef in sorted(zip(REG_FEAT, ridge.coef_), key=lambda x: -abs(x[1])):
    bar = '█' * int(abs(coef)*20)
    sign = '+' if coef >= 0 else '-'
    print(f"    {feat:<25} {coef:+.4f}  {sign}{bar}")

print("\n  ── Lasso 비제로 계수 ──")
for feat, coef in sorted(zip(REG_FEAT, lasso.coef_), key=lambda x: -abs(x[1])):
    if abs(coef) > 1e-6:
        print(f"    {feat:<25} {coef:+.4f}")

# OLS 잔차 (Moran's I 용)
Xr_const = np.column_stack([np.ones(len(Xr)), Xr])
coef_ols, *_ = np.linalg.lstsq(Xr_const, yr, rcond=None)
resid = yr - Xr_const @ coef_ols

# ══════════════════════════════════════════════════════════════
# STEP 4 │ Moran's I
# ══════════════════════════════════════════════════════════════
print("\n" + "━"*60)
print("STEP 4 │ Moran's I (OLS 잔차 공간자기상관)")
print("━"*60)

coords_r = df.loc[df_reg.index, GEO_VARS].values
w_r = KNN.from_array(coords_r, k=8)
w_r.transform = 'R'

mi = Moran(resid, w_r, permutations=999)
print(f"\n  Moran's I = {mi.I:+.4f}")
print(f"  E[I]      = {mi.EI:.4f}")
print(f"  z-score   = {mi.z_sim:.4f}")
print(f"  p-value   = {mi.p_sim:.4f}  ({'유의 → Spatial 모델 필요' if mi.p_sim < 0.05 else '비유의'} )")

# ══════════════════════════════════════════════════════════════
# STEP 5 │ Spatial Lag / Spatial Error
# ══════════════════════════════════════════════════════════════
print("\n" + "━"*60)
print("STEP 5 │ Spatial Lag · Spatial Error 모델")
print("━"*60)

coords_sp = df.loc[df_reg.index, GEO_VARS].values
y_sp = yr.reshape(-1, 1)
X_sp = np.column_stack([np.ones(len(Xr)), Xr])
w_sp = KNN.from_array(coords_sp, k=8)
w_sp.transform = 'R'
var_names = ['CONST'] + REG_FEAT

try:
    lag = ML_Lag(y_sp, X_sp, w=w_sp, name_y='AHP위험점수', name_x=var_names)
    print(f"\n  Spatial Lag")
    print(f"    ρ (공간시차)  = {lag.rho:+.4f}")
    print(f"    Pseudo-R²    = {lag.pr2:.4f}")
    print(f"    AIC          = {lag.aic:.2f}")
    print(f"    Log-lik      = {lag.logll:.2f}")
except Exception as e:
    print(f"  Spatial Lag 오류: {e}")

try:
    err = ML_Error(y_sp, X_sp, w=w_sp, name_y='AHP위험점수', name_x=var_names)
    print(f"\n  Spatial Error")
    print(f"    λ (공간오차)  = {err.lam:+.4f}")
    print(f"    Pseudo-R²    = {err.pr2:.4f}")
    print(f"    AIC          = {err.aic:.2f}")
    print(f"    Log-lik      = {err.logll:.2f}")
except Exception as e:
    print(f"  Spatial Error 오류: {e}")

# ══════════════════════════════════════════════════════════════
# STEP 6 │ GWR
# ══════════════════════════════════════════════════════════════
print("\n" + "━"*60)
print("STEP 6 │ GWR (Geographically Weighted Regression)")
print("━"*60)

# 샘플링 (전체 사용 시 수 분 소요 → 2000개로 제한)
N_GWR = 2000
gwr_idx   = np.random.RandomState(42).choice(len(df_reg), min(N_GWR, len(df_reg)), replace=False)
coords_gw = coords_sp[gwr_idx]
y_gw      = yr[gwr_idx].reshape(-1, 1)
X_gw      = StandardScaler().fit_transform(df_reg.iloc[gwr_idx][RISK_VARS].values)

print(f"\n  샘플 수: {len(gwr_idx):,}  변수: {RISK_VARS}")
try:
    bw_sel = Sel_BW(coords_gw, y_gw, X_gw, kernel='bisquare', fixed=False)
    bw     = bw_sel.search(search_method='golden_section')
    print(f"  최적 bandwidth (adaptive NN): {int(bw)}")

    gwr_res = GWR(coords_gw, y_gw, X_gw, bw=bw, kernel='bisquare', fixed=False).fit()
    print(f"  R²      = {gwr_res.R2:.4f}")
    print(f"  adj.R²  = {gwr_res.adj_R2:.4f}")
    aic_val = getattr(gwr_res, 'AIC', None) or getattr(gwr_res, 'aicc', None)
    if aic_val: print(f"  AICc    = {aic_val:.2f}")

    print("\n  ── 지역별 계수 변동 (평균 ± 표준편차) ──")
    for i, var in enumerate(RISK_VARS):
        p = gwr_res.params[:, i+1]
        print(f"    {var:<20}  mean={p.mean():+.4f}  std={p.std():.4f}"
              f"  [{p.min():+.4f} ~ {p.max():+.4f}]")

    # ── GWR 결과 CSV 저장 ──
    gwr_out = pd.DataFrame({
        '위도': coords_gw[:, 0],
        '경도': coords_gw[:, 1],
        'local_R2': gwr_res.localR2.flatten(),
    })
    for i, var in enumerate(RISK_VARS):
        gwr_out[f'coef_{var}'] = gwr_res.params[:, i+1]
        gwr_out[f'tval_{var}'] = gwr_res.tvalues[:, i+1]
    gwr_out.to_csv('data/gwr_results.csv', index=False, encoding='utf-8-sig')
    print(f"\n  GWR 결과 저장: data/gwr_results.csv  ({len(gwr_out)}행)")

except Exception as e:
    print(f"  GWR 오류: {e}")

print("\n" + "━"*60)
print("전체 파이프라인 완료")
print("━"*60)
