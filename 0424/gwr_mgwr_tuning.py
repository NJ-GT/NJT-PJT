# -*- coding: utf-8 -*-
"""
GWR/MGWR 튜닝 스크립트
  1. 변수 상관·VIF 확인 (소방위험도_점수 포함 6변수)
  2. GWR bandwidth 비교: 54 / 80 / 100 / 120
  3. MGWR — 변수마다 다른 bandwidth
  4. 최적 결과 → data/gwr_results.csv 갱신
"""
import sys, glob, warnings, time
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from mgwr.gwr   import GWR, MGWR
from mgwr.sel_bw import Sel_BW

# ─────────────────────────────────────────────
# 데이터 로드
# ─────────────────────────────────────────────
f    = glob.glob('data/*0423*.csv')[0]
main = pd.read_csv(f, encoding='utf-8-sig')
core = pd.read_csv('data/data_with_fire_targets.csv', encoding='utf-8-sig')

c = core.columns
core_key = core[[c[4], c[5], c[17], c[22]]].copy()
core_key.columns = ['위도', '경도', '소방위험도_점수', '위험점수_AHP']

df = pd.merge(main, core_key, on=['위도', '경도'], how='left')

RISK_VARS5 = ['구조노후도', '단속위험도', '도로폭위험도', '집중도', '주변건물수']
RISK_VARS6 = RISK_VARS5 + ['소방위험도_점수']

for v in RISK_VARS6 + ['위험점수_AHP']:
    df[v] = pd.to_numeric(df[v], errors='coerce')

df_reg = df.dropna(subset=RISK_VARS6 + ['위험점수_AHP', '위도', '경도']).reset_index(drop=True)
print(f"시설 수: {len(df_reg):,}개\n")

# ─────────────────────────────────────────────
# STEP 1 │ 상관행렬 + VIF
# ─────────────────────────────────────────────
print("━"*60)
print("STEP 1 │ 변수 상관·VIF (6변수)")
print("━"*60)

corr = df_reg[RISK_VARS6].corr().round(3)
print("\n  상관행렬:")
print(corr.to_string(col_space=14))

# VIF
Xv = StandardScaler().fit_transform(df_reg[RISK_VARS6])
try:
    from numpy.linalg import inv
    vif = np.diag(len(Xv) * inv(Xv.T @ Xv))
    print("\n  VIF (>5 주의, >10 심각):")
    for var, vi in zip(RISK_VARS6, vif):
        flag = "  ⚠️ 높음" if vi > 5 else ""
        print(f"    {var:<20} {vi:.2f}{flag}")
except Exception as e:
    print(f"  VIF 계산 오류: {e}")

# ─────────────────────────────────────────────
# 변수 선택 결정
# ─────────────────────────────────────────────
# 집중도-주변건물수 상관이 높으면 집중도 제외
corr_val = abs(corr.loc['집중도', '주변건물수'])
if corr_val > 0.7:
    FINAL_VARS = ['구조노후도', '단속위험도', '도로폭위험도', '주변건물수', '소방위험도_점수']
    print(f"\n  → 집중도↔주변건물수 상관={corr_val:.3f} > 0.7: 집중도 제외, 5변수 사용")
else:
    FINAL_VARS = RISK_VARS6
    print(f"\n  → 상관 양호: 6변수 전부 사용")

print(f"  최종 변수: {FINAL_VARS}")

# ─────────────────────────────────────────────
# 공통 샘플 (2000개, GWR용)
# ─────────────────────────────────────────────
N_GWR = 2000
rng = np.random.RandomState(42)
idx = rng.choice(len(df_reg), min(N_GWR, len(df_reg)), replace=False)

coords = df_reg[['위도', '경도']].values[idx]
yr     = df_reg['위험점수_AHP'].values[idx].reshape(-1, 1)
Xr     = StandardScaler().fit_transform(df_reg[FINAL_VARS].values[idx])

print(f"\n  GWR 샘플: {len(idx):,}개  변수: {FINAL_VARS}")

# ─────────────────────────────────────────────
# STEP 2 │ GWR Bandwidth 비교 (54 / 80 / 100 / 120)
# ─────────────────────────────────────────────
print("\n" + "━"*60)
print("STEP 2 │ GWR Bandwidth 비교 (bisquare, adaptive NN)")
print("━"*60)
print(f"\n  {'BW':>5}  {'R²':>8}  {'adj.R²':>8}  {'AICc':>10}  시간")

bw_results = {}
for bw in [54, 80, 100, 120]:
    t0 = time.time()
    res = GWR(coords, yr, Xr, bw=bw, kernel='bisquare', fixed=False).fit()
    t1 = time.time()
    aicc = getattr(res, 'AIC', None) or getattr(res, 'aicc', None) or float('nan')
    bw_results[bw] = res
    print(f"  {bw:>5}  {res.R2:>8.4f}  {res.adj_R2:>8.4f}  {aicc:>10.2f}  {t1-t0:.1f}s")

# ─────────────────────────────────────────────
# STEP 3 │ 최적 BW 자동 선택 (AICc 기준)
# ─────────────────────────────────────────────
print("\n" + "━"*60)
print("STEP 3 │ 최적 Bandwidth 자동 선택 (Sel_BW golden section)")
print("━"*60)

t0 = time.time()
bw_sel = Sel_BW(coords, yr, Xr, kernel='bisquare', fixed=False)
best_bw = bw_sel.search(search_method='golden_section')
t1 = time.time()
print(f"\n  최적 BW = {int(best_bw)}  ({t1-t0:.1f}s)")

gwr_best = GWR(coords, yr, Xr, bw=best_bw, kernel='bisquare', fixed=False).fit()
aicc_best = getattr(gwr_best, 'AIC', None) or getattr(gwr_best, 'aicc', None)
print(f"  R²={gwr_best.R2:.4f}  adj.R²={gwr_best.adj_R2:.4f}  AICc={aicc_best:.2f}")
print("\n  계수 변동 (mean ± std):")
for i, v in enumerate(FINAL_VARS):
    p = gwr_best.params[:, i+1]
    print(f"    {v:<20} mean={p.mean():+.4f}  std={p.std():.4f}  [{p.min():+.3f}~{p.max():+.3f}]")

# ─────────────────────────────────────────────
# STEP 4 │ MGWR (변수마다 다른 bandwidth)
# ─────────────────────────────────────────────
print("\n" + "━"*60)
print("STEP 4 │ MGWR — 변수별 최적 Bandwidth")
print("━"*60)

# MGWR는 느림 → 500개 샘플로 먼저 탐색
N_MGWR = min(500, len(idx))
m_idx  = rng.choice(len(idx), N_MGWR, replace=False)
c_mgwr = coords[m_idx]
y_mgwr = yr[m_idx]
X_mgwr = Xr[m_idx]

print(f"\n  MGWR 샘플: {N_MGWR}개 (속도상 제한)")

try:
    t0 = time.time()
    # mgwr 2.2.x: multi=True는 Sel_BW 생성자에, search()에는 넘기지 않음
    mgwr_sel = Sel_BW(c_mgwr, y_mgwr, X_mgwr,
                      multi=True, kernel='bisquare', fixed=False)
    mgwr_sel.search(verbose=False)
    t1 = time.time()

    bws_multi = mgwr_sel.bw  # 변수별 bandwidth 배열
    print(f"  변수별 최적 BW ({t1-t0:.1f}s):")
    var_labels = ['Intercept'] + FINAL_VARS
    for v, bw_v in zip(var_labels, bws_multi):
        print(f"    {v:<20} BW={int(bw_v)}")

    mgwr_res = MGWR(c_mgwr, y_mgwr, X_mgwr, mgwr_sel,
                    kernel='bisquare', fixed=False).fit()
    aicc_mgwr = getattr(mgwr_res, 'AIC', None) or getattr(mgwr_res, 'aicc', None)
    print(f"\n  MGWR R²={mgwr_res.R2:.4f}  adj.R²={mgwr_res.adj_R2:.4f}  AICc={aicc_mgwr:.2f}")
    print("\n  MGWR 계수 변동 (mean ± std):")
    for i, v in enumerate(FINAL_VARS):
        p = mgwr_res.params[:, i+1]
        print(f"    {v:<20} mean={p.mean():+.4f}  std={p.std():.4f}")

    MGWR_OK = True
except Exception as e:
    print(f"  MGWR 오류: {e}")
    MGWR_OK = False

# ─────────────────────────────────────────────
# STEP 5 │ 최종 결과 저장 (GWR best → CSV)
# ─────────────────────────────────────────────
print("\n" + "━"*60)
print("STEP 5 │ 최적 GWR 결과 저장")
print("━"*60)

out = pd.DataFrame({
    '위도': coords[:, 0],
    '경도': coords[:, 1],
    'local_R2': gwr_best.localR2.flatten(),
    'bandwidth': int(best_bw),
})
for i, v in enumerate(FINAL_VARS):
    out[f'coef_{v}'] = gwr_best.params[:, i+1]
    out[f'tval_{v}'] = gwr_best.tvalues[:, i+1]

out.to_csv('data/gwr_results.csv', index=False, encoding='utf-8-sig')
print(f"\n  저장: data/gwr_results.csv  ({len(out)}행, BW={int(best_bw)}, 변수={FINAL_VARS})")

print("\n" + "━"*60)
print("완료")
print("━"*60)
