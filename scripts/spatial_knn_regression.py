# -*- coding: utf-8 -*-
"""
숙박시설(X) ↔ 화재출동 재산피해액(Y) 공간 KNN 매칭 후 회귀 분석
  - 각 숙박시설에서 반경 내 가장 가까운 화재 5건의 평균 재산피해액을 Y로 설정
  - 평면좌표(EPSG:5179) 기반 거리 계산
  - OLS 회귀 + R² 출력
"""
import glob, sys
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from shapely.geometry import Point
from sklearn.neighbors import NearestNeighbors
import statsmodels.api as sm

sys.stdout.reconfigure(encoding="utf-8")
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

BASE = "C:/Users/USER/Documents/GitHub/기말공모전/NJT-PJT"

X_COLS = ["구조노후도", "단속위험도", "도로폭위험도", "집중도", "주변건물수",
          "승인연도", "총층수", "연면적"]
K = 5  # 최근접 화재 건수

# ── 1. 데이터 로드 ────────────────────────────────────────────────────
print("━" * 60)
print("STEP 1 │ 데이터 로드")
print("━" * 60)

src = glob.glob(f"{BASE}/0424/*/tables/*AHP3*.csv")[0]
acc = pd.read_csv(src, encoding="utf-8-sig")
for c in X_COLS + ["위도", "경도"]:
    acc[c] = pd.to_numeric(acc[c], errors="coerce")
acc = acc.dropna(subset=X_COLS + ["위도", "경도"]).reset_index(drop=True)
print(f"  숙박시설: {len(acc):,}개")

fire = pd.read_csv(f"{BASE}/data/화재출동/화재출동_2021_2024.csv",
                   encoding="utf-8-sig", low_memory=False)
for c in ["위도", "경도", "재산피해액(천원)"]:
    fire[c] = pd.to_numeric(fire[c], errors="coerce")
fire = fire.dropna(subset=["위도", "경도", "재산피해액(천원)"]).reset_index(drop=True)
print(f"  화재출동: {len(fire):,}건  (재산피해액 0원 포함)")

# ── 2. 평면좌표 변환 (EPSG:5179, 단위: m) ────────────────────────────
print("\n" + "━" * 60)
print("STEP 2 │ 평면좌표 변환 (EPSG:5179)")
print("━" * 60)

def to_proj(df, lat_col="위도", lon_col="경도"):
    gdf = gpd.GeoDataFrame(
        df.copy(),
        geometry=[Point(lon, lat) for lon, lat in zip(df[lon_col], df[lat_col])],
        crs="EPSG:4326",
    )
    gdf = gdf.to_crs(epsg=5179)
    return np.column_stack([gdf.geometry.x.values, gdf.geometry.y.values])

acc_coords  = to_proj(acc)
fire_coords = to_proj(fire)
print(f"  숙박시설 좌표 shape: {acc_coords.shape}")
print(f"  화재출동 좌표 shape: {fire_coords.shape}")

# ── 3. KNN 매칭 — 숙박시설별 최근접 화재 5건 평균 피해액 ─────────────
print("\n" + "━" * 60)
print(f"STEP 3 │ KNN 매칭 (K={K}, 거리 기준: 평면좌표)")
print("━" * 60)

nn = NearestNeighbors(n_neighbors=K, algorithm="ball_tree", metric="euclidean", n_jobs=-1)
nn.fit(fire_coords)
distances, indices = nn.kneighbors(acc_coords)

fire_y = fire["재산피해액(천원)"].values
acc["Y_평균재산피해액"] = np.mean(fire_y[indices], axis=1)
acc["Y_최근접거리m"]   = distances[:, 0].round(1)

print(f"  매칭 완료")
print(f"  최근접 화재까지 거리  — 중앙값: {np.median(distances[:,0]):.0f}m  최대: {np.max(distances[:,0]):.0f}m")
print(f"  Y(평균재산피해액) 분포 — 중앙값: {acc['Y_평균재산피해액'].median():,.0f}천원  "
      f"평균: {acc['Y_평균재산피해액'].mean():,.0f}천원")

# ── 4. OLS 회귀 ──────────────────────────────────────────────────────
print("\n" + "━" * 60)
print("STEP 4 │ OLS 회귀  Y = 평균재산피해액(천원)")
print("━" * 60)

df_reg = acc[X_COLS + ["Y_평균재산피해액"]].dropna().reset_index(drop=True)
X_mat  = sm.add_constant(df_reg[X_COLS].astype(float))
y_vec  = df_reg["Y_평균재산피해액"].astype(float)

ols = sm.OLS(y_vec, X_mat).fit(cov_type="HC3")

print(f"\n  N = {int(ols.nobs):,}   R² = {ols.rsquared:.4f}   Adj.R² = {ols.rsquared_adj:.4f}")
print(f"  F-stat = {ols.fvalue:.2f}   p = {ols.f_pvalue:.4f}")
print("\n  [변수별 계수]")
for v in X_COLS:
    coef = ols.params[v]
    pval = ols.pvalues[v]
    sig  = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else "n.s."))
    print(f"    {v:<20} coef={coef:+.4f}   p={pval:.4f}  {sig}")

# ── 5. 로그 변환 OLS (Y>0인 경우) ────────────────────────────────────
print("\n" + "━" * 60)
print("STEP 5 │ log(Y+1) OLS — 왜도 보정")
print("━" * 60)

df_log = df_reg.copy()
df_log["logY"] = np.log1p(df_log["Y_평균재산피해액"])
X_log = sm.add_constant(df_log[X_COLS].astype(float))

ols_log = sm.OLS(df_log["logY"], X_log).fit(cov_type="HC3")
print(f"\n  R² = {ols_log.rsquared:.4f}   Adj.R² = {ols_log.rsquared_adj:.4f}")
print(f"  F-stat = {ols_log.fvalue:.2f}   p = {ols_log.f_pvalue:.4f}")
print("\n  [변수별 계수 — log(Y+1) 기준]")
for v in X_COLS:
    coef = ols_log.params[v]
    pval = ols_log.pvalues[v]
    sig  = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else "n.s."))
    print(f"    {v:<20} coef={coef:+.4f}   p={pval:.4f}  {sig}")

# ── 6. 시각화 ─────────────────────────────────────────────────────────
print("\n" + "━" * 60)
print("STEP 6 │ 시각화 저장")
print("━" * 60)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# 왼쪽: Y 분포 (로그 스케일 히스토그램)
axes[0].hist(np.log1p(df_reg["Y_평균재산피해액"]), bins=40, color="#4C72B0", alpha=0.8, edgecolor="white")
axes[0].set_xlabel("log(평균재산피해액 + 1)")
axes[0].set_ylabel("숙박시설 수")
axes[0].set_title(f"Y 분포 — 최근접 화재 {K}건 평균 재산피해액")
axes[0].grid(axis="y", alpha=0.3)
axes[0].spines[["top","right"]].set_visible(False)

# 오른쪽: 실제 vs 예측 (log 모델)
y_pred_log = ols_log.fittedvalues
axes[1].scatter(df_log["logY"], y_pred_log, alpha=0.3, s=10, color="#55A868")
lim = [min(df_log["logY"].min(), y_pred_log.min()),
       max(df_log["logY"].max(), y_pred_log.max())]
axes[1].plot(lim, lim, "r--", linewidth=1.2, label="완벽예측선")
axes[1].set_xlabel("실제 log(Y+1)")
axes[1].set_ylabel("예측 log(Y+1)")
axes[1].set_title(f"실제 vs 예측  (R²={ols_log.rsquared:.3f})")
axes[1].legend(fontsize=9)
axes[1].grid(alpha=0.25)
axes[1].spines[["top","right"]].set_visible(False)

plt.suptitle("숙박시설 X → 주변 화재 재산피해액(Y) KNN 매칭 OLS 회귀", fontsize=13, y=1.01)
plt.tight_layout()
out = f"{BASE}/data/knn_ols_result.png"
fig.savefig(out, dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"  저장: {out}")

# ── 7. 결과 저장 ──────────────────────────────────────────────────────
out_csv = f"{BASE}/data/knn_ols_table.csv"
acc[X_COLS + ["업종그룹", "위도", "경도", "Y_평균재산피해액", "Y_최근접거리m"]].to_csv(
    out_csv, index=False, encoding="utf-8-sig"
)
print(f"  저장: {out_csv}")
print("\n" + "━" * 60)
print("완료")
print("━" * 60)
