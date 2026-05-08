# -*- coding: utf-8 -*-
"""6개 핵심 변수 기반 OLS / Spatial Lag / Spatial Error 모형 비교 스크립트.

목적:
    6변수만 사용한 상태에서 OLS, SLM, GM_Error 모형을 적합하여
    R², 잔차 Moran's I, ρ(공간 시차 계수), λ(공간 오차 계수) 등을 추출한다.
    Y(타깃)는 두 가지 — log(1+반경100m_화재수)와 위험점수_AHP.
"""

import sys
import glob
import warnings

# spreg 라이브러리 내부 경고 숨김
warnings.filterwarnings("ignore")
# Windows 콘솔 한글 깨짐 방지
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
# K-최근접 공간 가중치
from libpysal.weights import KNN
# Global Moran's I (잔차 공간자기상관)
from esda.moran import Moran

# 0423 패턴 파일 자동 탐색 (메인 변수 테이블)
f = glob.glob("data/*0423*.csv")[0]
main = pd.read_csv(f, encoding="utf-8-sig")
# 화재 타깃이 부착된 보조 테이블 (위/경도 키로 결합)
core = pd.read_csv("data/data_with_fire_targets.csv", encoding="utf-8-sig")
# 핵심 컬럼만 추리기 — 컬럼 인덱스로 접근(스키마 안정 가정)
c = core.columns
core_key = core[[c[4], c[5], c[17], c[22], c[45]]].copy()
core_key.columns = [
    "위도",
    "경도",
    "소방위험도_점수",
    "위험점수_AHP",
    "반경100m_화재수",
]

# 위/경도 키로 결합
df = pd.merge(main, core_key, on=["위도", "경도"], how="left")
# 분석에 사용할 6변수
VARS6 = [
    "구조노후도",
    "단속위험도",
    "도로폭위험도",
    "집중도",
    "주변건물수",
    "소방위험도_점수",
]
# 모든 분석 변수 + 타깃 + 좌표를 수치로 강제 변환
for v in VARS6 + ["위험점수_AHP", "반경100m_화재수"]:
    df[v] = pd.to_numeric(df[v], errors="coerce")
# 결측 한 행이라도 있으면 모델 적합 불가 -> 제거
df = df.dropna(
    subset=VARS6 + ["위험점수_AHP", "반경100m_화재수", "위도", "경도"]
).reset_index(drop=True)

# Y = log(1 + 반경100m 화재수) — 분포 안정화
df["Y"] = np.log1p(df["반경100m_화재수"])
N = len(df)
print(f"샘플: {N:,}")

# 위경도 좌표로 KNN 가중치 (k=8) — 행 표준화
coords = df[["위도", "경도"]].values
W = KNN.from_array(coords, k=8)
W.transform = "r"

# 독립변수 표준화 (평균 0, 분산 1)
Xs = StandardScaler().fit_transform(df[VARS6].values)
Y = df["Y"].values

# ── OLS 적합 (직접 lstsq로 해서 잔차/R² 계산) ──
# X에 절편(상수항) 컬럼 추가
Xc = np.column_stack([np.ones(N), Xs])
b, *_ = np.linalg.lstsq(Xc, Y, rcond=None)
Yhat = Xc @ b
resid = Y - Yhat
# R² = 1 - SSR/SST
r2_ols = 1 - ((resid**2).sum() / ((Y - Y.mean()) ** 2).sum())
# 잔차의 공간자기상관 — 0에 가깝지 않으면 공간 모형이 필요함을 시사
mi_res = Moran(resid, W, permutations=499)
print(f"\nOLS  R²={r2_ols:.4f}  잔차 Moran's I={mi_res.I:.4f} p={mi_res.p_sim:.4f}")

# ── Spatial Lag 모형 (OLS + lagged Y를 추가 변수로) ──
# WY: 이웃 평균 Y — 공간 상호작용을 X에 직접 포함
lag_Y = W.sparse.dot(Y)
Xc2 = np.column_stack([np.ones(N), Xs, lag_Y])
b2, *_ = np.linalg.lstsq(Xc2, Y, rcond=None)
Yhat2 = Xc2 @ b2
r2_slm = 1 - ((Y - Yhat2) ** 2).sum() / ((Y - Y.mean()) ** 2).sum()
# rho는 lag_Y 항의 계수
rho = b2[-1]
print(f"SLM  R²={r2_slm:.4f}  ρ(lag_Y)={rho:.4f}")

# ── Spatial Error (GM_Error로 λ 추정) ──
try:
    # GM_Error: 일반화 적률법 기반 공간 오차 모형
    from spreg import GM_Error

    gm = GM_Error(Y.reshape(-1, 1), Xs, w=W, name_y="Y", name_x=VARS6)
    # output 테이블에서 lambda 행을 찾아 값 추출
    lam_row = gm.output[gm.output["var_names"] == "lambda"]
    lam_val = lam_row["coefficients"].values[0]
    print(f"GM_Error  λ={lam_val:.4f}")
except Exception as e:
    # 모듈 로드/적합 실패 시 친절한 메시지
    print(f"GM_Error 오류: {e}")

# ── Y = AHP 점수 기반(GWR 페이지와 동일) 비교 ──
print("\n--- AHP 기준 (GWR page와 동일) ---")
Ya = df["위험점수_AHP"].values
Xca = np.column_stack([np.ones(N), Xs])
ba, *_ = np.linalg.lstsq(Xca, Ya, rcond=None)
resid_a = Ya - Xca @ ba
r2_a = 1 - ((resid_a**2).sum() / ((Ya - Ya.mean()) ** 2).sum())
mi_a = Moran(resid_a, W, permutations=499)
# AHP 기준에서도 동일하게 SLM
lag_Ya = W.sparse.dot(Ya)
Xca2 = np.column_stack([np.ones(N), Xs, lag_Ya])
ba2, *_ = np.linalg.lstsq(Xca2, Ya, rcond=None)
r2_slm_a = 1 - ((Ya - Xca2 @ ba2) ** 2).sum() / ((Ya - Ya.mean()) ** 2).sum()
print(f"AHP OLS  R²={r2_a:.4f}  잔차 Moran's I={mi_a.I:.4f} p={mi_a.p_sim:.4f}")
print(f"AHP SLM  R²={r2_slm_a:.4f}  ρ={ba2[-1]:.4f}")
