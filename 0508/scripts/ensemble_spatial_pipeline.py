# -*- coding: utf-8 -*-
"""
앙상블 분류 + 공간회귀 통합 파이프라인 (Step 1~6 일괄 실행).

전체 흐름:
    Step 1 : 업종별 K-Means (외국인민박 K=2 / 숙박업 K=3 / 관광숙박업 K=3)
    Step 2 : LightGBM + XGBoost + CatBoost + Stacking 분류
    Step 3 : Ridge / Lasso 회귀  (Y = AHP 위험점수)
    Step 4 : Moran's I (OLS 잔차 공간자기상관)
    Step 5 : Spatial Lag / Spatial Error
    Step 6 : GWR (Geographically Weighted Regression) — 결과는 CSV 로 저장

입력:
    - data/*0423*.csv       : 분석용 마스터 테이블(0423 일자 버전)
    - data/data_with_fire_targets.csv : AHP 위험점수/타겟 변수가 결합된 보조 테이블

출력 (콘솔/파일):
    - 콘솔: 각 단계 요약 통계 출력
    - data/gwr_results.csv : 좌표별 local R²와 변수별 계수/t-값 저장
"""

import sys  # 표준출력 인코딩 재설정
import glob  # 0423 일자 파일 패턴 매칭 로드
import warnings  # sklearn/spreg/mgwr 경고 억제

# 기타 deprecation/convergence 경고를 무시 — 출력 깔끔하게
warnings.filterwarnings("ignore")
# 한글 출력 깨짐 방지
sys.stdout.reconfigure(encoding="utf-8")

# 수치/데이터 처리
import numpy as np
import pandas as pd

# 전처리/머신러닝
from sklearn.preprocessing import StandardScaler  # 표준화(평균 0, 분산 1)
from sklearn.cluster import KMeans  # 비지도 클러스터링
from sklearn.model_selection import train_test_split  # 학습/테스트 분할
from sklearn.linear_model import LogisticRegression, RidgeCV, LassoCV  # 메타학습기/규제 회귀
from sklearn.ensemble import StackingClassifier  # 앙상블 스태킹
from sklearn.metrics import accuracy_score  # 분류 성능 지표

# 부스팅 트리 3종 — 분류 비교
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# 공간통계 라이브러리
from libpysal.weights import KNN  # K-최근접 이웃 공간가중행렬
from esda.moran import Moran  # 모란 I 통계
from spreg import ML_Lag, ML_Error  # 공간시차/공간오차 회귀
from mgwr.gwr import GWR  # 지리가중회귀
from mgwr.sel_bw import Sel_BW  # 최적 bandwidth 탐색

# ──────────────────────────────────────────────
# 데이터 로드
# ──────────────────────────────────────────────
# 0423 일자가 포함된 CSV 첫 번째 파일을 마스터로 사용
f = glob.glob("data/*0423*.csv")[0]
main = pd.read_csv(f, encoding="utf-8-sig")
# 화재 타겟/AHP 위험점수가 들어있는 보조 테이블
core = pd.read_csv("data/data_with_fire_targets.csv", encoding="utf-8-sig")

# 좌표(위도/경도) 기준 좌측 조인 — main 행 보존, core 의 핵심 변수만 가져오기
df = pd.merge(
    main,
    core[["위도", "경도", "위험점수_AHP", "반경100m_화재수", "이동시간초"]],
    on=["위도", "경도"],
    how="left",
)
# 5개 위험변수와 AHP 점수에 결측이 없는 행만 분석에 사용
df = df.dropna(
    subset=[
        "위험점수_AHP",
        "구조노후도",
        "단속위험도",
        "도로폭위험도",
        "집중도",
        "주변건물수",
    ]
).reset_index(drop=True)

# 5대 위험변수 — 모델 입력 핵심
RISK_VARS = ["구조노후도", "단속위험도", "도로폭위험도", "집중도", "주변건물수"]
# 공간 좌표 변수 — KNN/GWR 좌표
GEO_VARS = ["위도", "경도"]
# 분류 모델용 입력: 위험변수 + 좌표
FEAT_COLS = RISK_VARS + GEO_VARS
# 업종별 사전 결정된 최적 K — 별도 튜닝 결과를 하드코딩
BEST_K = {"외국인관광도시민박업": 2, "숙박업": 3, "관광숙박업": 3}

print(f"로드 완료: {len(df):,}개 시설\n")

# ══════════════════════════════════════════════════════════════
# STEP 1 │ 업종별 K-Means
# ══════════════════════════════════════════════════════════════
print("━" * 60)
print("STEP 1 │ 업종별 K-Means 클러스터링 (Method 1)")
print("━" * 60)

# 결과 컬럼 초기화 — -1 은 미할당 표식
df["업종내_군집"] = -1
for upjong, k in BEST_K.items():
    # 해당 업종의 행 인덱스 추출
    idx = df[df["업종"] == upjong].index
    # 위험변수만 표준화 (스케일 차이 보정)
    X = StandardScaler().fit_transform(df.loc[idx, RISK_VARS])
    # K-Means 학습 + 라벨 예측 — n_init=10 이면 초기화 10번 후 최적
    lbl = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)
    df.loc[idx, "업종내_군집"] = lbl
    # 군집별 할당 개수 요약
    sizes = dict(zip(*np.unique(lbl, return_counts=True)))
    print(f"  [{upjong}]  K={k}  분포: { {f'군집{k}': v for k, v in sizes.items()} }")

# ══════════════════════════════════════════════════════════════
# STEP 2 │ 앙상블 분류
# ══════════════════════════════════════════════════════════════
print("\n" + "━" * 60)
print("STEP 2 │ 앙상블 분류  LightGBM · XGBoost · CatBoost · Stacking")
print("━" * 60)


def run_ensemble(sub, feat_cols, target_col):
    """
    부분 데이터셋(sub)에 대해 4종 분류 모델 학습/평가를 수행.

    인자:
        sub : 업종별 부분 DataFrame
        feat_cols : 입력 피처 목록
        target_col : 클래스 라벨 컬럼명 ('업종내_군집')

    반환:
        results (dict) : 모델명 → (학습된 모델, 테스트 정확도)
        top3 (list)    : LGBM 기준 중요변수 상위 3개
    """
    # 입력 표준화 + 정수 라벨로 변환
    X = StandardScaler().fit_transform(sub[feat_cols])
    y = sub[target_col].astype(int).values
    # 학습:테스트 = 8:2, 클래스 비율 유지(stratify)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # LightGBM — 빠르고 leaf-wise 분기
    lgbm = LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1
    )
    # XGBoost — 안정적, 깊이 5
    xgb = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        random_state=42,
        eval_metric="mlogloss",
        verbosity=0,
    )
    # CatBoost — 범주형 친화적, 기본 깊이 5
    cat = CatBoostClassifier(
        iterations=300, learning_rate=0.05, depth=5, random_state=42, verbose=0
    )
    # Stacking — 위 3개를 1차 학습기로, LR 을 메타학습기로
    stack = StackingClassifier(
        estimators=[
            (
                "lgbm",
                LGBMClassifier(
                    n_estimators=300, learning_rate=0.05, random_state=42, verbose=-1
                ),
            ),
            (
                "xgb",
                XGBClassifier(
                    n_estimators=300,
                    learning_rate=0.05,
                    random_state=42,
                    eval_metric="mlogloss",
                    verbosity=0,
                ),
            ),
            (
                "cat",
                CatBoostClassifier(
                    iterations=300, learning_rate=0.05, random_state=42, verbose=0
                ),
            ),
        ],
        final_estimator=LogisticRegression(max_iter=1000),
        cv=5,
        passthrough=False,
    )

    # 4개 모델 모두 학습 + 테스트 정확도 측정
    results = {}
    for name, clf in [
        ("LightGBM", lgbm),
        ("XGBoost", xgb),
        ("CatBoost", cat),
        ("Stacking", stack),
    ]:
        clf.fit(Xtr, ytr)
        acc = accuracy_score(yte, clf.predict(Xte))
        results[name] = (clf, acc)

    # LGBM 의 변수 중요도 — 해석용 상위 3개
    fi = dict(zip(feat_cols, lgbm.feature_importances_))
    top3 = sorted(fi.items(), key=lambda x: -x[1])[:3]
    return results, top3


# 업종별로 분류 실행 — 표본 수와 모델 성능 출력
for upjong, k in BEST_K.items():
    sub = df[df["업종"] == upjong].dropna(subset=FEAT_COLS + ["업종내_군집"])
    print(f"\n  [{upjong}]  K={k}  N={len(sub)}")
    res, top3 = run_ensemble(sub, FEAT_COLS, "업종내_군집")
    for name, (clf, acc) in res.items():
        print(f"    {name:<12}  test_acc = {acc:.4f}")
    print(f"    LGBM 중요변수 top3: {', '.join([f'{k}({v:.0f})' for k, v in top3])}")

# ══════════════════════════════════════════════════════════════
# STEP 3 │ Ridge / Lasso 회귀
# ══════════════════════════════════════════════════════════════
print("\n" + "━" * 60)
print("STEP 3 │ Ridge / Lasso 회귀  (Y = AHP 위험점수)")
print("━" * 60)

# 업종을 더미변수화 — 첫 카테고리는 기준 범주(drop_first)
dummies = pd.get_dummies(df["업종"], drop_first=True, dtype=int)
# 회귀용 설명변수 + 종속변수 결측 제거
df_reg = pd.concat([df[RISK_VARS + ["위험점수_AHP"]], dummies], axis=1).dropna()
REG_FEAT = RISK_VARS + list(dummies.columns)

# 표준화한 X와 원래 스케일의 y
Xr = StandardScaler().fit_transform(df_reg[REG_FEAT])
yr = df_reg["위험점수_AHP"].values

# Ridge: L2 규제, alpha 그리드서치(CV)
ridge = RidgeCV(alphas=[0.001, 0.01, 0.1, 1, 10, 100]).fit(Xr, yr)
# Lasso: L1 규제 + CV — 자동 변수 선택 효과
lasso = LassoCV(cv=5, max_iter=10000, random_state=42).fit(Xr, yr)

print(f"\n  Ridge  α={ridge.alpha_:.4f}   R²={ridge.score(Xr, yr):.4f}")
print(f"  Lasso  α={lasso.alpha_:.6f}  R²={lasso.score(Xr, yr):.4f}")

# Ridge 계수를 절대값 기준 내림차순으로 출력 — 막대 시각화 풍
print("\n  ── Ridge 계수 (표준화) ──")
for feat, coef in sorted(zip(REG_FEAT, ridge.coef_), key=lambda x: -abs(x[1])):
    bar = "█" * int(abs(coef) * 20)
    sign = "+" if coef >= 0 else "-"
    print(f"    {feat:<25} {coef:+.4f}  {sign}{bar}")

# Lasso — 0이 아닌 계수만 출력 (선택된 변수)
print("\n  ── Lasso 비제로 계수 ──")
for feat, coef in sorted(zip(REG_FEAT, lasso.coef_), key=lambda x: -abs(x[1])):
    if abs(coef) > 1e-6:
        print(f"    {feat:<25} {coef:+.4f}")

# OLS 잔차 직접 계산 (statsmodels 의존 회피) — 다음 단계 모란I 입력
Xr_const = np.column_stack([np.ones(len(Xr)), Xr])  # 절편항 추가
coef_ols, *_ = np.linalg.lstsq(Xr_const, yr, rcond=None)
resid = yr - Xr_const @ coef_ols  # 관측 - 적합 = 잔차

# ══════════════════════════════════════════════════════════════
# STEP 4 │ Moran's I
# ══════════════════════════════════════════════════════════════
print("\n" + "━" * 60)
print("STEP 4 │ Moran's I (OLS 잔차 공간자기상관)")
print("━" * 60)

# 회귀 표본의 좌표만 추출
coords_r = df.loc[df_reg.index, GEO_VARS].values
# K=8 최근접 이웃으로 공간가중치 — 행기준 표준화
w_r = KNN.from_array(coords_r, k=8)
w_r.transform = "R"

# 모란 I — 999 회 순열로 유의성 검정
mi = Moran(resid, w_r, permutations=999)
print(f"\n  Moran's I = {mi.I:+.4f}")
print(f"  E[I]      = {mi.EI:.4f}")
print(f"  z-score   = {mi.z_sim:.4f}")
print(
    f"  p-value   = {mi.p_sim:.4f}  ({'유의 → Spatial 모델 필요' if mi.p_sim < 0.05 else '비유의'} )"
)

# ══════════════════════════════════════════════════════════════
# STEP 5 │ Spatial Lag / Spatial Error
# ══════════════════════════════════════════════════════════════
print("\n" + "━" * 60)
print("STEP 5 │ Spatial Lag · Spatial Error 모델")
print("━" * 60)

# 공간회귀 입력 준비 — y는 (n,1) 모양 필수
coords_sp = df.loc[df_reg.index, GEO_VARS].values
y_sp = yr.reshape(-1, 1)
X_sp = np.column_stack([np.ones(len(Xr)), Xr])  # 절편 포함
w_sp = KNN.from_array(coords_sp, k=8)
w_sp.transform = "R"
var_names = ["CONST"] + REG_FEAT

# Spatial Lag (SAR) — y = ρWy + Xβ + ε
try:
    lag = ML_Lag(y_sp, X_sp, w=w_sp, name_y="AHP위험점수", name_x=var_names)
    print("\n  Spatial Lag")
    print(f"    ρ (공간시차)  = {lag.rho:+.4f}")
    print(f"    Pseudo-R²    = {lag.pr2:.4f}")
    print(f"    AIC          = {lag.aic:.2f}")
    print(f"    Log-lik      = {lag.logll:.2f}")
except Exception as e:
    print(f"  Spatial Lag 오류: {e}")

# Spatial Error (SEM) — y = Xβ + u, u = λWu + ε
try:
    err = ML_Error(y_sp, X_sp, w=w_sp, name_y="AHP위험점수", name_x=var_names)
    print("\n  Spatial Error")
    print(f"    λ (공간오차)  = {err.lam:+.4f}")
    print(f"    Pseudo-R²    = {err.pr2:.4f}")
    print(f"    AIC          = {err.aic:.2f}")
    print(f"    Log-lik      = {err.logll:.2f}")
except Exception as e:
    print(f"  Spatial Error 오류: {e}")

# ══════════════════════════════════════════════════════════════
# STEP 6 │ GWR
# ══════════════════════════════════════════════════════════════
print("\n" + "━" * 60)
print("STEP 6 │ GWR (Geographically Weighted Regression)")
print("━" * 60)

# 전체 사용 시 수 분 소요 → 계산 시간 단축을 위해 2000개 랜덤 샘플링
N_GWR = 2000
gwr_idx = np.random.RandomState(42).choice(
    len(df_reg), min(N_GWR, len(df_reg)), replace=False
)
coords_gw = coords_sp[gwr_idx]
y_gw = yr[gwr_idx].reshape(-1, 1)
# GWR 입력은 표준화된 위험변수만 사용 (더미 제외) — 해석 단순화
X_gw = StandardScaler().fit_transform(df_reg.iloc[gwr_idx][RISK_VARS].values)

print(f"\n  샘플 수: {len(gwr_idx):,}  변수: {RISK_VARS}")
try:
    # bisquare 커널 + adaptive NN — 점밀도 따라 bandwidth 자동 조절
    bw_sel = Sel_BW(coords_gw, y_gw, X_gw, kernel="bisquare", fixed=False)
    bw = bw_sel.search(search_method="golden_section")
    print(f"  최적 bandwidth (adaptive NN): {int(bw)}")

    # 본 GWR 적합
    gwr_res = GWR(coords_gw, y_gw, X_gw, bw=bw, kernel="bisquare", fixed=False).fit()
    print(f"  R²      = {gwr_res.R2:.4f}")
    print(f"  adj.R²  = {gwr_res.adj_R2:.4f}")
    # mgwr 버전마다 속성명이 다른 경우 호환 처리
    aic_val = getattr(gwr_res, "AIC", None) or getattr(gwr_res, "aicc", None)
    if aic_val:
        print(f"  AICc    = {aic_val:.2f}")

    # 변수별 지역 계수 분포 요약 — 평균/표준편차/최소/최대
    print("\n  ── 지역별 계수 변동 (평균 ± 표준편차) ──")
    for i, var in enumerate(RISK_VARS):
        # 0번 인덱스는 절편이므로 i+1
        p = gwr_res.params[:, i + 1]
        print(
            f"    {var:<20}  mean={p.mean():+.4f}  std={p.std():.4f}"
            f"  [{p.min():+.4f} ~ {p.max():+.4f}]"
        )

    # ── GWR 결과 CSV 저장 — 좌표/local R²/계수/t-값 ──
    gwr_out = pd.DataFrame(
        {
            "위도": coords_gw[:, 0],
            "경도": coords_gw[:, 1],
            "local_R2": gwr_res.localR2.flatten(),
        }
    )
    for i, var in enumerate(RISK_VARS):
        gwr_out[f"coef_{var}"] = gwr_res.params[:, i + 1]
        gwr_out[f"tval_{var}"] = gwr_res.tvalues[:, i + 1]
    gwr_out.to_csv("data/gwr_results.csv", index=False, encoding="utf-8-sig")
    print(f"\n  GWR 결과 저장: data/gwr_results.csv  ({len(gwr_out)}행)")

except Exception as e:
    print(f"  GWR 오류: {e}")

print("\n" + "━" * 60)
print("전체 파이프라인 완료")
print("━" * 60)
