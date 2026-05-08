# -*- coding: utf-8 -*-
"""
K=2 공간회귀 통합 파이프라인.

목적:
    cluster3 산출물에 fire_count_150m 타겟을 결합한 뒤,
    여러 후보 특성 집합 중 실루엣이 가장 높은 조합으로 K=2 KMeans 군집을 만들고,
    각 군집에서 OLS / SLM / SEM / GWR / MGWR 모형을 적합하여 비교한다.
    동시에 군집 프로파일 히트맵, 모델 성능 비교 그래프를 PNG로 저장한다.

산출물:
    OUT_DIR/최최최종0428변수테이블_cluster_k2.csv
    OUT_DIR/k2_cluster_feature_set_tuning.csv
    OUT_DIR/cluster_k2_feature_summary.csv / .png
    OUT_DIR/spatial_model_summary_by_cluster_k2.csv
    OUT_DIR/slm_sem_knn_tuning_by_cluster_k2.csv
    OUT_DIR/ols_coefficients_by_cluster_k2.csv
    OUT_DIR/gwr_local_diagnostics_by_cluster_k2.csv
    OUT_DIR/mgwr_bandwidth_by_variable_k2.csv
    OUT_DIR/cluster_k2_model_performance.png
    OUT_DIR/metadata.json
"""

from __future__ import annotations

# 메타데이터 직렬화
import json
# 모델별 소요시간 측정
import time
# spreg/mgwr이 출력하는 경고를 끄기 위함
import warnings
from pathlib import Path

import matplotlib

# 그래프는 파일로만 저장 -> Agg 백엔드
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
# OLS 회귀 (HC3 표준오차 사용 위해 statsmodels)
import statsmodels.api as sm
# Global Moran's I (잔차 공간자기상관)
from esda.moran import Moran
# K-최근접 공간 가중치
from libpysal.weights import KNN
# 지리가중회귀 (GWR/MGWR) — bandwidth 선택기 + 본 모델
from mgwr.gwr import GWR, MGWR
from mgwr.sel_bw import Sel_BW
# 군집화 + 군집 품질 지표
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.preprocessing import StandardScaler
# 공간 시차 / 공간 오차 모델 (ML 추정)
from spreg import ML_Error, ML_Lag


# 라이브러리 내부 경고 메시지 숨김 (출력 노이즈 ↓)
warnings.filterwarnings("ignore")

# 프로젝트 경로 정의
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "0424" / "data" / "cluster3_spatial_pipeline_fire_count_150m_0428"
FIRE_TARGET_PATH = (
    ROOT / "data" / "team_pipeline_validation" / "team_pipeline_scored_dataset.csv"
)
OUT_DIR = ROOT / "0429" / "cluster2_spatial_pipeline_fire_count_150m_0429"

# 종속변수 / 군집 라벨 컬럼 / 좌표 컬럼
TARGET = "fire_count_150m"
CLUSTER_COL = "cluster_k2"
COORD_COLS = ["x_5181", "y_5181"]
# 회귀에 들어갈 10개 독립 변수 (전체 후보)
REG_FEATURES = [
    "승인연도",
    "소방위험도_점수",
    "주변건물수",
    "집중도",
    "단속위험도",
    "구조노후도",
    "도로폭위험도",
    "최근접_소화용수_거리등급",
    "총층수",
    "연면적",
]

# K=2 군집화 후보 특성 집합 — 실루엣 비교 후 최적 1개를 선택
CLUSTER_FEATURE_SETS = {
    "all_10_original": REG_FEATURES,
    "discriminative_6": [
        "단속위험도",
        "도로폭위험도",
        "집중도",
        "주변건물수",
        "최근접_소화용수_거리등급",
        "소방위험도_점수",
    ],
    "policy_5": [
        "도로폭위험도",
        "집중도",
        "주변건물수",
        "최근접_소화용수_거리등급",
        "소방위험도_점수",
    ],
    "risk_score_plus_5": [
        "최종_화재위험점수",
        "도로폭위험도",
        "집중도",
        "주변건물수",
        "최근접_소화용수_거리등급",
    ],
}

# KNN k 후보 — SLM/SEM에서 가중치 행렬 K값 튜닝
KNN_CANDIDATES = [6, 8, 10, 12, 15, 20]
# Moran I p값 안정화를 위한 순열 횟수
MORAN_PERMUTATIONS = 199
# GWR/MGWR은 계산량이 크므로 표본 상한을 둠
GWR_SAMPLE_CAP = 700
MGWR_SAMPLE_CAP = 220
# 표본 추출 재현성 확보용 RNG
RNG = np.random.RandomState(42)


def set_korean_font() -> None:
    """그래프에 한글이 깨지지 않도록 윈도우 한글 폰트로 설정."""
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.family"] = "Malgun Gothic"


def read_main_csv() -> pd.DataFrame:
    """DATA_DIR에서 가장 큰 CSV(가장 완전한 변수 테이블)를 로드."""
    csv_files = sorted(
        DATA_DIR.glob("*.csv"), key=lambda p: p.stat().st_size, reverse=True
    )
    if not csv_files:
        raise FileNotFoundError(DATA_DIR)
    return pd.read_csv(csv_files[0], encoding="utf-8-sig")


def name_key(s: pd.Series) -> pd.Series:
    """병합 키용 정규화 — 공백 제거 + strip."""
    return s.astype(str).str.replace(r"\s+", "", regex=True).str.strip()


def attach_fire_count(df: pd.DataFrame) -> pd.DataFrame:
    """fire_count_150m이 없으면 외부 검증 데이터에서 매칭해 부착."""
    if TARGET in df.columns:
        return df
    fire = pd.read_csv(FIRE_TARGET_PATH, encoding="utf-8-sig")
    # 매칭 키: 정규화 숙소명 + 위경도 6자리 반올림
    fire_key = pd.DataFrame(
        {
            "_name_key": name_key(fire["숙소명"]),
            "_lat_key": pd.to_numeric(fire["위도"], errors="coerce").round(6),
            "_lon_key": pd.to_numeric(fire["경도"], errors="coerce").round(6),
            TARGET: pd.to_numeric(fire[TARGET], errors="coerce"),
        }
    ).drop_duplicates(["_name_key", "_lat_key", "_lon_key"])
    keyed = df.copy()
    keyed["_name_key"] = name_key(keyed["숙소명"])
    keyed["_lat_key"] = pd.to_numeric(keyed["위도"], errors="coerce").round(6)
    keyed["_lon_key"] = pd.to_numeric(keyed["경도"], errors="coerce").round(6)
    keyed = keyed.merge(fire_key, on=["_name_key", "_lat_key", "_lon_key"], how="left")
    # 임시 키 컬럼 제거
    return keyed.drop(columns=["_name_key", "_lat_key", "_lon_key"], errors="ignore")


def prepare_data() -> pd.DataFrame:
    """변수 테이블에 타겟 부착, 필요한 컬럼만 골라 수치 변환 + 결측 제거."""
    df = attach_fire_count(read_main_csv())
    needed = [
        "구",
        "동",
        "숙소명",
        "경도",
        "위도",
        TARGET,
        *COORD_COLS,
        *REG_FEATURES,
        "최종_화재위험점수",
    ]
    # 컬럼이 부분적으로 없을 수 있으니 존재하는 것만 선택
    df = df[[c for c in needed if c in df.columns]].copy()
    # 수치형 강제 변환 (회귀/공간모델 입력 준비)
    for col in [TARGET, *COORD_COLS, *REG_FEATURES, "최종_화재위험점수"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # 핵심 컬럼 결측 행은 제거
    return df.dropna(subset=[TARGET, *COORD_COLS, *REG_FEATURES]).reset_index(drop=True)


def build_k2_clusters(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """후보 특성집합 별 K=2 군집을 적합 -> 실루엣 최고 조합으로 라벨 부여.

    반환:
        out: cluster_k2 컬럼이 추가된 df
        tuning: 후보별 silhouette/CH/표본수 비교 표
        best_name: 최종 채택 특성집합 이름
    """
    rows = []
    fitted = {}
    for name, features in CLUSTER_FEATURE_SETS.items():
        # 해당 특성에 결측이 있는 행은 제외 (특성집합마다 표본 다를 수 있음)
        work = df.dropna(subset=features).copy()
        x = StandardScaler().fit_transform(work[features].to_numpy(dtype=float))
        # K=2 KMeans (50회 초기화로 안정적 결과)
        labels = KMeans(n_clusters=2, random_state=42, n_init=50).fit_predict(x)
        # 군집 품질 두 지표
        sil = silhouette_score(x, labels)
        ch = calinski_harabasz_score(x, labels)
        rows.append(
            {
                "feature_set": name,
                "n_features": len(features),
                "features": ", ".join(features),
                "silhouette": sil,
                "calinski_harabasz": ch,
                "cluster0_n": int((labels == 0).sum()),
                "cluster1_n": int((labels == 1).sum()),
            }
        )
        # 라벨은 후보별로 보관해 나중에 최적 항목만 사용
        fitted[name] = labels
    # 실루엣 우선, 동률이면 CH로 정렬해 최상위 채택
    tuning = pd.DataFrame(rows).sort_values(
        ["silhouette", "calinski_harabasz"], ascending=False
    )
    best_name = str(tuning.iloc[0]["feature_set"])
    out = df.copy()
    out[CLUSTER_COL] = fitted[best_name]
    return out, tuning, best_name


def standardize_x(df: pd.DataFrame) -> np.ndarray:
    """회귀 X 행렬 — 표준화된 numpy 배열로 변환."""
    return StandardScaler().fit_transform(df[REG_FEATURES].to_numpy(dtype=float))


def build_weights(coords: np.ndarray, k: int) -> KNN:
    """KNN 가중치 행렬 (n이 k+1보다 작으면 안전 축소) + 행 표준화."""
    kk = min(k, max(1, len(coords) - 1))
    w = KNN.from_array(coords, k=kk)
    w.transform = "r"
    return w


def aicc_of(result) -> float:
    """모델 객체에서 AICc/AIC를 안전하게 추출 (속성명이 버전마다 다름)."""
    for attr in ("aicc", "AICc", "aic", "AIC"):
        try:
            return float(getattr(result, attr))
        except Exception:
            pass
    return float("nan")


def sample_for_local_model(df: pd.DataFrame, cap: int) -> pd.DataFrame:
    """국지 모형(GWR/MGWR)이 다루기 어려운 표본 크기를 cap으로 축소."""
    if len(df) <= cap:
        return df.copy().reset_index(drop=True)
    # 인덱스 무작위 추출 (seed 고정 RNG 사용)
    sampled_idx = RNG.choice(df.index.to_numpy(), cap, replace=False)
    # 정렬 후 반환 — 후속 처리에서 위치 의존 안정성 확보
    return df.loc[np.sort(sampled_idx)].copy().reset_index(drop=True)


def run_ols(df: pd.DataFrame, cluster_id: int) -> tuple[dict, pd.DataFrame]:
    """OLS 적합 + KNN_CANDIDATES별 잔차 Moran I 측정 -> 최적 행 반환.

    반환:
        best: |Moran I| 가장 작은 KNN k 시점의 진단/적합 요약 (dict)
        coef: 변수별 계수/p-value 표
    """
    x = standardize_x(df)
    y = df[TARGET].to_numpy(dtype=float)
    coords = df[COORD_COLS].to_numpy(dtype=float)
    # HC3 강건 표준오차로 OLS 적합 (이분산 강건성)
    model = sm.OLS(y, sm.add_constant(x)).fit(cov_type="HC3")
    rows = []
    # KNN k별로 잔차 공간자기상관 점검
    for k in KNN_CANDIDATES:
        w = build_weights(coords, k)
        moran = Moran(model.resid, w, permutations=MORAN_PERMUTATIONS)
        rows.append(
            {
                "cluster": cluster_id,
                "model": "OLS",
                "knn_k": k,
                "n": len(df),
                "fit": float(model.rsquared),
                "adj_fit": float(model.rsquared_adj),
                "aic": float(model.aic),
                "resid_moran_I": float(moran.I),
                "resid_moran_p": float(moran.p_sim),
                "status": "ok",
            }
        )
    # 잔차의 공간자기상관이 가장 작은 K를 대표 행으로 선택
    best = min(rows, key=lambda r: abs(r["resid_moran_I"]))
    coef = pd.DataFrame(
        {
            "cluster": cluster_id,
            "term": ["const", *REG_FEATURES],
            "coef": model.params,
            "p_value": model.pvalues,
        }
    )
    return best, coef


def run_spatial_family(df: pd.DataFrame, cluster_id: int) -> pd.DataFrame:
    """SLM/SEM × KNN_CANDIDATES 조합별 적합 결과 표 생성."""
    x = standardize_x(df)
    y = df[TARGET].to_numpy(dtype=float).reshape(-1, 1)
    coords = df[COORD_COLS].to_numpy(dtype=float)
    rows = []
    # 두 모델 패밀리 순회
    for model_name, model_cls in [("SLM", ML_Lag), ("SEM", ML_Error)]:
        for k in KNN_CANDIDATES:
            t0 = time.time()
            try:
                w = build_weights(coords, k)
                model = model_cls(y, x, w=w, name_y=TARGET, name_x=REG_FEATURES)
                # 잔차 평탄화 후 Moran I
                resid = np.asarray(model.u).flatten()
                moran = Moran(resid, w, permutations=MORAN_PERMUTATIONS)
                # SLM은 rho, SEM은 lambda를 보고
                rho_or_lambda = np.nan
                if model_name == "SLM":
                    rho_or_lambda = float(np.asarray(model.rho).reshape(-1)[0])
                elif hasattr(model, "lam"):
                    rho_or_lambda = float(np.asarray(model.lam).reshape(-1)[0])
                rows.append(
                    {
                        "cluster": cluster_id,
                        "model": model_name,
                        "knn_k": k,
                        "n": len(df),
                        # pr2 우선, 없으면 r2 — 버전별 속성 차이 대응
                        "fit": float(
                            getattr(model, "pr2", getattr(model, "r2", np.nan))
                        ),
                        "adj_fit": np.nan,
                        "aic": float(getattr(model, "aic", np.nan)),
                        "rho_or_lambda": rho_or_lambda,
                        "resid_moran_I": float(moran.I),
                        "resid_moran_p": float(moran.p_sim),
                        "seconds": round(time.time() - t0, 2),
                        "status": "ok",
                    }
                )
            except Exception as exc:
                # 실패해도 표에 빈 행을 남겨 추후 디버깅
                rows.append(
                    {
                        "cluster": cluster_id,
                        "model": model_name,
                        "knn_k": k,
                        "n": len(df),
                        "fit": np.nan,
                        "adj_fit": np.nan,
                        "aic": np.nan,
                        "rho_or_lambda": np.nan,
                        "resid_moran_I": np.nan,
                        "resid_moran_p": np.nan,
                        "seconds": round(time.time() - t0, 2),
                        "status": f"failed: {exc}",
                    }
                )
    return pd.DataFrame(rows)


def select_best_spatial(rows: pd.DataFrame) -> pd.DataFrame:
    """SLM/SEM 각 모델별로 AIC가 가장 작은 K행을 1개 선택."""
    ok = rows[rows["status"].eq("ok")].copy()
    if ok.empty:
        # 모두 실패한 경우라도 모델당 1행 보장
        return rows.groupby("model", as_index=False).head(1)
    best_rows = []
    for model, sub in ok.groupby("model"):
        # AIC 우선, 동률이면 잔차 Moran I 작은 쪽 우선
        sub = sub.sort_values(["aic", "resid_moran_I"], ascending=[True, True])
        best_rows.append(sub.iloc[0])
    return pd.DataFrame(best_rows)


def select_gwr_bw(coords: np.ndarray, y: np.ndarray, x: np.ndarray) -> float:
    """GWR 단일 bandwidth 탐색 (golden section)."""
    # bw 하한: 변수 수 + 3 또는 30 중 큰 값
    bw_min = max(30, x.shape[1] + 3)
    # bw 상한: 표본의 거의 끝까지 — 단, 420 이하로 제한 (시간 절약)
    bw_max = max(bw_min + 2, min(len(y) - 1, 420))
    selector = Sel_BW(coords, y, x, fixed=False, kernel="bisquare", n_jobs=1)
    return float(
        selector.search(search_method="golden_section", bw_min=bw_min, bw_max=bw_max)
    )


def run_gwr(df: pd.DataFrame, cluster_id: int) -> tuple[dict, pd.DataFrame]:
    """클러스터 단위 GWR 적합 + 잔차 Moran I + 국지 R² 테이블 생성."""
    # 표본 상한 적용
    work = sample_for_local_model(df, GWR_SAMPLE_CAP)
    coords = work[COORD_COLS].to_numpy(dtype=float)
    y = work[TARGET].to_numpy(dtype=float).reshape(-1, 1)
    x = standardize_x(work)
    t0 = time.time()
    try:
        # bandwidth 탐색 -> 본 GWR 적합
        bw = select_gwr_bw(coords, y, x)
        result = GWR(
            coords, y, x, bw=bw, fixed=False, kernel="bisquare", n_jobs=1
        ).fit()
        # 잔차 평탄화
        resid = np.asarray(result.resid_response).flatten()
        # GWR 잔차에 대해 KNN k=12 기준 Moran I 측정
        w = build_weights(coords, 12)
        moran = Moran(resid, w, permutations=MORAN_PERMUTATIONS)
        # 시설별 국지 R²와 잔차를 같은 좌표와 함께 저장
        local = pd.DataFrame(
            {
                "cluster": cluster_id,
                "x_5181": coords[:, 0],
                "y_5181": coords[:, 1],
                "local_R2": np.asarray(result.localR2).flatten(),
                "residual": resid,
            }
        )
        summary = {
            "cluster": cluster_id,
            "model": "GWR",
            "knn_k": 12,
            "n": len(work),
            "sampled": len(work) < len(df),
            "bandwidth": bw,
            "fit": float(result.R2),
            "adj_fit": float(result.adj_R2),
            "aic": aicc_of(result),
            "rho_or_lambda": np.nan,
            "resid_moran_I": float(moran.I),
            "resid_moran_p": float(moran.p_sim),
            "seconds": round(time.time() - t0, 2),
            "status": "ok",
        }
        return summary, local
    except Exception as exc:
        # 실패 케이스도 NaN으로 일관 기록
        return {
            "cluster": cluster_id,
            "model": "GWR",
            "knn_k": 12,
            "n": len(work),
            "sampled": len(work) < len(df),
            "bandwidth": np.nan,
            "fit": np.nan,
            "adj_fit": np.nan,
            "aic": np.nan,
            "rho_or_lambda": np.nan,
            "resid_moran_I": np.nan,
            "resid_moran_p": np.nan,
            "seconds": round(time.time() - t0, 2),
            "status": f"failed: {exc}",
        }, pd.DataFrame()


def run_mgwr(df: pd.DataFrame, cluster_id: int) -> tuple[dict, pd.DataFrame]:
    """MGWR 적합 (변수별 bandwidth) + 잔차 Moran I + 변수별 bw 테이블 반환."""
    # MGWR은 GWR보다 더 무거우므로 더 강한 표본 상한
    work = sample_for_local_model(df, MGWR_SAMPLE_CAP)
    coords = work[COORD_COLS].to_numpy(dtype=float)
    y = work[TARGET].to_numpy(dtype=float).reshape(-1, 1)
    x = standardize_x(work)
    t0 = time.time()
    try:
        # 변수별 bandwidth 탐색 (multi=True)
        selector = Sel_BW(
            coords, y, x, multi=True, fixed=False, kernel="bisquare", n_jobs=1
        )
        selector.search(
            multi_bw_min=[max(30, x.shape[1] + 3)],
            multi_bw_max=[min(len(work) - 1, 180)],
            max_iter_multi=15,
            verbose=False,
        )
        # 최종 MGWR 적합
        result = MGWR(
            coords, y, x, selector, fixed=False, kernel="bisquare", n_jobs=1
        ).fit()
        # 변수별 bandwidth 평탄화
        bw_values = np.asarray(selector.bw[0]).flatten()
        resid = np.asarray(result.resid_response).flatten()
        w = build_weights(coords, 12)
        moran = Moran(resid, w, permutations=MORAN_PERMUTATIONS)
        # 변수명과 매칭하여 표 작성 (intercept 포함, 길이 보호)
        bw_table = pd.DataFrame(
            {
                "cluster": cluster_id,
                "term": ["intercept", *REG_FEATURES],
                "bandwidth": bw_values[: len(REG_FEATURES) + 1],
            }
        )
        summary = {
            "cluster": cluster_id,
            "model": "MGWR",
            "knn_k": 12,
            "n": len(work),
            "sampled": len(work) < len(df),
            # 대표 bandwidth는 변수별 평균 (요약용)
            "bandwidth": float(np.nanmean(bw_values)),
            "fit": float(result.R2),
            "adj_fit": float(result.adj_R2),
            "aic": aicc_of(result),
            "rho_or_lambda": np.nan,
            "resid_moran_I": float(moran.I),
            "resid_moran_p": float(moran.p_sim),
            "seconds": round(time.time() - t0, 2),
            "status": "ok",
        }
        return summary, bw_table
    except Exception as exc:
        return {
            "cluster": cluster_id,
            "model": "MGWR",
            "knn_k": 12,
            "n": len(work),
            "sampled": len(work) < len(df),
            "bandwidth": np.nan,
            "fit": np.nan,
            "adj_fit": np.nan,
            "aic": np.nan,
            "rho_or_lambda": np.nan,
            "resid_moran_I": np.nan,
            "resid_moran_p": np.nan,
            "seconds": round(time.time() - t0, 2),
            "status": f"failed: {exc}",
        }, pd.DataFrame()


def summarize_clusters(df: pd.DataFrame) -> pd.DataFrame:
    """군집별 핵심 변수의 count/mean/median/std 요약 표."""
    cols = [
        TARGET,
        "최종_화재위험점수",
        "구조노후도",
        "단속위험도",
        "도로폭위험도",
        "집중도",
        "주변건물수",
        "최근접_소화용수_거리등급",
        "소방위험도_점수",
    ]
    summary = df.groupby(CLUSTER_COL)[cols].agg(["count", "mean", "median", "std"])
    # 멀티컬럼 -> 평탄화 (변수_통계 형태)
    summary.columns = ["_".join(c).strip() for c in summary.columns]
    return summary.reset_index()


def save_cluster_profile_png(cluster_summary: pd.DataFrame, out_path: Path) -> None:
    """군집 평균 기반 프로파일 히트맵 PNG."""
    profile_cols = [
        "최종_화재위험점수_mean",
        "fire_count_150m_mean",
        "도로폭위험도_mean",
        "집중도_mean",
        "주변건물수_mean",
        "최근접_소화용수_거리등급_mean",
        "소방위험도_점수_mean",
        "구조노후도_mean",
    ]
    raw = cluster_summary.set_index(CLUSTER_COL)[profile_cols]
    # 시각화용 라벨 매핑 (컬럼명 단축)
    labels = {
        "최종_화재위험점수_mean": "최종위험",
        "fire_count_150m_mean": "150m화재",
        "도로폭위험도_mean": "도로폭",
        "집중도_mean": "집중도",
        "주변건물수_mean": "주변건물",
        "최근접_소화용수_거리등급_mean": "소화용수",
        "소방위험도_점수_mean": "소방위험",
        "구조노후도_mean": "구조노후",
    }
    # 색은 전체 평균 대비 % 차이 — ±80% 클립
    relative = raw.copy()
    for col in relative.columns:
        mean = relative[col].mean()
        if np.isclose(mean, 0):
            relative[col] = 0
        else:
            relative[col] = ((relative[col] - mean) / abs(mean) * 100).clip(-80, 80)
    relative.columns = [labels[c] for c in relative.columns]

    fig, ax = plt.subplots(figsize=(11.8, 4.8), dpi=180)
    sns.heatmap(
        relative,
        cmap=sns.diverging_palette(220, 20, as_cmap=True),
        center=0,
        vmin=-80,
        vmax=80,
        # 셀 표시값은 원자료 평균(원래 단위)
        annot=raw.rename(columns=labels).round(2),
        fmt=".2f",
        linewidths=1.2,
        linecolor="white",
        cbar_kws={"label": "전체 평균 대비 차이(%)"},
        ax=ax,
    )
    ax.set_title("K=2 군집 핵심 특징 프로파일", fontsize=17, weight="bold", pad=14)
    ax.set_xlabel("")
    ax.set_ylabel("Cluster")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_model_png(model_summary: pd.DataFrame, out_path: Path) -> None:
    """OLS/SLM/SEM/GWR/MGWR 의 fit/AIC/잔차 Moran I를 군집별로 비교 그래프."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), dpi=180)
    plot = model_summary.copy()
    # cluster를 카테고리(문자열)로 다뤄 hue 스케일에 안정성 부여
    plot["cluster"] = plot["cluster"].astype(str)

    # (1) fit
    sns.barplot(
        data=plot, x="model", y="fit", hue="cluster", palette="Set2", ax=axes[0]
    )
    axes[0].set_title("모델 설명력")
    axes[0].set_ylabel("R2 / pseudo R2")
    axes[0].set_xlabel("")
    axes[0].grid(axis="y", alpha=0.25)

    # (2) AIC — 낮을수록 좋음
    sns.barplot(
        data=plot, x="model", y="aic", hue="cluster", palette="Set2", ax=axes[1]
    )
    axes[1].set_title("AIC")
    axes[1].set_ylabel("낮을수록 유리")
    axes[1].set_xlabel("")
    axes[1].grid(axis="y", alpha=0.25)

    # (3) 잔차 공간자기상관 (Moran I)
    sns.barplot(
        data=plot,
        x="model",
        y="resid_moran_I",
        hue="cluster",
        palette="Set2",
        ax=axes[2],
    )
    # 0선 표시 — 절대값이 작을수록 잔차에 공간 패턴이 적음
    axes[2].axhline(0, color="#333333", linewidth=1)
    axes[2].set_title("잔차 Moran's I")
    axes[2].set_ylabel("0에 가까울수록 공간잔차 작음")
    axes[2].set_xlabel("")
    axes[2].grid(axis="y", alpha=0.25)

    for ax in axes:
        ax.tick_params(axis="x", rotation=0)
        ax.legend(title="Cluster", fontsize=8)
    fig.suptitle("K=2 공간회귀 모델 성능 비교", fontsize=17, weight="bold", y=1.03)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    """엔드투엔드 실행: 데이터 준비 -> 군집화 -> 모델 적합 -> 산출물 저장."""
    set_korean_font()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) 데이터 준비
    df = prepare_data()
    # 2) 후보 특성 집합 중 최적으로 K=2 군집 부여
    df, cluster_tuning, best_feature_set = build_k2_clusters(df)

    # 군집 라벨이 부여된 표 / 튜닝 결과 저장
    df.to_csv(
        OUT_DIR / "최최최종0428변수테이블_cluster_k2.csv",
        index=False,
        encoding="utf-8-sig",
    )
    cluster_tuning.to_csv(
        OUT_DIR / "k2_cluster_feature_set_tuning.csv", index=False, encoding="utf-8-sig"
    )

    # 3) 군집 요약 통계 + 프로파일 히트맵
    cluster_summary = summarize_clusters(df)
    cluster_summary.to_csv(
        OUT_DIR / "cluster_k2_feature_summary.csv", index=False, encoding="utf-8-sig"
    )
    save_cluster_profile_png(
        cluster_summary, OUT_DIR / "cluster_k2_feature_profile.png"
    )

    # 4) 군집별 OLS/SLM/SEM/GWR/MGWR 적합 결과 누적
    model_rows = []
    coef_tables = []
    all_spatial_tuning = []
    gwr_local_tables = []
    mgwr_bw_tables = []

    for cluster_id in sorted(df[CLUSTER_COL].dropna().astype(int).unique()):
        sub = df[df[CLUSTER_COL].astype(int).eq(cluster_id)].reset_index(drop=True)
        # 진행 상황 콘솔 표시
        print(f"=== K=2 cluster {cluster_id} / n={len(sub):,} ===")
        # OLS — 잔차 Moran I 기준 best K 행 + 변수 계수 표
        ols_summary, coef = run_ols(sub, cluster_id)
        model_rows.append(ols_summary)
        coef_tables.append(coef)

        # SLM/SEM 후보 전체 + 모델별 best 행만 요약에 추가
        spatial_tuning = run_spatial_family(sub, cluster_id)
        all_spatial_tuning.append(spatial_tuning)
        model_rows.extend(select_best_spatial(spatial_tuning).to_dict("records"))

        # GWR 적합 + 국지 R² 테이블
        gwr_summary, gwr_local = run_gwr(sub, cluster_id)
        model_rows.append(gwr_summary)
        if not gwr_local.empty:
            gwr_local_tables.append(gwr_local)

        # MGWR 적합 + 변수별 bandwidth 테이블
        mgwr_summary, mgwr_bw = run_mgwr(sub, cluster_id)
        model_rows.append(mgwr_summary)
        if not mgwr_bw.empty:
            mgwr_bw_tables.append(mgwr_bw)

    # 결과 테이블 모으기
    model_summary = pd.DataFrame(model_rows)
    spatial_tuning = pd.concat(all_spatial_tuning, ignore_index=True)
    coef_df = pd.concat(coef_tables, ignore_index=True)
    gwr_local_df = (
        pd.concat(gwr_local_tables, ignore_index=True)
        if gwr_local_tables
        else pd.DataFrame()
    )
    mgwr_bw_df = (
        pd.concat(mgwr_bw_tables, ignore_index=True)
        if mgwr_bw_tables
        else pd.DataFrame()
    )

    # 5) 결과 CSV들 저장
    model_summary.to_csv(
        OUT_DIR / "spatial_model_summary_by_cluster_k2.csv",
        index=False,
        encoding="utf-8-sig",
    )
    spatial_tuning.to_csv(
        OUT_DIR / "slm_sem_knn_tuning_by_cluster_k2.csv",
        index=False,
        encoding="utf-8-sig",
    )
    coef_df.to_csv(
        OUT_DIR / "ols_coefficients_by_cluster_k2.csv",
        index=False,
        encoding="utf-8-sig",
    )
    gwr_local_df.to_csv(
        OUT_DIR / "gwr_local_diagnostics_by_cluster_k2.csv",
        index=False,
        encoding="utf-8-sig",
    )
    mgwr_bw_df.to_csv(
        OUT_DIR / "mgwr_bandwidth_by_variable_k2.csv", index=False, encoding="utf-8-sig"
    )
    # 모델 성능 비교 PNG
    save_model_png(model_summary, OUT_DIR / "cluster_k2_model_performance.png")

    # 6) 메타데이터 JSON — 재현 가능성 확보
    metadata = {
        "target": TARGET,
        "cluster_column": CLUSTER_COL,
        "best_cluster_feature_set": best_feature_set,
        "cluster_feature_sets": CLUSTER_FEATURE_SETS,
        "regression_features": REG_FEATURES,
        "coordinates": COORD_COLS,
        "knn_candidates": KNN_CANDIDATES,
        "moran_permutations": MORAN_PERMUTATIONS,
        "gwr_sample_cap": GWR_SAMPLE_CAP,
        "mgwr_sample_cap": MGWR_SAMPLE_CAP,
        "note": "K=2 is fixed. Feature-set tuning selects the highest silhouette score. SLM/SEM keep the lowest-AIC KNN candidate per cluster/model.",
    }
    (OUT_DIR / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 콘솔 검증 출력
    print(f"best_feature_set={best_feature_set}")
    print(model_summary.to_string(index=False))
    print(f"saved={OUT_DIR}")


if __name__ == "__main__":
    main()
