# -*- coding: utf-8 -*-
"""
최종 공간분석 파이프라인 — 7단계 일괄 실행 후 분석 산출물 일괄 저장.

목적:
    - 클러스터링/회귀(Ridge/Lasso)/OLS+Moran/공간시차+오차/GWR 요약을 한 번에 실행해
      분석 결과 CSV/JSON 으로 저장 → 발표 자료용.

처리 단계:
    Step1  업종별 KMeans 클러스터링 (K 자동 선택, silhouette 기준)
    Step2  Ridge / Lasso 변수 선택 — 전체 + 업종별
    Step3  OLS + Moran's I (잔차 자기상관 진단)
    Step4  Spatial Lag (SAR) / Spatial Error (SEM)
    Step5  사전 산출 GWR 결과 요약
    Step6  최종 시설 위험 순위 결합
    그리고 사각지대(고립/저밀집) 위험점수 산출 (팀 공식)

입력:
    - 0424/분석/tables/분석변수_최종테이블0423_AHP3등급비교*.csv (보정 우선)
    - data/clustering_result_all.csv (fallback)
    - data/data_with_fire_targets.csv (화재 타깃/소방위험)
    - 0424/data/facility_expected_property_damage_two_stage.csv (있으면 결합)
    - data/gwr_results.csv (있으면 요약)

출력:
    - data/final_spatial_pipeline/  (analysis_dataset, step1~6 CSV, manifest.json)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm  # OLS (HC3 강건 표준오차)
from esda.moran import Moran
from libpysal.weights import KNN
from sklearn.cluster import KMeans
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# spreg 가 환경에 따라 미설치일 수 있어 안전 임포트
try:
    from spreg import ML_Error, ML_Lag
except Exception:  # pragma: no cover - optional environment failure
    ML_Error = None
    ML_Lag = None


# 경로
BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "data" / "final_spatial_pipeline"
OUT.mkdir(parents=True, exist_ok=True)

# 클러스터 입력 — 보정 버전 우선, 없으면 미보정, 그 외 fallback
CLUSTER_SOURCE_REPAIRED = (
    BASE
    / "0424"
    / "분석"
    / "tables"
    / "분석변수_최종테이블0423_AHP3등급비교_주변건물수보정.csv"
)
CLUSTER_SOURCE = (
    BASE / "0424" / "분석" / "tables" / "분석변수_최종테이블0423_AHP3등급비교.csv"
)
CLUSTER_FALLBACK_SOURCE = BASE / "data" / "clustering_result_all.csv"
# 화재 타깃 + 소방위험도 결합용
FIRE_SOURCE = BASE / "data" / "data_with_fire_targets.csv"
# 기대피해액 (선택적 결합 — 있으면 합침)
EXPECTED_DAMAGE = (
    BASE / "0424" / "data" / "facility_expected_property_damage_two_stage.csv"
)
# GWR 요약 입력
GWR_SOURCE = BASE / "data" / "gwr_results.csv"

# 모델 설정
GROUP_COL = "업종"
TARGET_COL = "log1p_반경100m"  # 종속변수: 반경 100m 화재수의 log1p
# 6개 위험 변수 (회귀/공간분석 입력)
RISK_VARS = [
    "소방위험도_점수",
    "주변건물수",
    "집중도",
    "단속위험도",
    "구조노후도",
    "도로폭위험도",
]
# 팀 위험점수 산식 입력 5개 (소방위험도 제외)
TEAM_SCORE_VARS = ["주변건물수", "집중도", "단속위험도", "구조노후도", "도로폭위험도"]
# 업종 정렬 순서 — 보고서 출력 일관성
GROUP_ORDER = ["관광숙박업", "숙박업", "외국인관광도시민박업"]


def name_key(s: pd.Series) -> pd.Series:
    """이름 기반 매칭 키 — 모든 공백 제거 + 좌우 공백 정리."""
    return s.astype(str).str.replace(r"\s+", "", regex=True).str.strip()


def read_csv(path: Path) -> pd.DataFrame:
    """UTF-8 BOM 기본 로드 헬퍼."""
    return pd.read_csv(path, encoding="utf-8-sig")


def merge_sources() -> pd.DataFrame:
    """
    클러스터/화재/기대피해액 등 여러 출처를 (이름+좌표) 키로 결합한 분석 마스터 생성.

    매칭 키: (정규화 시설명, 위도 round(6), 경도 round(6))
    """
    # 보정 → 미보정 → fallback 순으로 클러스터 입력 결정
    if CLUSTER_SOURCE_REPAIRED.exists():
        cluster_source = CLUSTER_SOURCE_REPAIRED
    else:
        cluster_source = (
            CLUSTER_SOURCE if CLUSTER_SOURCE.exists() else CLUSTER_FALLBACK_SOURCE
        )
    cluster = read_csv(cluster_source)
    fire = read_csv(FIRE_SOURCE)

    # 화재 데이터 → 매칭 키 + 필요한 수치 컬럼만 추려 중복 제거
    fire_map = pd.DataFrame(
        {
            "_name_key": name_key(fire["업소명"]),
            "_lat_key": fire["위도"].round(6),
            "_lon_key": fire["경도"].round(6),
            "소방위험도_점수": pd.to_numeric(fire["소방위험도_점수"], errors="coerce"),
            "반경100m_화재수": pd.to_numeric(fire["반경100m_화재수"], errors="coerce"),
            "log1p_반경100m": pd.to_numeric(fire["log1p_반경100m"], errors="coerce"),
        }
    ).drop_duplicates(["_name_key", "_lat_key", "_lon_key"])

    # 클러스터 데이터에 매칭 키 부착
    cluster = cluster.copy()
    cluster["_name_key"] = name_key(cluster["숙소명"])
    cluster["_lat_key"] = cluster["위도"].round(6)
    cluster["_lon_key"] = cluster["경도"].round(6)

    # 좌측 결합 — 클러스터 행 보존, 화재 정보 추가
    merged = cluster.merge(
        fire_map, on=["_name_key", "_lat_key", "_lon_key"], how="left"
    )

    # 기대피해액 — 있을 때만 결합
    if EXPECTED_DAMAGE.exists():
        damage = read_csv(EXPECTED_DAMAGE)
        damage_map = pd.DataFrame(
            {
                "_name_key": name_key(damage["시설명"]),
                "_lat_key": damage["위도"].round(6),
                "_lon_key": damage["경도"].round(6),
                "예상_화재발생확률": damage["예상_화재발생확률"],
                "조건부_예상피해액_백만원": damage["조건부_예상피해액_백만원"],
                "기대피해액_백만원": damage["기대피해액_백만원"],
                "기대피해액_순위": damage["기대피해액_순위"],
            }
        ).drop_duplicates(["_name_key", "_lat_key", "_lon_key"])
        merged = merged.merge(
            damage_map, on=["_name_key", "_lat_key", "_lon_key"], how="left"
        )

    # 분석에 자주 쓰는 컬럼들을 미리 숫자형으로 (이후 모든 단계가 안전해짐)
    merged["집중도"] = pd.to_numeric(merged["집중도"], errors="coerce")
    merged["단속위험도"] = pd.to_numeric(merged["단속위험도"], errors="coerce")
    merged["구조노후도"] = pd.to_numeric(merged["구조노후도"], errors="coerce")
    merged["도로폭위험도"] = pd.to_numeric(merged["도로폭위험도"], errors="coerce")
    merged["주변건물수"] = pd.to_numeric(merged["주변건물수"], errors="coerce")
    merged["x_5181"] = pd.to_numeric(merged["x_5181"], errors="coerce")
    merged["y_5181"] = pd.to_numeric(merged["y_5181"], errors="coerce")

    # 매칭용 임시 키 컬럼 제거
    merged = merged.drop(columns=["_name_key", "_lat_key", "_lon_key"], errors="ignore")
    return merged


def add_team_blindspot_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    팀원 공식: 고립(주변건물 적음) + 저밀집(집중도 낮음) 사각지대 발굴형 위험도 점수.

    가중치:
        고립위험 0.35 + 밀집사각지대 0.20 + 도로폭 0.15 + 노후 0.15 + 단속 0.15
    """
    scored = df.copy()
    for col in TEAM_SCORE_VARS:
        scored[col] = pd.to_numeric(scored[col], errors="coerce")

    # 주변건물수/집중도 가 0 인 행은 데이터 신뢰도 의심 → 점수 산정 시 NaN 처리
    suspicious_zero = scored["주변건물수"].eq(0) | scored["집중도"].eq(0)
    if "주변건물수_보정여부" in scored.columns:
        scored["주변건물수_검증상태"] = scored["주변건물수_보정여부"].fillna("사용")
        scored.loc[suspicious_zero, "주변건물수_검증상태"] = "주변건물/집중도_검토필요"
    else:
        scored["주변건물수_검증상태"] = np.where(
            suspicious_zero, "주변건물/집중도_검토필요", "사용"
        )

    # 의심 행은 점수 산정에서 제외(NaN) → MinMax 스케일링
    score_input = scored[TEAM_SCORE_VARS].mask(suspicious_zero, np.nan)
    scaled = MinMaxScaler().fit_transform(score_input)
    scaled_df = pd.DataFrame(scaled, columns=TEAM_SCORE_VARS, index=scored.index)

    # '낮을수록 위험' 변수는 1- 반전
    scored["고립위험_정규화"] = 1 - scaled_df["주변건물수"]
    scored["밀집사각지대_정규화"] = 1 - scaled_df["집중도"]
    scored["단속위험도_정규화"] = scaled_df["단속위험도"]
    scored["구조노후도_정규화"] = scaled_df["구조노후도"]
    scored["도로폭위험도_정규화"] = scaled_df["도로폭위험도"]
    # 가중합 × 100 = 0~100 점수
    scored["사각지대_위험도점수"] = (
        scored["고립위험_정규화"] * 0.35
        + scored["밀집사각지대_정규화"] * 0.20
        + scored["도로폭위험도_정규화"] * 0.15
        + scored["구조노후도_정규화"] * 0.15
        + scored["단속위험도_정규화"] * 0.15
    ) * 100
    # 점수 기준 순위 (동점은 작은 순위)
    scored["사각지대_위험순위"] = (
        scored["사각지대_위험도점수"]
        .rank(ascending=False, method="min")
        .astype("Int64")
    )

    # 보고용 컬럼 — 존재하는 것만 추려 CSV 저장
    score_cols = [
        "사각지대_위험순위",
        "구",
        "동",
        "숙소명",
        "업종",
        "주변건물수_검증상태",
        "주변건물수_보정출처",
        "사각지대_위험도점수",
        "고립위험_정규화",
        "밀집사각지대_정규화",
        "도로폭위험도_정규화",
        "구조노후도_정규화",
        "단속위험도_정규화",
        "주변건물수",
        "집중도",
        "도로폭위험도",
        "구조노후도",
        "단속위험도",
        "위험점수_AHP",
    ]
    score_cols = [c for c in score_cols if c in scored.columns]
    scored.sort_values("사각지대_위험도점수", ascending=False)[score_cols].to_csv(
        OUT / "blindspot_risk_score_team_formula.csv", index=False, encoding="utf-8-sig"
    )
    return scored


def run_clustering(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """업종별로 K 후보(2~6) 중 silhouette 최대 K 선택 → KMeans 적합."""
    rows = []
    frames = []

    for group in GROUP_ORDER:
        sub = df[df[GROUP_COL] == group].dropna(subset=RISK_VARS).copy()
        # 표본이 너무 작으면 스킵 (silhouette 계산 불가)
        if len(sub) < 10:
            continue
        X = StandardScaler().fit_transform(sub[RISK_VARS])
        # K 후보 — 표본 수가 적으면 최댓값 자동 축소
        k_candidates = list(range(2, min(7, len(sub) - 1)))
        scores = []
        for k in k_candidates:
            labels = KMeans(n_clusters=k, random_state=42, n_init=20).fit_predict(X)
            scores.append(silhouette_score(X, labels))
        # 최고 silhouette 의 K 채택
        best_k = k_candidates[int(np.argmax(scores))]
        model = KMeans(n_clusters=best_k, random_state=42, n_init=20)
        sub["업종별_군집"] = model.fit_predict(X)
        sub["업종별_군집명"] = (
            sub[GROUP_COL] + " 군집 " + sub["업종별_군집"].astype(str)
        )
        frames.append(sub)

        # 군집 요약 — 변수별 평균 + 시설수 + 선택 K + silhouette
        summary = (
            sub.groupby("업종별_군집")[RISK_VARS + ["위험점수_AHP"]].mean().round(4)
        )
        summary["시설수"] = sub.groupby("업종별_군집").size()
        summary["업종"] = group
        summary["선택_K"] = best_k
        summary["silhouette"] = max(scores)
        rows.append(summary.reset_index())

    clustered = pd.concat(frames, ignore_index=True)
    cluster_summary = pd.concat(rows, ignore_index=True)
    clustered.to_csv(
        OUT / "step1_industry_clusters.csv", index=False, encoding="utf-8-sig"
    )
    cluster_summary.to_csv(
        OUT / "step1_cluster_summary.csv", index=False, encoding="utf-8-sig"
    )
    return clustered, cluster_summary


def run_ridge_lasso(df: pd.DataFrame) -> pd.DataFrame:
    """전체 + 업종별로 RidgeCV / LassoCV 적합 → 변수 선택 결과 비교."""
    rows = []
    # alpha 후보 — 로그스케일 60개
    alphas = np.logspace(-3, 3, 60)

    for group in ["전체"] + GROUP_ORDER:
        sub = df if group == "전체" else df[df[GROUP_COL] == group]
        sub = sub.dropna(subset=RISK_VARS + [TARGET_COL]).copy()
        # 통계적 유의성 확보를 위해 30 미만 표본은 스킵
        if len(sub) < 30:
            continue
        X = StandardScaler().fit_transform(sub[RISK_VARS])
        y = sub[TARGET_COL].to_numpy()

        ridge = RidgeCV(alphas=alphas, cv=5).fit(X, y)
        lasso = LassoCV(alphas=alphas, cv=5, random_state=42, max_iter=20000).fit(X, y)
        # 변수별 계수 + Lasso 선택 여부
        for var, rc, lc in zip(RISK_VARS, ridge.coef_, lasso.coef_):
            rows.append(
                {
                    "업종": group,
                    "변수": var,
                    "ridge_coef": rc,
                    "lasso_coef": lc,
                    "lasso_selected": abs(lc) > 1e-8,
                    "ridge_alpha": ridge.alpha_,
                    "lasso_alpha": lasso.alpha_,
                    "표본수": len(sub),
                }
            )

    result = pd.DataFrame(rows)
    result.to_csv(
        OUT / "step2_ridge_lasso_coefficients.csv", index=False, encoding="utf-8-sig"
    )
    return result


def run_ols_moran(df: pd.DataFrame) -> pd.DataFrame:
    """OLS 적합 + 잔차에 대한 Moran I 검정 (HC3 강건 SE)."""
    rows = []

    for group in ["전체"] + GROUP_ORDER:
        sub = df if group == "전체" else df[df[GROUP_COL] == group]
        sub = sub.dropna(subset=RISK_VARS + [TARGET_COL, "x_5181", "y_5181"]).copy()
        # 모란 I 까지 안정적으로 보려면 40 이상
        if len(sub) < 40:
            continue

        X = sm.add_constant(StandardScaler().fit_transform(sub[RISK_VARS]))
        # OLS — HC3 이분산 강건 표준오차
        model = sm.OLS(sub[TARGET_COL].to_numpy(), X).fit(cov_type="HC3")
        # 잔차 모란 I — KNN k=8(작은 표본은 자동 축소)
        coords = sub[["x_5181", "y_5181"]].to_numpy()
        w = KNN.from_array(coords, k=min(8, len(sub) - 1))
        w.transform = "r"
        mi = Moran(model.resid, w, permutations=999)

        # 변수별 계수/p값 + 모형 R² + 잔차 모란 I
        for idx, term in enumerate(["const"] + RISK_VARS):
            rows.append(
                {
                    "업종": group,
                    "term": term,
                    "coef": model.params[idx],
                    "p_value": model.pvalues[idx],
                    "significant_0_05": model.pvalues[idx] < 0.05,
                    "ols_r2": model.rsquared,
                    "moran_I_residual": mi.I,
                    "moran_p_sim": mi.p_sim,
                    "표본수": len(sub),
                }
            )

    result = pd.DataFrame(rows)
    result.to_csv(OUT / "step3_ols_moran.csv", index=False, encoding="utf-8-sig")
    return result


def _spatial_row(model, model_name: str, group: str, n: int) -> dict:
    """공간시차/오차 모형 결과 → 보고용 dict 변환 (속성 안전 추출)."""
    return {
        "업종": group,
        "model": model_name,
        "pseudo_r2": getattr(model, "pr2", np.nan),
        "log_likelihood": getattr(model, "logll", np.nan),
        "aic": getattr(model, "aic", np.nan),
        # SAR 은 rho, SEM 은 lam → 어느 쪽이든 추출
        "spatial_param": float(getattr(model, "rho", getattr(model, "lam", np.nan))),
        "표본수": n,
    }


def run_spatial_lag_error(df: pd.DataFrame) -> pd.DataFrame:
    """Spatial Lag (SAR) + Spatial Error (SEM) 적합 — 업종별 + 전체."""
    rows = []
    # spreg 미설치 환경에서는 skip 메시지만 저장
    if ML_Lag is None or ML_Error is None:
        result = pd.DataFrame([{"업종": "전체", "model": "spreg unavailable"}])
        result.to_csv(
            OUT / "step4_spatial_lag_error.csv", index=False, encoding="utf-8-sig"
        )
        return result

    for group in ["전체"] + GROUP_ORDER:
        sub = df if group == "전체" else df[df[GROUP_COL] == group]
        sub = sub.dropna(subset=RISK_VARS + [TARGET_COL, "x_5181", "y_5181"]).copy()
        # 표본이 80 이상일 때만 — SAR/SEM 안정성 확보
        if len(sub) < 80:
            continue

        # 대시보드 빌드 시간 안정화를 위해 1800 행으로 다운샘플
        # 최종 추론은 여전히 OLS/GWR 페이지 기준 — 공간시차/오차는 보조 진단
        if len(sub) > 1800:
            sub = sub.sample(1800, random_state=42)

        y = sub[[TARGET_COL]].to_numpy()
        X = StandardScaler().fit_transform(sub[RISK_VARS])
        w = KNN.from_array(sub[["x_5181", "y_5181"]].to_numpy(), k=min(8, len(sub) - 1))
        w.transform = "r"

        # SAR (Spatial Lag) — y = ρWy + Xβ + ε
        try:
            lag = ML_Lag(y, X, w=w, name_y=TARGET_COL, name_x=RISK_VARS)
            rows.append(_spatial_row(lag, "Spatial Lag", group, len(sub)))
        except Exception as exc:
            rows.append(
                {
                    "업종": group,
                    "model": "Spatial Lag",
                    "error": str(exc),
                    "표본수": len(sub),
                }
            )

        # SEM (Spatial Error) — y = Xβ + u, u = λWu + ε
        try:
            err = ML_Error(y, X, w=w, name_y=TARGET_COL, name_x=RISK_VARS)
            rows.append(_spatial_row(err, "Spatial Error", group, len(sub)))
        except Exception as exc:
            rows.append(
                {
                    "업종": group,
                    "model": "Spatial Error",
                    "error": str(exc),
                    "표본수": len(sub),
                }
            )

    result = pd.DataFrame(rows)
    result.to_csv(
        OUT / "step4_spatial_lag_error.csv", index=False, encoding="utf-8-sig"
    )
    return result


def summarize_gwr() -> pd.DataFrame:
    """사전 산출된 GWR 결과 CSV 의 수치 컬럼별 평균/표준편차/min/max 요약."""
    if not GWR_SOURCE.exists():
        result = pd.DataFrame(
            [
                {
                    "source": "gwr_results.csv",
                    "status": "not_found",
                    "note": "GWR 결과 파일 없음",
                }
            ]
        )
    else:
        gwr = read_csv(GWR_SOURCE)
        # 수치 컬럼만 추출 → 4개 통계 (평균/표준편차/최소/최대)
        numeric = gwr.select_dtypes(include=[np.number])
        summary = numeric.agg(["mean", "std", "min", "max"]).T.reset_index()
        summary = summary.rename(columns={"index": "metric"})
        result = summary
        result["source"] = str(GWR_SOURCE.relative_to(BASE))
        result["status"] = "loaded"
    result.to_csv(OUT / "step5_gwr_mgwr_summary.csv", index=False, encoding="utf-8-sig")
    return result


def build_final_rank(df: pd.DataFrame) -> pd.DataFrame:
    """AHP 위험점수 기준 최종 시설 순위 + 보조 지표(사각지대/기대피해액) 결합 표."""
    rank_cols = [
        "구",
        "동",
        "숙소명",
        "업종",
        "업종별_군집명",
        "위험점수_AHP",
        "사각지대_위험도점수",
        "사각지대_위험순위",
        "주변건물수_검증상태",
        "주변건물수_보정출처",
        "기대피해액_백만원",
        "예상_화재발생확률",
        "조건부_예상피해액_백만원",
        "소방위험도_점수",
        "주변건물수",
        "집중도",
        "단속위험도",
        "구조노후도",
        "도로폭위험도",
        "위도",
        "경도",
    ]
    ranked = df.copy()
    # 기대피해액 컬럼이 결합되지 않았다면 NaN 으로 추가 — 후속 rank 계산 안전
    if "기대피해액_백만원" not in ranked.columns:
        ranked["기대피해액_백만원"] = np.nan
    ranked["AHP위험순위"] = (
        ranked["위험점수_AHP"].rank(ascending=False, method="min").astype(int)
    )
    ranked["기대피해액순위"] = ranked["기대피해액_백만원"].rank(
        ascending=False, method="min"
    )
    # 결측은 마지막 순위로 채워 정렬 시 끝으로 가도록
    ranked["기대피해액순위"] = ranked["기대피해액순위"].fillna(len(ranked)).astype(int)
    # 발표용 주 위험도는 AHP 기준
    # 기대피해액은 금액 예측 정확도가 낮아 보조 비교 지표로만 사용
    ranked["최종위험순위"] = ranked["AHP위험순위"]
    rank_cols = [
        "최종위험순위",
        "AHP위험순위",
        "사각지대_위험순위",
        "기대피해액순위",
    ] + [c for c in rank_cols if c in ranked.columns]
    # 중복 제거하면서 입력 순서 유지
    rank_cols = list(dict.fromkeys(rank_cols))
    ranked = ranked.sort_values(["최종위험순위", "기대피해액순위"])[rank_cols]
    ranked.to_csv(
        OUT / "step6_final_facility_rank.csv", index=False, encoding="utf-8-sig"
    )
    return ranked


def main() -> None:
    """파이프라인 메인 — 통합 데이터 → 점수 → 군집 → 회귀 → 공간 → 순위 → 매니페스트."""
    base = merge_sources()
    base = add_team_blindspot_score(base)
    base.to_csv(OUT / "analysis_dataset.csv", index=False, encoding="utf-8-sig")
    clustered, cluster_summary = run_clustering(base)
    ridge_lasso = run_ridge_lasso(clustered)
    ols_moran = run_ols_moran(clustered)
    spatial = run_spatial_lag_error(clustered)
    gwr = summarize_gwr()
    final_rank = build_final_rank(clustered)

    # 매니페스트 — 모든 산출물 행수와 파일명 인벤토리
    manifest = {
        "target": TARGET_COL,
        "risk_variables": RISK_VARS,
        "groups": GROUP_ORDER,
        "rows": {
            "analysis_dataset": len(base),
            "clustered": len(clustered),
            "cluster_summary": len(cluster_summary),
            "ridge_lasso": len(ridge_lasso),
            "ols_moran": len(ols_moran),
            "spatial_lag_error": len(spatial),
            "gwr_summary": len(gwr),
            "final_rank": len(final_rank),
        },
        "files": sorted(p.name for p in OUT.glob("*.csv")),
    }
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
