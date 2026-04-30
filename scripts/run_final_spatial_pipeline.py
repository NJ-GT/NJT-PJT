# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from esda.moran import Moran
from libpysal.weights import KNN
from sklearn.cluster import KMeans
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

try:
    from spreg import ML_Error, ML_Lag
except Exception:  # pragma: no cover - optional environment failure
    ML_Error = None
    ML_Lag = None


BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "data" / "final_spatial_pipeline"
OUT.mkdir(parents=True, exist_ok=True)

CLUSTER_SOURCE_REPAIRED = BASE / "0424" / "분석" / "tables" / "분석변수_최종테이블0423_AHP3등급비교_주변건물수보정.csv"
CLUSTER_SOURCE = BASE / "0424" / "분석" / "tables" / "분석변수_최종테이블0423_AHP3등급비교.csv"
CLUSTER_FALLBACK_SOURCE = BASE / "data" / "clustering_result_all.csv"
FIRE_SOURCE = BASE / "data" / "data_with_fire_targets.csv"
EXPECTED_DAMAGE = BASE / "0424" / "data" / "facility_expected_property_damage_two_stage.csv"
GWR_SOURCE = BASE / "data" / "gwr_results.csv"

GROUP_COL = "업종"
TARGET_COL = "log1p_반경100m"
RISK_VARS = [
    "소방위험도_점수",
    "주변건물수",
    "집중도",
    "단속위험도",
    "구조노후도",
    "도로폭위험도",
]
TEAM_SCORE_VARS = ["주변건물수", "집중도", "단속위험도", "구조노후도", "도로폭위험도"]
GROUP_ORDER = ["관광숙박업", "숙박업", "외국인관광도시민박업"]


def name_key(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r"\s+", "", regex=True).str.strip()


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def merge_sources() -> pd.DataFrame:
    if CLUSTER_SOURCE_REPAIRED.exists():
        cluster_source = CLUSTER_SOURCE_REPAIRED
    else:
        cluster_source = CLUSTER_SOURCE if CLUSTER_SOURCE.exists() else CLUSTER_FALLBACK_SOURCE
    cluster = read_csv(cluster_source)
    fire = read_csv(FIRE_SOURCE)

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

    cluster = cluster.copy()
    cluster["_name_key"] = name_key(cluster["숙소명"])
    cluster["_lat_key"] = cluster["위도"].round(6)
    cluster["_lon_key"] = cluster["경도"].round(6)

    merged = cluster.merge(fire_map, on=["_name_key", "_lat_key", "_lon_key"], how="left")

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
        merged = merged.merge(damage_map, on=["_name_key", "_lat_key", "_lon_key"], how="left")

    merged["집중도"] = pd.to_numeric(merged["집중도"], errors="coerce")
    merged["단속위험도"] = pd.to_numeric(merged["단속위험도"], errors="coerce")
    merged["구조노후도"] = pd.to_numeric(merged["구조노후도"], errors="coerce")
    merged["도로폭위험도"] = pd.to_numeric(merged["도로폭위험도"], errors="coerce")
    merged["주변건물수"] = pd.to_numeric(merged["주변건물수"], errors="coerce")
    merged["x_5181"] = pd.to_numeric(merged["x_5181"], errors="coerce")
    merged["y_5181"] = pd.to_numeric(merged["y_5181"], errors="coerce")

    merged = merged.drop(columns=["_name_key", "_lat_key", "_lon_key"], errors="ignore")
    return merged


def add_team_blindspot_score(df: pd.DataFrame) -> pd.DataFrame:
    """팀원 공식: 고립/저밀집 사각지대 발굴형 위험도 점수."""
    scored = df.copy()
    for col in TEAM_SCORE_VARS:
        scored[col] = pd.to_numeric(scored[col], errors="coerce")

    suspicious_zero = scored["주변건물수"].eq(0) | scored["집중도"].eq(0)
    if "주변건물수_보정여부" in scored.columns:
        scored["주변건물수_검증상태"] = scored["주변건물수_보정여부"].fillna("사용")
        scored.loc[suspicious_zero, "주변건물수_검증상태"] = "주변건물/집중도_검토필요"
    else:
        scored["주변건물수_검증상태"] = np.where(suspicious_zero, "주변건물/집중도_검토필요", "사용")

    score_input = scored[TEAM_SCORE_VARS].mask(suspicious_zero, np.nan)
    scaled = MinMaxScaler().fit_transform(score_input)
    scaled_df = pd.DataFrame(scaled, columns=TEAM_SCORE_VARS, index=scored.index)

    scored["고립위험_정규화"] = 1 - scaled_df["주변건물수"]
    scored["밀집사각지대_정규화"] = 1 - scaled_df["집중도"]
    scored["단속위험도_정규화"] = scaled_df["단속위험도"]
    scored["구조노후도_정규화"] = scaled_df["구조노후도"]
    scored["도로폭위험도_정규화"] = scaled_df["도로폭위험도"]
    scored["사각지대_위험도점수"] = (
        scored["고립위험_정규화"] * 0.35
        + scored["밀집사각지대_정규화"] * 0.20
        + scored["도로폭위험도_정규화"] * 0.15
        + scored["구조노후도_정규화"] * 0.15
        + scored["단속위험도_정규화"] * 0.15
    ) * 100
    scored["사각지대_위험순위"] = scored["사각지대_위험도점수"].rank(
        ascending=False, method="min"
    ).astype("Int64")

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
    rows = []
    frames = []

    for group in GROUP_ORDER:
        sub = df[df[GROUP_COL] == group].dropna(subset=RISK_VARS).copy()
        if len(sub) < 10:
            continue
        X = StandardScaler().fit_transform(sub[RISK_VARS])
        k_candidates = list(range(2, min(7, len(sub) - 1)))
        scores = []
        for k in k_candidates:
            labels = KMeans(n_clusters=k, random_state=42, n_init=20).fit_predict(X)
            scores.append(silhouette_score(X, labels))
        best_k = k_candidates[int(np.argmax(scores))]
        model = KMeans(n_clusters=best_k, random_state=42, n_init=20)
        sub["업종별_군집"] = model.fit_predict(X)
        sub["업종별_군집명"] = sub[GROUP_COL] + " 군집 " + sub["업종별_군집"].astype(str)
        frames.append(sub)

        summary = sub.groupby("업종별_군집")[RISK_VARS + ["위험점수_AHP"]].mean().round(4)
        summary["시설수"] = sub.groupby("업종별_군집").size()
        summary["업종"] = group
        summary["선택_K"] = best_k
        summary["silhouette"] = max(scores)
        rows.append(summary.reset_index())

    clustered = pd.concat(frames, ignore_index=True)
    cluster_summary = pd.concat(rows, ignore_index=True)
    clustered.to_csv(OUT / "step1_industry_clusters.csv", index=False, encoding="utf-8-sig")
    cluster_summary.to_csv(OUT / "step1_cluster_summary.csv", index=False, encoding="utf-8-sig")
    return clustered, cluster_summary


def run_ridge_lasso(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    alphas = np.logspace(-3, 3, 60)

    for group in ["전체"] + GROUP_ORDER:
        sub = df if group == "전체" else df[df[GROUP_COL] == group]
        sub = sub.dropna(subset=RISK_VARS + [TARGET_COL]).copy()
        if len(sub) < 30:
            continue
        X = StandardScaler().fit_transform(sub[RISK_VARS])
        y = sub[TARGET_COL].to_numpy()

        ridge = RidgeCV(alphas=alphas, cv=5).fit(X, y)
        lasso = LassoCV(alphas=alphas, cv=5, random_state=42, max_iter=20000).fit(X, y)
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
    result.to_csv(OUT / "step2_ridge_lasso_coefficients.csv", index=False, encoding="utf-8-sig")
    return result


def run_ols_moran(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for group in ["전체"] + GROUP_ORDER:
        sub = df if group == "전체" else df[df[GROUP_COL] == group]
        sub = sub.dropna(subset=RISK_VARS + [TARGET_COL, "x_5181", "y_5181"]).copy()
        if len(sub) < 40:
            continue

        X = sm.add_constant(StandardScaler().fit_transform(sub[RISK_VARS]))
        model = sm.OLS(sub[TARGET_COL].to_numpy(), X).fit(cov_type="HC3")
        coords = sub[["x_5181", "y_5181"]].to_numpy()
        w = KNN.from_array(coords, k=min(8, len(sub) - 1))
        w.transform = "r"
        mi = Moran(model.resid, w, permutations=999)

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
    return {
        "업종": group,
        "model": model_name,
        "pseudo_r2": getattr(model, "pr2", np.nan),
        "log_likelihood": getattr(model, "logll", np.nan),
        "aic": getattr(model, "aic", np.nan),
        "spatial_param": float(getattr(model, "rho", getattr(model, "lam", np.nan))),
        "표본수": n,
    }


def run_spatial_lag_error(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if ML_Lag is None or ML_Error is None:
        result = pd.DataFrame([{"업종": "전체", "model": "spreg unavailable"}])
        result.to_csv(OUT / "step4_spatial_lag_error.csv", index=False, encoding="utf-8-sig")
        return result

    for group in ["전체"] + GROUP_ORDER:
        sub = df if group == "전체" else df[df[GROUP_COL] == group]
        sub = sub.dropna(subset=RISK_VARS + [TARGET_COL, "x_5181", "y_5181"]).copy()
        if len(sub) < 80:
            continue

        # Keep runtime stable for the dashboard build; final inference remains based on OLS/GWR pages.
        if len(sub) > 1800:
            sub = sub.sample(1800, random_state=42)

        y = sub[[TARGET_COL]].to_numpy()
        X = StandardScaler().fit_transform(sub[RISK_VARS])
        w = KNN.from_array(sub[["x_5181", "y_5181"]].to_numpy(), k=min(8, len(sub) - 1))
        w.transform = "r"

        try:
            lag = ML_Lag(y, X, w=w, name_y=TARGET_COL, name_x=RISK_VARS)
            rows.append(_spatial_row(lag, "Spatial Lag", group, len(sub)))
        except Exception as exc:
            rows.append({"업종": group, "model": "Spatial Lag", "error": str(exc), "표본수": len(sub)})

        try:
            err = ML_Error(y, X, w=w, name_y=TARGET_COL, name_x=RISK_VARS)
            rows.append(_spatial_row(err, "Spatial Error", group, len(sub)))
        except Exception as exc:
            rows.append({"업종": group, "model": "Spatial Error", "error": str(exc), "표본수": len(sub)})

    result = pd.DataFrame(rows)
    result.to_csv(OUT / "step4_spatial_lag_error.csv", index=False, encoding="utf-8-sig")
    return result


def summarize_gwr() -> pd.DataFrame:
    if not GWR_SOURCE.exists():
        result = pd.DataFrame(
            [{"source": "gwr_results.csv", "status": "not_found", "note": "GWR 결과 파일 없음"}]
        )
    else:
        gwr = read_csv(GWR_SOURCE)
        numeric = gwr.select_dtypes(include=[np.number])
        summary = numeric.agg(["mean", "std", "min", "max"]).T.reset_index()
        summary = summary.rename(columns={"index": "metric"})
        result = summary
        result["source"] = str(GWR_SOURCE.relative_to(BASE))
        result["status"] = "loaded"
    result.to_csv(OUT / "step5_gwr_mgwr_summary.csv", index=False, encoding="utf-8-sig")
    return result


def build_final_rank(df: pd.DataFrame) -> pd.DataFrame:
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
    if "기대피해액_백만원" not in ranked.columns:
        ranked["기대피해액_백만원"] = np.nan
    ranked["AHP위험순위"] = ranked["위험점수_AHP"].rank(ascending=False, method="min").astype(int)
    ranked["기대피해액순위"] = ranked["기대피해액_백만원"].rank(ascending=False, method="min")
    ranked["기대피해액순위"] = ranked["기대피해액순위"].fillna(len(ranked)).astype(int)
    # 최종위험순위는 발표용 주 위험도인 AHP를 기준으로 둔다.
    # 기대피해액은 금액 예측력이 낮으므로 보조 비교 지표로만 사용한다.
    ranked["최종위험순위"] = ranked["AHP위험순위"]
    rank_cols = [
        "최종위험순위",
        "AHP위험순위",
        "사각지대_위험순위",
        "기대피해액순위",
    ] + [c for c in rank_cols if c in ranked.columns]
    rank_cols = list(dict.fromkeys(rank_cols))
    ranked = ranked.sort_values(["최종위험순위", "기대피해액순위"])[rank_cols]
    ranked.to_csv(OUT / "step6_final_facility_rank.csv", index=False, encoding="utf-8-sig")
    return ranked


def main() -> None:
    base = merge_sources()
    base = add_team_blindspot_score(base)
    base.to_csv(OUT / "analysis_dataset.csv", index=False, encoding="utf-8-sig")
    clustered, cluster_summary = run_clustering(base)
    ridge_lasso = run_ridge_lasso(clustered)
    ols_moran = run_ols_moran(clustered)
    spatial = run_spatial_lag_error(clustered)
    gwr = summarize_gwr()
    final_rank = build_final_rank(clustered)

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
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
