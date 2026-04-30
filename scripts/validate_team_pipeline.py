# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from esda.moran import Moran
from libpysal.weights import KNN
from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LassoCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import BallTree
from sklearn.preprocessing import MinMaxScaler, StandardScaler


BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "data" / "team_pipeline_validation"
OUT.mkdir(parents=True, exist_ok=True)

ACC_PATH = BASE / "0424" / "분석" / "tables" / "분석변수_최종테이블0423_AHP3등급비교_주변건물수보정.csv"
FIRE_PATH = BASE / "data" / "화재출동" / "화재출동_2021_2024.csv"

FEATURES = ["승인연도", "주변건물수", "집중도", "단속위험도", "구조노후도", "도로폭위험도"]
RISK_FEATURES = ["주변건물수", "집중도", "단속위험도", "구조노후도", "도로폭위험도"]
GWR_X_VARS = ["구조노후도", "도로폭위험도", "주변건물수", "집중도"]
RANDOM_STATE = 42


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    acc = read_csv(ACC_PATH).copy()
    fire = read_csv(FIRE_PATH).copy()
    fire["발생연도"] = pd.to_numeric(fire["발생연도"], errors="coerce")
    fire = fire[fire["발생연도"].between(2021, 2024)].copy()
    for col in FEATURES + ["위도", "경도", "x_5181", "y_5181"]:
        if col in acc.columns:
            acc[col] = pd.to_numeric(acc[col], errors="coerce")
    for col in ["위도", "경도", "재산피해액(천원)"]:
        fire[col] = pd.to_numeric(fire[col], errors="coerce")
    acc = acc.dropna(subset=FEATURES + ["위도", "경도"]).reset_index(drop=True)
    fire = fire.dropna(subset=["위도", "경도"]).reset_index(drop=True)
    return acc, fire


def add_fire_targets(acc: pd.DataFrame, fire: pd.DataFrame, radius_m: int = 150) -> pd.DataFrame:
    result = acc.copy()
    fire_coords = np.radians(fire[["위도", "경도"]].to_numpy())
    acc_coords = np.radians(result[["위도", "경도"]].to_numpy())
    tree = BallTree(fire_coords, metric="haversine")
    indices = tree.query_radius(acc_coords, r=radius_m / 6371000)

    damage = pd.to_numeric(fire["재산피해액(천원)"], errors="coerce").fillna(0).to_numpy()
    result["fire_count_150m"] = [len(idx) for idx in indices]
    result["target_damage_sum_천원"] = [float(damage[idx].sum()) for idx in indices]
    result["target_damage_mean_천원"] = [
        float(damage[idx].mean()) if len(idx) else 0.0 for idx in indices
    ]
    result["fire_exists_150m"] = result["fire_count_150m"].gt(0).astype(int)
    return result


def run_clustering(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    x_scaled = StandardScaler().fit_transform(df[FEATURES].fillna(0))
    inertia_rows = []
    for k in range(1, 11):
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit(x_scaled)
        inertia_rows.append({"k": k, "inertia": model.inertia_})
    inertia = pd.DataFrame(inertia_rows)
    inertia.to_csv(OUT / "01_elbow_inertia.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(inertia["k"], inertia["inertia"], marker="o")
    ax.set_title("Optimal k 선택을 위한 Elbow Method")
    ax.set_xlabel("k")
    ax.set_ylabel("Inertia")
    fig.tight_layout()
    fig.savefig(OUT / "01_elbow_method.png", dpi=180)
    plt.close(fig)

    clustered = df.copy()
    clustered["cluster"] = KMeans(n_clusters=4, random_state=RANDOM_STATE, n_init=10).fit_predict(x_scaled)
    summary = (
        clustered.groupby("cluster")
        .agg(
            시설수=("숙소명", "count"),
            평균_화재수=("fire_count_150m", "mean"),
            중앙_화재수=("fire_count_150m", "median"),
            평균_피해액_천원=("target_damage_sum_천원", "mean"),
            평균_주변건물수=("주변건물수", "mean"),
            평균_집중도=("집중도", "mean"),
            평균_도로폭위험도=("도로폭위험도", "mean"),
        )
        .round(4)
        .reset_index()
    )
    summary.to_csv(OUT / "01_cluster_fire_summary.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x="cluster", y="fire_count_150m", data=clustered, ax=ax)
    ax.set_title("군집별 화재 발생 빈도 분포")
    ax.set_xlabel("cluster")
    ax.set_ylabel("fire_count within 150m")
    fig.tight_layout()
    fig.savefig(OUT / "01_cluster_fire_boxplot.png", dpi=180)
    plt.close(fig)
    return clustered, summary


def run_lasso(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    train_df = df[df["fire_exists_150m"].eq(1)].copy()
    x = train_df[FEATURES].fillna(0)
    y = np.log1p(train_df["target_damage_sum_천원"])
    x_scaled = StandardScaler().fit_transform(x)
    lasso = Lasso(alpha=0.01, max_iter=20000).fit(x_scaled, y)
    lasso_cv = LassoCV(cv=5, random_state=RANDOM_STATE, max_iter=20000).fit(x_scaled, y)
    coef = pd.DataFrame(
        {
            "변수": FEATURES,
            "lasso_alpha_0_01_coef": lasso.coef_,
            "lasso_cv_coef": lasso_cv.coef_,
            "abs_cv_coef": np.abs(lasso_cv.coef_),
        }
    ).sort_values("abs_cv_coef", ascending=False)
    coef.to_csv(OUT / "02_lasso_coefficients.csv", index=False, encoding="utf-8-sig")
    return coef, {"train_rows_fire_matched": int(len(train_df)), "lasso_cv_alpha": float(lasso_cv.alpha_)}


def add_team_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    scaled = MinMaxScaler().fit_transform(result[RISK_FEATURES].fillna(0))
    s = pd.DataFrame(scaled, columns=RISK_FEATURES, index=result.index)
    result["고립위험"] = 1 - s["주변건물수"]
    result["밀집사각지대"] = 1 - s["집중도"]
    result["위험도점수"] = (
        result["고립위험"] * 0.35
        + result["밀집사각지대"] * 0.20
        + s["도로폭위험도"] * 0.15
        + s["구조노후도"] * 0.15
        + s["단속위험도"] * 0.15
    ) * 100
    result.sort_values("위험도점수", ascending=False).head(30).to_csv(
        OUT / "03_team_risk_top30.csv", index=False, encoding="utf-8-sig"
    )
    return result


def run_moran(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    coords_latlon = df[["위도", "경도"]].to_numpy()
    w_latlon = KNN.from_array(coords_latlon, k=5)
    w_latlon.transform = "R"
    moran_latlon = Moran(df["위험도점수"].to_numpy(), w_latlon, permutations=999)
    rows.append({"좌표": "위경도_그대로", "moran_I": moran_latlon.I, "p_value": moran_latlon.p_sim})

    if {"x_5181", "y_5181"}.issubset(df.columns):
        valid = df[["x_5181", "y_5181", "위험도점수"]].dropna()
        w_proj = KNN.from_array(valid[["x_5181", "y_5181"]].to_numpy(), k=5)
        w_proj.transform = "R"
        moran_proj = Moran(valid["위험도점수"].to_numpy(), w_proj, permutations=999)
        rows.append({"좌표": "EPSG5181_평면좌표", "moran_I": moran_proj.I, "p_value": moran_proj.p_sim})
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "04_moran_results.csv", index=False, encoding="utf-8-sig")
    return result


def run_gwr_sample(df: pd.DataFrame, max_rows: int = 650) -> tuple[pd.DataFrame, dict]:
    work = df.dropna(subset=["x_5181", "y_5181", "위험도점수"] + GWR_X_VARS).copy()
    if len(work) > max_rows:
        work = work.sample(max_rows, random_state=RANDOM_STATE).sort_index().copy()
    coords = work[["x_5181", "y_5181"]].to_numpy()
    y = work["위험도점수"].to_numpy().reshape((-1, 1))
    x = StandardScaler().fit_transform(work[GWR_X_VARS])
    selector = Sel_BW(coords, y, x, spherical=False, n_jobs=1)
    bw = selector.search(bw_min=30)
    model = GWR(coords, y, x, bw, spherical=False, n_jobs=1).fit()
    params = pd.DataFrame(
        model.params,
        columns=["Intercept", "C_구조노후도", "C_도로폭위험도", "C_주변건물수", "C_집중도"],
        index=work.index,
    )
    out = pd.concat([work.reset_index(drop=True), params.reset_index(drop=True)], axis=1)
    out.to_csv(OUT / "05_gwr_sample_params.csv", index=False, encoding="utf-8-sig")
    return out, {
        "gwr_rows_sampled": int(len(out)),
        "gwr_bandwidth": float(bw),
        "gwr_aicc": float(model.aicc),
        "gwr_r2": float(model.R2),
        "gwr_adj_r2": float(model.adj_R2),
    }


def run_rf_checks(df: pd.DataFrame, gwr_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []

    def fit_rf(name: str, data: pd.DataFrame, features: list[str], target: str) -> None:
        model_df = data.dropna(subset=features + [target]).copy()
        x_train, x_test, y_train, y_test = train_test_split(
            model_df[features], model_df[target], test_size=0.2, random_state=RANDOM_STATE
        )
        rf = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE)
        rf.fit(x_train, y_train)
        pred = rf.predict(x_test)
        rows.append(
            {
                "검증": name,
                "target": target,
                "features": ", ".join(features),
                "rows": len(model_df),
                "r2": r2_score(y_test, pred),
                "mae": mean_absolute_error(y_test, pred),
            }
        )

    fit_rf("순환검증_A: 위험도점수를 구성변수로 다시 예측", df, GWR_X_VARS, "위험도점수")
    fit_rf(
        "순환검증_B: 위험도점수 + GWR계수 사용",
        gwr_df,
        GWR_X_VARS + ["C_구조노후도", "C_도로폭위험도", "C_주변건물수", "C_집중도"],
        "위험도점수",
    )
    true_df = df.copy()
    true_df["log1p_fire_count_150m"] = np.log1p(true_df["fire_count_150m"])
    true_df["log1p_damage_sum_천원"] = np.log1p(true_df["target_damage_sum_천원"])
    fit_rf("진짜검증_C: 실제 화재수 예측", true_df, FEATURES, "log1p_fire_count_150m")
    fit_rf("진짜검증_D: 실제 재산피해액 예측", true_df, FEATURES, "log1p_damage_sum_천원")

    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT / "06_rf_validation_metrics.csv", index=False, encoding="utf-8-sig")

    # Feature importance for the shiny but circular model.
    sample = gwr_df.dropna(
        subset=GWR_X_VARS + ["C_구조노후도", "C_도로폭위험도", "C_주변건물수", "C_집중도", "위험도점수"]
    ).copy()
    final_features = GWR_X_VARS + ["C_구조노후도", "C_도로폭위험도", "C_주변건물수", "C_집중도"]
    x_train, x_test, y_train, y_test = train_test_split(
        sample[final_features], sample["위험도점수"], test_size=0.2, random_state=RANDOM_STATE
    )
    rf = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE)
    rf.fit(x_train, y_train)
    importance = pd.DataFrame({"변수": final_features, "importance": rf.feature_importances_}).sort_values(
        "importance", ascending=False
    )
    importance.to_csv(OUT / "06_rf_circular_feature_importance.csv", index=False, encoding="utf-8-sig")
    return metrics, importance


def run_ols_sanity(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in ["위험도점수", "log1p_fire_count_150m", "log1p_damage_sum_천원"]:
        work = df.copy()
        if target == "log1p_fire_count_150m":
            work[target] = np.log1p(work["fire_count_150m"])
        if target == "log1p_damage_sum_천원":
            work[target] = np.log1p(work["target_damage_sum_천원"])
        reg = work.dropna(subset=FEATURES + [target])
        x = sm.add_constant(StandardScaler().fit_transform(reg[FEATURES]))
        model = sm.OLS(reg[target], x).fit(cov_type="HC3")
        for term, coef, p in zip(["const"] + FEATURES, model.params, model.pvalues):
            rows.append({"target": target, "term": term, "coef": coef, "p_value": p, "r2": model.rsquared})
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "07_ols_sanity.csv", index=False, encoding="utf-8-sig")
    return result


def main() -> None:
    acc, fire = load_data()
    df = add_fire_targets(acc, fire, 150)
    clustered, cluster_summary = run_clustering(df)
    lasso_coef, lasso_meta = run_lasso(clustered)
    scored = add_team_risk_score(clustered)
    moran = run_moran(scored)
    gwr_df, gwr_meta = run_gwr_sample(scored)
    rf_metrics, rf_importance = run_rf_checks(scored, gwr_df)
    ols = run_ols_sanity(scored)

    scored.to_csv(OUT / "team_pipeline_scored_dataset.csv", index=False, encoding="utf-8-sig")
    summary = {
        "rows": int(len(scored)),
        "fire_rows_2021_2024": int(len(fire)),
        "facilities_with_fire_150m": int(scored["fire_exists_150m"].sum()),
        "mean_fire_count_150m": float(scored["fire_count_150m"].mean()),
        "max_fire_count_150m": int(scored["fire_count_150m"].max()),
        **lasso_meta,
        **gwr_meta,
        "rf_metrics": rf_metrics.to_dict(orient="records"),
        "moran": moran.to_dict(orient="records"),
        "outputs": sorted(p.name for p in OUT.glob("*")),
    }
    (OUT / "validation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
