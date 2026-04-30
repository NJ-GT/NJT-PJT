from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from esda.moran import Moran
from libpysal.weights import KNN
from mgwr.gwr import GWR, MGWR
from mgwr.sel_bw import Sel_BW
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from spreg import ML_Error, ML_Lag


warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "0424" / "data" / "cluster3_spatial_pipeline_fire_count_150m_0428"
FIRE_TARGET_PATH = ROOT / "data" / "team_pipeline_validation" / "team_pipeline_scored_dataset.csv"
OUT_DIR = ROOT / "0429" / "cluster2_spatial_pipeline_fire_count_150m_0429"

TARGET = "fire_count_150m"
CLUSTER_COL = "cluster_k2"
COORD_COLS = ["x_5181", "y_5181"]
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

KNN_CANDIDATES = [6, 8, 10, 12, 15, 20]
MORAN_PERMUTATIONS = 199
GWR_SAMPLE_CAP = 700
MGWR_SAMPLE_CAP = 220
RNG = np.random.RandomState(42)


def set_korean_font() -> None:
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.family"] = "Malgun Gothic"


def read_main_csv() -> pd.DataFrame:
    csv_files = sorted(DATA_DIR.glob("*.csv"), key=lambda p: p.stat().st_size, reverse=True)
    if not csv_files:
        raise FileNotFoundError(DATA_DIR)
    return pd.read_csv(csv_files[0], encoding="utf-8-sig")


def name_key(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r"\s+", "", regex=True).str.strip()


def attach_fire_count(df: pd.DataFrame) -> pd.DataFrame:
    if TARGET in df.columns:
        return df
    fire = pd.read_csv(FIRE_TARGET_PATH, encoding="utf-8-sig")
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
    return keyed.drop(columns=["_name_key", "_lat_key", "_lon_key"], errors="ignore")


def prepare_data() -> pd.DataFrame:
    df = attach_fire_count(read_main_csv())
    needed = ["구", "동", "숙소명", "경도", "위도", TARGET, *COORD_COLS, *REG_FEATURES, "최종_화재위험점수"]
    df = df[[c for c in needed if c in df.columns]].copy()
    for col in [TARGET, *COORD_COLS, *REG_FEATURES, "최종_화재위험점수"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=[TARGET, *COORD_COLS, *REG_FEATURES]).reset_index(drop=True)


def build_k2_clusters(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    rows = []
    fitted = {}
    for name, features in CLUSTER_FEATURE_SETS.items():
        work = df.dropna(subset=features).copy()
        x = StandardScaler().fit_transform(work[features].to_numpy(dtype=float))
        labels = KMeans(n_clusters=2, random_state=42, n_init=50).fit_predict(x)
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
        fitted[name] = labels
    tuning = pd.DataFrame(rows).sort_values(["silhouette", "calinski_harabasz"], ascending=False)
    best_name = str(tuning.iloc[0]["feature_set"])
    out = df.copy()
    out[CLUSTER_COL] = fitted[best_name]
    return out, tuning, best_name


def standardize_x(df: pd.DataFrame) -> np.ndarray:
    return StandardScaler().fit_transform(df[REG_FEATURES].to_numpy(dtype=float))


def build_weights(coords: np.ndarray, k: int) -> KNN:
    kk = min(k, max(1, len(coords) - 1))
    w = KNN.from_array(coords, k=kk)
    w.transform = "r"
    return w


def aicc_of(result) -> float:
    for attr in ("aicc", "AICc", "aic", "AIC"):
        try:
            return float(getattr(result, attr))
        except Exception:
            pass
    return float("nan")


def sample_for_local_model(df: pd.DataFrame, cap: int) -> pd.DataFrame:
    if len(df) <= cap:
        return df.copy().reset_index(drop=True)
    sampled_idx = RNG.choice(df.index.to_numpy(), cap, replace=False)
    return df.loc[np.sort(sampled_idx)].copy().reset_index(drop=True)


def run_ols(df: pd.DataFrame, cluster_id: int) -> tuple[dict, pd.DataFrame]:
    x = standardize_x(df)
    y = df[TARGET].to_numpy(dtype=float)
    coords = df[COORD_COLS].to_numpy(dtype=float)
    model = sm.OLS(y, sm.add_constant(x)).fit(cov_type="HC3")
    rows = []
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
    x = standardize_x(df)
    y = df[TARGET].to_numpy(dtype=float).reshape(-1, 1)
    coords = df[COORD_COLS].to_numpy(dtype=float)
    rows = []
    for model_name, model_cls in [("SLM", ML_Lag), ("SEM", ML_Error)]:
        for k in KNN_CANDIDATES:
            t0 = time.time()
            try:
                w = build_weights(coords, k)
                model = model_cls(y, x, w=w, name_y=TARGET, name_x=REG_FEATURES)
                resid = np.asarray(model.u).flatten()
                moran = Moran(resid, w, permutations=MORAN_PERMUTATIONS)
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
                        "fit": float(getattr(model, "pr2", getattr(model, "r2", np.nan))),
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
    ok = rows[rows["status"].eq("ok")].copy()
    if ok.empty:
        return rows.groupby("model", as_index=False).head(1)
    best_rows = []
    for model, sub in ok.groupby("model"):
        sub = sub.sort_values(["aic", "resid_moran_I"], ascending=[True, True])
        best_rows.append(sub.iloc[0])
    return pd.DataFrame(best_rows)


def select_gwr_bw(coords: np.ndarray, y: np.ndarray, x: np.ndarray) -> float:
    bw_min = max(30, x.shape[1] + 3)
    bw_max = max(bw_min + 2, min(len(y) - 1, 420))
    selector = Sel_BW(coords, y, x, fixed=False, kernel="bisquare", n_jobs=1)
    return float(selector.search(search_method="golden_section", bw_min=bw_min, bw_max=bw_max))


def run_gwr(df: pd.DataFrame, cluster_id: int) -> tuple[dict, pd.DataFrame]:
    work = sample_for_local_model(df, GWR_SAMPLE_CAP)
    coords = work[COORD_COLS].to_numpy(dtype=float)
    y = work[TARGET].to_numpy(dtype=float).reshape(-1, 1)
    x = standardize_x(work)
    t0 = time.time()
    try:
        bw = select_gwr_bw(coords, y, x)
        result = GWR(coords, y, x, bw=bw, fixed=False, kernel="bisquare", n_jobs=1).fit()
        resid = np.asarray(result.resid_response).flatten()
        w = build_weights(coords, 12)
        moran = Moran(resid, w, permutations=MORAN_PERMUTATIONS)
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
    work = sample_for_local_model(df, MGWR_SAMPLE_CAP)
    coords = work[COORD_COLS].to_numpy(dtype=float)
    y = work[TARGET].to_numpy(dtype=float).reshape(-1, 1)
    x = standardize_x(work)
    t0 = time.time()
    try:
        selector = Sel_BW(coords, y, x, multi=True, fixed=False, kernel="bisquare", n_jobs=1)
        selector.search(
            multi_bw_min=[max(30, x.shape[1] + 3)],
            multi_bw_max=[min(len(work) - 1, 180)],
            max_iter_multi=15,
            verbose=False,
        )
        result = MGWR(coords, y, x, selector, fixed=False, kernel="bisquare", n_jobs=1).fit()
        bw_values = np.asarray(selector.bw[0]).flatten()
        resid = np.asarray(result.resid_response).flatten()
        w = build_weights(coords, 12)
        moran = Moran(resid, w, permutations=MORAN_PERMUTATIONS)
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
    summary.columns = ["_".join(c).strip() for c in summary.columns]
    return summary.reset_index()


def save_cluster_profile_png(cluster_summary: pd.DataFrame, out_path: Path) -> None:
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
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), dpi=180)
    plot = model_summary.copy()
    plot["cluster"] = plot["cluster"].astype(str)

    sns.barplot(data=plot, x="model", y="fit", hue="cluster", palette="Set2", ax=axes[0])
    axes[0].set_title("모델 설명력")
    axes[0].set_ylabel("R2 / pseudo R2")
    axes[0].set_xlabel("")
    axes[0].grid(axis="y", alpha=0.25)

    sns.barplot(data=plot, x="model", y="aic", hue="cluster", palette="Set2", ax=axes[1])
    axes[1].set_title("AIC")
    axes[1].set_ylabel("낮을수록 유리")
    axes[1].set_xlabel("")
    axes[1].grid(axis="y", alpha=0.25)

    sns.barplot(data=plot, x="model", y="resid_moran_I", hue="cluster", palette="Set2", ax=axes[2])
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
    set_korean_font()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = prepare_data()
    df, cluster_tuning, best_feature_set = build_k2_clusters(df)

    df.to_csv(OUT_DIR / "최최최종0428변수테이블_cluster_k2.csv", index=False, encoding="utf-8-sig")
    cluster_tuning.to_csv(OUT_DIR / "k2_cluster_feature_set_tuning.csv", index=False, encoding="utf-8-sig")

    cluster_summary = summarize_clusters(df)
    cluster_summary.to_csv(OUT_DIR / "cluster_k2_feature_summary.csv", index=False, encoding="utf-8-sig")
    save_cluster_profile_png(cluster_summary, OUT_DIR / "cluster_k2_feature_profile.png")

    model_rows = []
    coef_tables = []
    all_spatial_tuning = []
    gwr_local_tables = []
    mgwr_bw_tables = []

    for cluster_id in sorted(df[CLUSTER_COL].dropna().astype(int).unique()):
        sub = df[df[CLUSTER_COL].astype(int).eq(cluster_id)].reset_index(drop=True)
        print(f"=== K=2 cluster {cluster_id} / n={len(sub):,} ===")
        ols_summary, coef = run_ols(sub, cluster_id)
        model_rows.append(ols_summary)
        coef_tables.append(coef)

        spatial_tuning = run_spatial_family(sub, cluster_id)
        all_spatial_tuning.append(spatial_tuning)
        model_rows.extend(select_best_spatial(spatial_tuning).to_dict("records"))

        gwr_summary, gwr_local = run_gwr(sub, cluster_id)
        model_rows.append(gwr_summary)
        if not gwr_local.empty:
            gwr_local_tables.append(gwr_local)

        mgwr_summary, mgwr_bw = run_mgwr(sub, cluster_id)
        model_rows.append(mgwr_summary)
        if not mgwr_bw.empty:
            mgwr_bw_tables.append(mgwr_bw)

    model_summary = pd.DataFrame(model_rows)
    spatial_tuning = pd.concat(all_spatial_tuning, ignore_index=True)
    coef_df = pd.concat(coef_tables, ignore_index=True)
    gwr_local_df = pd.concat(gwr_local_tables, ignore_index=True) if gwr_local_tables else pd.DataFrame()
    mgwr_bw_df = pd.concat(mgwr_bw_tables, ignore_index=True) if mgwr_bw_tables else pd.DataFrame()

    model_summary.to_csv(OUT_DIR / "spatial_model_summary_by_cluster_k2.csv", index=False, encoding="utf-8-sig")
    spatial_tuning.to_csv(OUT_DIR / "slm_sem_knn_tuning_by_cluster_k2.csv", index=False, encoding="utf-8-sig")
    coef_df.to_csv(OUT_DIR / "ols_coefficients_by_cluster_k2.csv", index=False, encoding="utf-8-sig")
    gwr_local_df.to_csv(OUT_DIR / "gwr_local_diagnostics_by_cluster_k2.csv", index=False, encoding="utf-8-sig")
    mgwr_bw_df.to_csv(OUT_DIR / "mgwr_bandwidth_by_variable_k2.csv", index=False, encoding="utf-8-sig")
    save_model_png(model_summary, OUT_DIR / "cluster_k2_model_performance.png")

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
    (OUT_DIR / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"best_feature_set={best_feature_set}")
    print(model_summary.to_string(index=False))
    print(f"saved={OUT_DIR}")


if __name__ == "__main__":
    main()
