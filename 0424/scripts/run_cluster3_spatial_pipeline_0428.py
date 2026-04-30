# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import sys
import time
import warnings
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from esda.moran import Moran
from libpysal.weights import KNN
from mgwr.gwr import GWR, MGWR
from mgwr.sel_bw import Sel_BW
from sklearn.preprocessing import StandardScaler
from spreg import ML_Error, ML_Lag


warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = ROOT / "data" / "최최최종0428변수테이블.csv"
BUNDLE_DIR_150M = ROOT / "data" / "cluster3_spatial_pipeline_fire_count_150m_0428"
DATA_PATH = DEFAULT_DATA_PATH
FIRE_TARGET_PATH = ROOT.parents[0] / "data" / "team_pipeline_validation" / "team_pipeline_scored_dataset.csv"
OUT_DIR = ROOT / "data" / "cluster3_spatial_pipeline_0428"

TARGET = "최종_화재위험점수"
CLUSTER_COL = "cluster"
COORD_COLS = ["x_5181", "y_5181"]
FEATURES = [
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

RNG = np.random.RandomState(42)
MORAN_PERMUTATIONS = 199
GWR_SAMPLE_CAP = 700
MGWR_SAMPLE_CAP = 220


def name_key(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r"\s+", "", regex=True).str.strip()


def attach_external_target(df: pd.DataFrame, target: str) -> pd.DataFrame:
    if target in df.columns:
        return df
    if target != "fire_count_150m":
        raise ValueError(f"Target column not found: {target}")
    fire = pd.read_csv(FIRE_TARGET_PATH, encoding="utf-8-sig")
    fire_key = pd.DataFrame(
        {
            "_name_key": name_key(fire["숙소명"]),
            "_lat_key": pd.to_numeric(fire["위도"], errors="coerce").round(6),
            "_lon_key": pd.to_numeric(fire["경도"], errors="coerce").round(6),
            target: pd.to_numeric(fire[target], errors="coerce"),
        }
    ).drop_duplicates(["_name_key", "_lat_key", "_lon_key"])
    keyed = df.copy()
    keyed["_name_key"] = name_key(keyed["숙소명"])
    keyed["_lat_key"] = pd.to_numeric(keyed["위도"], errors="coerce").round(6)
    keyed["_lon_key"] = pd.to_numeric(keyed["경도"], errors="coerce").round(6)
    keyed = keyed.merge(fire_key, on=["_name_key", "_lat_key", "_lon_key"], how="left")
    return keyed.drop(columns=["_name_key", "_lat_key", "_lon_key"], errors="ignore")


def read_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    df = attach_external_target(df, TARGET)
    needed = FEATURES + [TARGET, CLUSTER_COL] + COORD_COLS + ["구", "동", "숙소명"]
    df = df[[c for c in needed if c in df.columns]].copy()
    for col in FEATURES + [TARGET, CLUSTER_COL] + COORD_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=FEATURES + [TARGET, CLUSTER_COL] + COORD_COLS).reset_index(drop=True)


def standardize_x(df: pd.DataFrame) -> np.ndarray:
    return StandardScaler().fit_transform(df[FEATURES].to_numpy(dtype=float))


def build_weights(coords: np.ndarray) -> KNN:
    k = min(12, max(1, len(coords) - 1))
    w = KNN.from_array(coords, k=k)
    w.transform = "r"
    return w


def aicc_of(result) -> float:
    for attr in ("aicc", "AICc", "aic", "AIC"):
        value = getattr(result, attr, np.nan)
        try:
            return float(value)
        except Exception:
            continue
    return float("nan")


def sample_for_local_model(df: pd.DataFrame, cap: int) -> pd.DataFrame:
    if len(df) <= cap:
        return df.copy().reset_index(drop=True)
    sampled_idx = RNG.choice(df.index.to_numpy(), cap, replace=False)
    return df.loc[np.sort(sampled_idx)].copy().reset_index(drop=True)


def run_ols_moran(df: pd.DataFrame, cluster_id: int) -> tuple[dict, pd.DataFrame, np.ndarray]:
    x = standardize_x(df)
    y = df[TARGET].to_numpy(dtype=float)
    coords = df[COORD_COLS].to_numpy(dtype=float)

    model = sm.OLS(y, sm.add_constant(x)).fit(cov_type="HC3")
    w = build_weights(coords)
    moran = Moran(model.resid, w, permutations=MORAN_PERMUTATIONS)

    coef_rows = []
    for term, coef, pval in zip(["const"] + FEATURES, model.params, model.pvalues):
        coef_rows.append(
            {
                "cluster": cluster_id,
                "term": term,
                "coef": float(coef),
                "p_value": float(pval),
            }
        )

    summary = {
        "cluster": cluster_id,
        "n": int(len(df)),
        "ols_r2": float(model.rsquared),
        "ols_adj_r2": float(model.rsquared_adj),
        "ols_aic": float(model.aic),
        "ols_resid_moran_I": float(moran.I),
        "ols_resid_moran_p": float(moran.p_sim),
    }
    return summary, pd.DataFrame(coef_rows), model.resid


def run_spatial_lag_error(df: pd.DataFrame, cluster_id: int) -> tuple[list[dict], dict[str, np.ndarray]]:
    x = standardize_x(df)
    y = df[TARGET].to_numpy(dtype=float).reshape(-1, 1)
    coords = df[COORD_COLS].to_numpy(dtype=float)
    w = build_weights(coords)

    rows: list[dict] = []
    residuals: dict[str, np.ndarray] = {}
    for model_name, model_cls in [("SLM", ML_Lag), ("SEM", ML_Error)]:
        t0 = time.time()
        try:
            model = model_cls(y, x, w=w, name_y=TARGET, name_x=FEATURES)
            resid = np.asarray(model.u).flatten()
            moran = Moran(resid, w, permutations=MORAN_PERMUTATIONS)
            residuals[model_name] = resid
            rows.append(
                {
                    "cluster": cluster_id,
                    "model": model_name,
                    "n": int(len(df)),
                    "fit": float(getattr(model, "pr2", getattr(model, "r2", np.nan))),
                    "aic": float(getattr(model, "aic", np.nan)),
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
                    "n": int(len(df)),
                    "fit": np.nan,
                    "aic": np.nan,
                    "resid_moran_I": np.nan,
                    "resid_moran_p": np.nan,
                    "seconds": round(time.time() - t0, 2),
                    "status": f"failed: {exc}",
                }
            )
    return rows, residuals


def select_gwr_bw(coords: np.ndarray, y: np.ndarray, x: np.ndarray) -> float:
    bw_min = max(30, x.shape[1] + 3)
    bw_max = max(bw_min + 2, min(len(y) - 1, 420))
    selector = Sel_BW(coords, y, x, fixed=False, kernel="bisquare", n_jobs=1)
    return float(selector.search(search_method="golden_section", bw_min=bw_min, bw_max=bw_max))


def run_gwr(df: pd.DataFrame, cluster_id: int) -> tuple[dict, pd.DataFrame, np.ndarray | None]:
    work = sample_for_local_model(df, GWR_SAMPLE_CAP)
    coords = work[COORD_COLS].to_numpy(dtype=float)
    y = work[TARGET].to_numpy(dtype=float).reshape(-1, 1)
    x = standardize_x(work)
    t0 = time.time()
    try:
        bw = select_gwr_bw(coords, y, x)
        result = GWR(coords, y, x, bw=bw, fixed=False, kernel="bisquare", n_jobs=1).fit()
        resid = np.asarray(result.resid_response).flatten()
        w = build_weights(coords)
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
        params = np.asarray(result.params)
        for i, feature in enumerate(["intercept"] + FEATURES):
            local[f"coef_{feature}"] = params[:, i]
        summary = {
            "cluster": cluster_id,
            "model": "GWR",
            "n": int(len(work)),
            "sampled": bool(len(work) < len(df)),
            "bandwidth": float(bw),
            "fit": float(result.R2),
            "adj_fit": float(result.adj_R2),
            "aic": aicc_of(result),
            "resid_moran_I": float(moran.I),
            "resid_moran_p": float(moran.p_sim),
            "seconds": round(time.time() - t0, 2),
            "status": "ok",
        }
        return summary, local, resid
    except Exception as exc:
        summary = {
            "cluster": cluster_id,
            "model": "GWR",
            "n": int(len(work)),
            "sampled": bool(len(work) < len(df)),
            "bandwidth": np.nan,
            "fit": np.nan,
            "adj_fit": np.nan,
            "aic": np.nan,
            "resid_moran_I": np.nan,
            "resid_moran_p": np.nan,
            "seconds": round(time.time() - t0, 2),
            "status": f"failed: {exc}",
        }
        return summary, pd.DataFrame(), None


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
        w = build_weights(coords)
        moran = Moran(resid, w, permutations=MORAN_PERMUTATIONS)
        bw_table = pd.DataFrame(
            {
                "cluster": cluster_id,
                "term": ["intercept"] + FEATURES,
                "bandwidth": bw_values[: len(FEATURES) + 1],
            }
        )
        summary = {
            "cluster": cluster_id,
            "model": "MGWR",
            "n": int(len(work)),
            "sampled": bool(len(work) < len(df)),
            "bandwidth": float(np.nanmean(bw_values)),
            "fit": float(result.R2),
            "adj_fit": float(result.adj_R2),
            "aic": aicc_of(result),
            "resid_moran_I": float(moran.I),
            "resid_moran_p": float(moran.p_sim),
            "seconds": round(time.time() - t0, 2),
            "status": "ok",
        }
        return summary, bw_table
    except Exception as exc:
        summary = {
            "cluster": cluster_id,
            "model": "MGWR",
            "n": int(len(work)),
            "sampled": bool(len(work) < len(df)),
            "bandwidth": np.nan,
            "fit": np.nan,
            "adj_fit": np.nan,
            "aic": np.nan,
            "resid_moran_I": np.nan,
            "resid_moran_p": np.nan,
            "seconds": round(time.time() - t0, 2),
            "status": f"failed: {exc}",
        }
        return summary, pd.DataFrame()


def save_dashboard(
    model_summary: pd.DataFrame,
    ols_summary: pd.DataFrame,
    gwr_local: pd.DataFrame,
    out_path: Path,
) -> None:
    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    colors = {0: "#4C78A8", 1: "#59A14F", 2: "#E15759"}

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    ax = axes[0, 0]
    for cid, row in ols_summary.sort_values("cluster").iterrows():
        ax.bar(
            str(int(row["cluster"])),
            row["ols_r2"],
            color=colors.get(int(row["cluster"]), "#777777"),
            alpha=0.9,
        )
        ax.text(str(int(row["cluster"])), row["ols_r2"], f"{row['ols_r2']:.3f}", ha="center", va="bottom")
    ax.set_title("OLS 설명력 by cluster")
    ax.set_xlabel("cluster")
    ax.set_ylabel("R-squared")
    ax.grid(axis="y", alpha=0.2)

    ax = axes[0, 1]
    moran_plot = model_summary.dropna(subset=["resid_moran_I"]).copy()
    for model, sub in moran_plot.groupby("model"):
        ax.plot(sub["cluster"], sub["resid_moran_I"], marker="o", label=model)
    ax.axhline(0, color="#999999", linewidth=1)
    ax.set_title("잔차 Moran's I: OLS -> SEM/SLM -> GWR/MGWR")
    ax.set_xlabel("cluster")
    ax.set_ylabel("Moran's I")
    ax.set_xticks(sorted(ols_summary["cluster"].astype(int).unique()))
    ax.grid(alpha=0.2)
    ax.legend(ncol=3, fontsize=9)

    ax = axes[1, 0]
    bw_plot = model_summary[model_summary["model"].isin(["GWR", "MGWR"])].dropna(subset=["bandwidth"])
    width = 0.34
    clusters = sorted(ols_summary["cluster"].astype(int).unique())
    for offset, model in [(-width / 2, "GWR"), (width / 2, "MGWR")]:
        vals = []
        for cid in clusters:
            matched = bw_plot[(bw_plot["cluster"] == cid) & (bw_plot["model"] == model)]
            vals.append(float(matched["bandwidth"].iloc[0]) if len(matched) else np.nan)
        ax.bar(np.asarray(clusters) + offset, vals, width=width, label=model)
    ax.set_title("최적 bandwidth")
    ax.set_xlabel("cluster")
    ax.set_ylabel("Adaptive nearest-neighbor BW")
    ax.set_xticks(clusters)
    ax.grid(axis="y", alpha=0.2)
    ax.legend()

    ax = axes[1, 1]
    if not gwr_local.empty:
        sc = ax.scatter(
            gwr_local["x_5181"],
            gwr_local["y_5181"],
            c=gwr_local["local_R2"],
            s=10,
            cmap="viridis",
            alpha=0.85,
            linewidths=0,
        )
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="GWR local R2")
    ax.set_title("GWR local R2 공간분포")
    ax.set_xlabel("x_5181")
    ax.set_ylabel("y_5181")
    ax.grid(alpha=0.15)

    fig.suptitle(f"3군집 공간회귀 파이프라인: {TARGET} / OLS -> Moran's I -> SEM/SLM -> GWR/MGWR", fontsize=17)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        default="최종_화재위험점수",
        choices=["최종_화재위험점수", "fire_count_150m"],
        help="Dependent variable for the spatial regression pipeline.",
    )
    parser.add_argument(
        "--bundle-dir",
        default=None,
        help="Folder containing the bundled input CSV and where result CSV/PNG files are written.",
    )
    return parser.parse_args()


def main() -> None:
    global TARGET, OUT_DIR, DATA_PATH
    args = parse_args()
    TARGET = args.target
    if TARGET == "fire_count_150m":
        OUT_DIR = Path(args.bundle_dir) if args.bundle_dir else BUNDLE_DIR_150M
        bundled_input = OUT_DIR / "최최최종0428변수테이블.csv"
        DATA_PATH = bundled_input if bundled_input.exists() else DEFAULT_DATA_PATH

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = read_dataset()

    ols_rows: list[dict] = []
    coef_tables: list[pd.DataFrame] = []
    model_rows: list[dict] = []
    local_tables: list[pd.DataFrame] = []
    mgwr_bw_tables: list[pd.DataFrame] = []

    for cluster_id in sorted(df[CLUSTER_COL].dropna().astype(int).unique()):
        sub = df[df[CLUSTER_COL].astype(int) == cluster_id].reset_index(drop=True)
        print(f"\n=== cluster {cluster_id} / n={len(sub):,} ===")

        ols_summary, coef_df, ols_resid = run_ols_moran(sub, cluster_id)
        ols_rows.append(ols_summary)
        coef_tables.append(coef_df)
        model_rows.append(
            {
                "cluster": cluster_id,
                "model": "OLS",
                "n": int(len(sub)),
                "sampled": False,
                "bandwidth": np.nan,
                "fit": ols_summary["ols_r2"],
                "adj_fit": ols_summary["ols_adj_r2"],
                "aic": ols_summary["ols_aic"],
                "resid_moran_I": ols_summary["ols_resid_moran_I"],
                "resid_moran_p": ols_summary["ols_resid_moran_p"],
                "seconds": np.nan,
                "status": "ok",
            }
        )
        print(
            f"OLS R2={ols_summary['ols_r2']:.4f}, Moran I={ols_summary['ols_resid_moran_I']:.4f}, "
            f"p={ols_summary['ols_resid_moran_p']:.4f}"
        )

        spatial_rows, _ = run_spatial_lag_error(sub, cluster_id)
        model_rows.extend(spatial_rows)
        for row in spatial_rows:
            print(f"{row['model']} fit={row['fit']:.4f}, Moran I={row['resid_moran_I']:.4f}, {row['status']}")

        gwr_summary, local_df, _ = run_gwr(sub, cluster_id)
        model_rows.append(gwr_summary)
        if not local_df.empty:
            local_tables.append(local_df)
        print(
            f"GWR BW={gwr_summary['bandwidth']}, fit={gwr_summary['fit']:.4f}, "
            f"Moran I={gwr_summary['resid_moran_I']:.4f}, {gwr_summary['status']}"
        )

        mgwr_summary, mgwr_bw_df = run_mgwr(sub, cluster_id)
        model_rows.append(mgwr_summary)
        if not mgwr_bw_df.empty:
            mgwr_bw_tables.append(mgwr_bw_df)
        print(
            f"MGWR mean BW={mgwr_summary['bandwidth']}, fit={mgwr_summary['fit']:.4f}, "
            f"Moran I={mgwr_summary['resid_moran_I']:.4f}, {mgwr_summary['status']}"
        )

    ols_summary_df = pd.DataFrame(ols_rows)
    coef_df = pd.concat(coef_tables, ignore_index=True)
    model_summary_df = pd.DataFrame(model_rows)
    gwr_local_df = pd.concat(local_tables, ignore_index=True) if local_tables else pd.DataFrame()
    mgwr_bw_df = pd.concat(mgwr_bw_tables, ignore_index=True) if mgwr_bw_tables else pd.DataFrame()

    ols_summary_df.to_csv(OUT_DIR / "ols_moran_by_cluster.csv", index=False, encoding="utf-8-sig")
    coef_df.to_csv(OUT_DIR / "ols_coefficients_by_cluster.csv", index=False, encoding="utf-8-sig")
    model_summary_df.to_csv(OUT_DIR / "spatial_model_summary_by_cluster.csv", index=False, encoding="utf-8-sig")
    gwr_local_df.to_csv(OUT_DIR / "gwr_local_diagnostics_by_cluster.csv", index=False, encoding="utf-8-sig")
    mgwr_bw_df.to_csv(OUT_DIR / "mgwr_bandwidth_by_variable.csv", index=False, encoding="utf-8-sig")

    metadata = {
        "input": str(DATA_PATH),
        "target": TARGET,
        "cluster_column": CLUSTER_COL,
        "features": FEATURES,
        "coordinates": COORD_COLS,
        "weights": "KNN k=min(12, n-1), row-standardized",
        "moran_permutations": MORAN_PERMUTATIONS,
        "gwr_sample_cap": GWR_SAMPLE_CAP,
        "mgwr_sample_cap": MGWR_SAMPLE_CAP,
        "note": "GWR/MGWR use adaptive bisquare bandwidth search. Large clusters are sampled for local models to keep the run reproducible on a laptop.",
    }
    (OUT_DIR / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    save_dashboard(model_summary_df, ols_summary_df, gwr_local_df, OUT_DIR / "cluster3_spatial_pipeline_dashboard.png")
    print(f"\nSaved results to: {OUT_DIR}")


if __name__ == "__main__":
    main()
