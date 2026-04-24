from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from esda.moran import Moran
from libpysal.weights import KNN
from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW
from shapely.geometry import Point
from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from spreg import ML_Error, ML_Lag, OLS


ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT / "0424" / "분석" / "tables" / "분석변수_최종테이블0423_AHP3등급비교.csv"
TABLE_DIR = ROOT / "0424" / "분석" / "tables"
FIG_DIR = ROOT / "0424" / "분석" / "figures"

TARGET_COL = "위험점수_AHP"
FEATURES = [
    "승인연도",
    "주변건물수",
    "집중도",
    "단속위험도",
    "구조노후도",
    "도로폭위험도",
    "총층수",
    "시설규모/연면적",
    "외국인도시민박업여부",
]
SPATIAL_FEATURES = [
    "주변건물수",
    "집중도",
    "단속위험도",
    "구조노후도",
    "도로폭위험도",
]


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(INPUT_PATH, encoding="utf-8-sig")
    df = df.copy()
    df["외국인도시민박업여부"] = (df["업종"] == "외국인관광도시민박업").astype(int)
    base_cols = FEATURES + [TARGET_COL, "위도", "경도", "구", "동", "숙소명", "업종"]
    df = df[base_cols].copy()
    for col in FEATURES + [TARGET_COL, "위도", "경도"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=FEATURES + [TARGET_COL, "위도", "경도"]).reset_index(drop=True)
    return df


def build_projected_coords(df: pd.DataFrame) -> tuple[np.ndarray, gpd.GeoDataFrame]:
    gdf = gpd.GeoDataFrame(
        df.copy(),
        geometry=[Point(xy) for xy in zip(df["경도"], df["위도"])],
        crs="EPSG:4326",
    )
    gdf_proj = gdf.to_crs(epsg=5179)
    coords = np.column_stack([gdf_proj.geometry.x.to_numpy(), gdf_proj.geometry.y.to_numpy()])
    return coords, gdf_proj


def regression_holdout(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    X = df[FEATURES].to_numpy(dtype=float)
    y = df[TARGET_COL].to_numpy(dtype=float)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "OLS": LinearRegression(),
        "Ridge": RidgeCV(alphas=np.logspace(-3, 3, 25)),
        "Lasso": LassoCV(alphas=np.logspace(-3, 1.5, 30), cv=5, random_state=42, max_iter=20000),
    }

    rows: list[dict[str, object]] = []
    fitted: dict[str, object] = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        pred = model.predict(X_test_scaled)
        rows.append(
            {
                "model": name,
                "test_r2": r2_score(y_test, pred),
                "test_rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
                "test_mae": mean_absolute_error(y_test, pred),
                "alpha": getattr(model, "alpha_", np.nan) if name != "OLS" else np.nan,
            }
        )
        fitted[name] = model

    metrics = pd.DataFrame(rows).sort_values("test_r2", ascending=False).reset_index(drop=True)
    full_scaler = StandardScaler()
    X_full_scaled = full_scaler.fit_transform(X)
    full_models = {
        "OLS": LinearRegression().fit(X_full_scaled, y),
        "Ridge": RidgeCV(alphas=np.logspace(-3, 3, 25)).fit(X_full_scaled, y),
        "Lasso": LassoCV(alphas=np.logspace(-3, 1.5, 30), cv=5, random_state=42, max_iter=20000).fit(X_full_scaled, y),
    }
    residuals = {name: y - model.predict(X_full_scaled) for name, model in full_models.items()}

    full_info = {
        "scaler": full_scaler,
        "X_scaled": X_full_scaled,
        "y": y,
        "residuals": residuals,
        "models": full_models,
    }
    return metrics, full_info


def build_weights(coords: np.ndarray) -> KNN:
    w = KNN.from_array(coords, k=15)
    w.transform = "r"
    return w


def moran_residuals(residuals: dict[str, np.ndarray], w: KNN) -> pd.DataFrame:
    rows = []
    for name, resid in residuals.items():
        moran = Moran(np.asarray(resid).flatten(), w, permutations=999)
        rows.append(
            {
                "model": name,
                "moran_I": moran.I,
                "expected_I": moran.EI,
                "z_score": moran.z_sim,
                "p_value": moran.p_sim,
            }
        )
    return pd.DataFrame(rows).sort_values("moran_I", ascending=False).reset_index(drop=True)


def run_spatial_models(
    X_spatial_scaled: np.ndarray,
    y: np.ndarray,
    coords: np.ndarray,
    w: KNN,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    y2 = y.reshape((-1, 1))

    ols = OLS(y2, X_spatial_scaled, w=w, spat_diag=False, name_y=TARGET_COL, name_x=SPATIAL_FEATURES)
    lag = ML_Lag(y2, X_spatial_scaled, w=w, name_y=TARGET_COL, name_x=SPATIAL_FEATURES)
    error = ML_Error(y2, X_spatial_scaled, w=w, name_y=TARGET_COL, name_x=SPATIAL_FEATURES)

    selector = Sel_BW(coords, y2, X_spatial_scaled, fixed=False, kernel="bisquare", n_jobs=1)
    bw = selector.search(bw_min=140, bw_max=1200)
    gwr = GWR(coords, y2, X_spatial_scaled, bw, fixed=False, kernel="bisquare", n_jobs=1).fit()

    resid_map = {
        "Spatial_OLS": np.asarray(ols.u).flatten(),
        "Spatial_Lag": np.asarray(lag.u).flatten(),
        "Spatial_Error": np.asarray(error.u).flatten(),
        "GWR": np.asarray(gwr.resid_response).flatten(),
    }
    moran_df = moran_residuals(resid_map, w)

    summary_rows = [
        {
            "model": "Spatial_OLS",
            "r2_like": float(getattr(ols, "r2", np.nan)),
            "aic_like": float(getattr(ols, "aic", np.nan)),
            "residual_moran_I": float(moran_df.loc[moran_df["model"] == "Spatial_OLS", "moran_I"].iloc[0]),
        },
        {
            "model": "Spatial_Lag",
            "r2_like": float(getattr(lag, "pr2", np.nan)),
            "aic_like": float(getattr(lag, "aic", np.nan)),
            "residual_moran_I": float(moran_df.loc[moran_df["model"] == "Spatial_Lag", "moran_I"].iloc[0]),
        },
        {
            "model": "Spatial_Error",
            "r2_like": float(getattr(error, "pr2", np.nan)),
            "aic_like": float(getattr(error, "aic", np.nan)),
            "residual_moran_I": float(moran_df.loc[moran_df["model"] == "Spatial_Error", "moran_I"].iloc[0]),
        },
        {
            "model": "GWR",
            "r2_like": float(getattr(gwr, "R2", np.nan)),
            "aic_like": float(getattr(gwr, "aicc", np.nan)),
            "residual_moran_I": float(moran_df.loc[moran_df["model"] == "GWR", "moran_I"].iloc[0]),
        },
    ]
    summary_df = pd.DataFrame(summary_rows).sort_values("r2_like", ascending=False).reset_index(drop=True)

    params = np.asarray(gwr.params)
    gwr_local = pd.DataFrame(
        {
            "경도": coords[:, 0],
            "위도": coords[:, 1],
            "local_R2": np.asarray(gwr.localR2).flatten(),
            "coef_intercept": params[:, 0],
            "coef_구조노후도": params[:, SPATIAL_FEATURES.index("구조노후도") + 1],
            "coef_도로폭위험도": params[:, SPATIAL_FEATURES.index("도로폭위험도") + 1],
        }
    )

    extras = {
        "ols": ols,
        "lag": lag,
        "error": error,
        "gwr": gwr,
        "bandwidth": bw,
    }
    return summary_df, moran_df, {"gwr_local": gwr_local, "extras": extras}


def plot_regularized(metrics: pd.DataFrame, out_path: Path) -> None:
    ordered = metrics.sort_values("test_r2", ascending=True).reset_index(drop=True)
    y = np.arange(len(ordered))

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.4))
    axes[0].barh(y, ordered["test_r2"], color=["#577590", "#81B29A", "#E07A5F"])
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(ordered["model"])
    axes[0].set_title("Holdout Test R2")
    axes[0].grid(axis="x", alpha=0.18)

    axes[1].barh(y, ordered["test_rmse"], color=["#577590", "#81B29A", "#E07A5F"])
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(ordered["model"])
    axes[1].set_title("Holdout Test RMSE")
    axes[1].grid(axis="x", alpha=0.18)

    fig.suptitle("Regularized Regression on AHP Risk Score")
    plt.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_moran(moran_df: pd.DataFrame, out_path: Path) -> None:
    ordered = moran_df.sort_values("moran_I", ascending=True).reset_index(drop=True)
    y = np.arange(len(ordered))

    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    ax.barh(y, ordered["moran_I"], color="#6D597A", alpha=0.9)
    ax.axvline(0, color="#999999", linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(ordered["model"])
    ax.set_xlabel("Moran's I")
    ax.set_title("Residual Spatial Autocorrelation")
    ax.grid(axis="x", alpha=0.18)
    for idx, row in ordered.iterrows():
        ax.text(row["moran_I"] + 0.003, idx, f"p={row['p_value']:.3f}", va="center", fontsize=9, color="#333333")
    plt.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_spatial_summary(summary_df: pd.DataFrame, out_path: Path) -> None:
    ordered = summary_df.copy()
    ordered["aic_rank_score"] = ordered["aic_like"].max() - ordered["aic_like"]

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.6))
    axes[0].bar(ordered["model"], ordered["r2_like"], color=["#577590", "#81B29A", "#E07A5F", "#264653"])
    axes[0].set_title("Model Fit (R2-like)")
    axes[0].grid(axis="y", alpha=0.18)
    axes[0].tick_params(axis="x", rotation=20)

    axes[1].bar(ordered["model"], ordered["residual_moran_I"], color=["#577590", "#81B29A", "#E07A5F", "#264653"])
    axes[1].axhline(0, color="#999999", linewidth=1.0)
    axes[1].set_title("Residual Moran's I")
    axes[1].grid(axis="y", alpha=0.18)
    axes[1].tick_params(axis="x", rotation=20)

    fig.suptitle("Spatial Model Comparison")
    plt.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_gwr(gdf_proj: gpd.GeoDataFrame, gwr_local: pd.DataFrame, out_path: Path) -> None:
    plot_df = gdf_proj.copy()
    plot_df["local_R2"] = gwr_local["local_R2"].to_numpy()
    plot_df["coef_구조노후도"] = gwr_local["coef_구조노후도"].to_numpy()
    plot_df = plot_df.to_crs(epsg=4326)

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 6.0))
    sc1 = axes[0].scatter(
        plot_df.geometry.x,
        plot_df.geometry.y,
        c=plot_df["local_R2"],
        s=22,
        cmap="YlOrRd",
        alpha=0.85,
        linewidths=0,
    )
    axes[0].set_title("GWR Local R2")
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")
    fig.colorbar(sc1, ax=axes[0], fraction=0.046, pad=0.04)

    sc2 = axes[1].scatter(
        plot_df.geometry.x,
        plot_df.geometry.y,
        c=plot_df["coef_구조노후도"],
        s=22,
        cmap="coolwarm",
        alpha=0.85,
        linewidths=0,
    )
    axes[1].set_title("GWR Local Coef: Structural Aging")
    axes[1].set_xlabel("Longitude")
    axes[1].set_ylabel("Latitude")
    fig.colorbar(sc2, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset()
    coords, gdf_proj = build_projected_coords(df)
    reg_metrics, reg_info = regression_holdout(df)
    spatial_scaler = StandardScaler()
    X_spatial_scaled = spatial_scaler.fit_transform(df[SPATIAL_FEATURES].to_numpy(dtype=float))

    w = build_weights(coords)
    moran_reg_df = moran_residuals(reg_info["residuals"], w)
    spatial_summary, spatial_moran_df, extras = run_spatial_models(X_spatial_scaled, reg_info["y"], coords, w)

    reg_metrics_path = TABLE_DIR / "ahp_regularized_regression_metrics.csv"
    moran_reg_path = TABLE_DIR / "ahp_regularized_residual_moran.csv"
    spatial_summary_path = TABLE_DIR / "ahp_spatial_model_summary.csv"
    spatial_moran_path = TABLE_DIR / "ahp_spatial_model_residual_moran.csv"
    gwr_local_path = TABLE_DIR / "ahp_gwr_local_diagnostics.csv"
    meta_path = TABLE_DIR / "ahp_spatial_pipeline_metadata.json"

    reg_fig = FIG_DIR / "ahp_regularized_regression_comparison.png"
    moran_fig = FIG_DIR / "ahp_moran_residuals.png"
    spatial_fig = FIG_DIR / "ahp_spatial_model_comparison.png"
    gwr_fig = FIG_DIR / "ahp_gwr_local_patterns.png"

    reg_metrics.to_csv(reg_metrics_path, index=False, encoding="utf-8-sig")
    moran_reg_df.to_csv(moran_reg_path, index=False, encoding="utf-8-sig")
    spatial_summary.to_csv(spatial_summary_path, index=False, encoding="utf-8-sig")
    spatial_moran_df.to_csv(spatial_moran_path, index=False, encoding="utf-8-sig")
    extras["gwr_local"].to_csv(gwr_local_path, index=False, encoding="utf-8-sig")

    metadata = {
        "rows": int(len(df)),
        "features": FEATURES,
        "spatial_features": SPATIAL_FEATURES,
        "target": TARGET_COL,
        "weights": "KNN k=15 row-standardized",
        "gwr_bandwidth": float(extras["extras"]["bandwidth"]),
    }
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    plot_regularized(reg_metrics, reg_fig)
    plot_moran(pd.concat([moran_reg_df, spatial_moran_df], ignore_index=True), moran_fig)
    plot_spatial_summary(spatial_summary, spatial_fig)
    plot_gwr(gdf_proj, extras["gwr_local"], gwr_fig)

    print(reg_metrics.to_string(index=False))
    print()
    print(spatial_summary.to_string(index=False))
    print(f"\nSaved metrics: {reg_metrics_path}")
    print(f"Saved moran: {moran_reg_path}")
    print(f"Saved spatial summary: {spatial_summary_path}")
    print(f"Saved gwr local: {gwr_local_path}")
    print(f"Saved figure: {reg_fig}")
    print(f"Saved figure: {moran_fig}")
    print(f"Saved figure: {spatial_fig}")
    print(f"Saved figure: {gwr_fig}")


if __name__ == "__main__":
    main()
