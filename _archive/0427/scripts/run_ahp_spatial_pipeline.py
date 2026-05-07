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
INPUT_PATH = ROOT / "0424" / "분析" / "tables" / "분析변수_최종테이블0423_AHP3등급비교.csv"
TABLE_DIR = ROOT / "0424" / "분析" / "tables"
FIG_DIR = ROOT / "0424" / "분析" / "figures"

TARGET_COL = "위험점수_AHP"

# 그룹별 분석 — 업종더미 제거, 그룹 단독 실행
FEATURES = [
    "승인연도",
    "주변건물수",
    "집중도",
    "단속위험도",
    "구조노후도",
    "도로폭위험도",
    "총층수",
    "시설규모/연면적",
]
SPATIAL_FEATURES = [
    "주변건물수",
    "집중도",
    "단속위험도",
    "구조노후도",
    "도로폭위험도",
]

GROUPS = {
    "기존숙박군": "A_기존숙박군",
    "외국인관광도시민박업": "B_외국인민박",
}


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(INPUT_PATH, encoding="utf-8-sig")
    base_cols = FEATURES + [TARGET_COL, "위도", "경도", "구", "동", "숙소명", "업종", "업종그룹"]
    df = df[base_cols].copy()
    for col in FEATURES + [TARGET_COL, "위도", "경도"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
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


def regression_holdout(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    X = df[FEATURES].to_numpy(dtype=float)
    y = df[TARGET_COL].to_numpy(dtype=float)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = {
        "OLS": LinearRegression(),
        "Ridge": RidgeCV(alphas=np.logspace(-3, 3, 25)),
        "Lasso": LassoCV(alphas=np.logspace(-3, 1.5, 30), cv=5, random_state=42, max_iter=20000),
    }
    rows = []
    for name, model in models.items():
        model.fit(X_train_s, y_train)
        pred = model.predict(X_test_s)
        rows.append({
            "model": name,
            "test_r2": r2_score(y_test, pred),
            "test_rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
            "test_mae": mean_absolute_error(y_test, pred),
            "alpha": getattr(model, "alpha_", np.nan) if name != "OLS" else np.nan,
        })

    metrics = pd.DataFrame(rows).sort_values("test_r2", ascending=False).reset_index(drop=True)
    full_scaler = StandardScaler()
    X_full_s = full_scaler.fit_transform(X)
    full_models = {
        "OLS": LinearRegression().fit(X_full_s, y),
        "Ridge": RidgeCV(alphas=np.logspace(-3, 3, 25)).fit(X_full_s, y),
        "Lasso": LassoCV(alphas=np.logspace(-3, 1.5, 30), cv=5, random_state=42, max_iter=20000).fit(X_full_s, y),
    }
    residuals = {name: y - m.predict(X_full_s) for name, m in full_models.items()}
    full_info = {"scaler": full_scaler, "X_scaled": X_full_s, "y": y, "residuals": residuals, "models": full_models}
    return metrics, full_info


def build_weights(coords: np.ndarray) -> KNN:
    w = KNN.from_array(coords, k=15)
    w.transform = "r"
    return w


def moran_residuals(residuals: dict[str, np.ndarray], w: KNN) -> pd.DataFrame:
    rows = []
    for name, resid in residuals.items():
        moran = Moran(np.asarray(resid).flatten(), w, permutations=999)
        rows.append({
            "model": name,
            "moran_I": moran.I,
            "expected_I": moran.EI,
            "z_score": moran.z_sim,
            "p_value": moran.p_sim,
        })
    return pd.DataFrame(rows).sort_values("moran_I", ascending=False).reset_index(drop=True)


def run_spatial_models(
    X_spatial_scaled: np.ndarray,
    y: np.ndarray,
    coords: np.ndarray,
    w: KNN,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    y2 = y.reshape((-1, 1))

    ols = OLS(y2, X_spatial_scaled, w=w, spat_diag=False, name_y=TARGET_COL, name_x=SPATIAL_FEATURES)
    lag = ML_Lag(y2, X_spatial_scaled, w=w, name_y=TARGET_COL, name_x=SPATIAL_FEATURES)
    error = ML_Error(y2, X_spatial_scaled, w=w, name_y=TARGET_COL, name_x=SPATIAL_FEATURES)

    selector = Sel_BW(coords, y2, X_spatial_scaled, fixed=False, kernel="bisquare", n_jobs=1)
    bw = selector.search(bw_min=30, bw_max=1200)
    gwr = GWR(coords, y2, X_spatial_scaled, bw, fixed=False, kernel="bisquare", n_jobs=1).fit()

    resid_map = {
        "Spatial_OLS": np.asarray(ols.u).flatten(),
        "Spatial_Lag": np.asarray(lag.u).flatten(),
        "Spatial_Error": np.asarray(error.u).flatten(),
        "GWR": np.asarray(gwr.resid_response).flatten(),
    }
    moran_df = moran_residuals(resid_map, w)

    summary_rows = [
        {"model": "Spatial_OLS",   "r2_like": float(getattr(ols,   "r2",  np.nan)), "aic_like": float(getattr(ols,   "aic", np.nan)), "residual_moran_I": float(moran_df.loc[moran_df["model"] == "Spatial_OLS",   "moran_I"].iloc[0])},
        {"model": "Spatial_Lag",   "r2_like": float(getattr(lag,   "pr2", np.nan)), "aic_like": float(getattr(lag,   "aic", np.nan)), "residual_moran_I": float(moran_df.loc[moran_df["model"] == "Spatial_Lag",   "moran_I"].iloc[0])},
        {"model": "Spatial_Error", "r2_like": float(getattr(error, "pr2", np.nan)), "aic_like": float(getattr(error, "aic", np.nan)), "residual_moran_I": float(moran_df.loc[moran_df["model"] == "Spatial_Error", "moran_I"].iloc[0])},
        {"model": "GWR",           "r2_like": float(getattr(gwr,   "R2",  np.nan)), "aic_like": float(getattr(gwr,   "aicc",np.nan)), "residual_moran_I": float(moran_df.loc[moran_df["model"] == "GWR",           "moran_I"].iloc[0])},
    ]
    summary_df = pd.DataFrame(summary_rows).sort_values("r2_like", ascending=False).reset_index(drop=True)

    params = np.asarray(gwr.params)
    gwr_local = pd.DataFrame({
        "경도": coords[:, 0],
        "위도": coords[:, 1],
        "local_R2": np.asarray(gwr.localR2).flatten(),
        "coef_intercept": params[:, 0],
        "coef_구조노후도": params[:, SPATIAL_FEATURES.index("구조노후도") + 1],
        "coef_도로폭위험도": params[:, SPATIAL_FEATURES.index("도로폭위험도") + 1],
    })

    extras = {"ols": ols, "lag": lag, "error": error, "gwr": gwr, "bandwidth": bw}
    return summary_df, moran_df, {"gwr_local": gwr_local, "extras": extras}


# ── 플롯 함수들 ───────────────────────────────────────────────────────

def plot_regularized(metrics: pd.DataFrame, out_path: Path, title: str) -> None:
    ordered = metrics.sort_values("test_r2", ascending=True).reset_index(drop=True)
    y = np.arange(len(ordered))
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.4))
    axes[0].barh(y, ordered["test_r2"], color=["#577590", "#81B29A", "#E07A5F"])
    axes[0].set_yticks(y); axes[0].set_yticklabels(ordered["model"])
    axes[0].set_title("Holdout Test R²"); axes[0].grid(axis="x", alpha=0.18)
    axes[1].barh(y, ordered["test_rmse"], color=["#577590", "#81B29A", "#E07A5F"])
    axes[1].set_yticks(y); axes[1].set_yticklabels(ordered["model"])
    axes[1].set_title("Holdout Test RMSE"); axes[1].grid(axis="x", alpha=0.18)
    fig.suptitle(f"Regularized Regression — {title}")
    plt.tight_layout(); fig.savefig(out_path, dpi=220, bbox_inches="tight"); plt.close(fig)


def plot_moran(moran_df: pd.DataFrame, out_path: Path, title: str) -> None:
    ordered = moran_df.sort_values("moran_I", ascending=True).reset_index(drop=True)
    y = np.arange(len(ordered))
    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    ax.barh(y, ordered["moran_I"], color="#6D597A", alpha=0.9)
    ax.axvline(0, color="#999999", linewidth=1.0)
    ax.set_yticks(y); ax.set_yticklabels(ordered["model"])
    ax.set_xlabel("Moran's I"); ax.set_title(f"Residual Spatial Autocorrelation — {title}")
    ax.grid(axis="x", alpha=0.18)
    for idx, row in ordered.iterrows():
        ax.text(row["moran_I"] + 0.003, idx, f"p={row['p_value']:.3f}", va="center", fontsize=9)
    plt.tight_layout(); fig.savefig(out_path, dpi=220, bbox_inches="tight"); plt.close(fig)


def plot_spatial_summary(summary_df: pd.DataFrame, out_path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.6))
    axes[0].bar(summary_df["model"], summary_df["r2_like"], color=["#577590", "#81B29A", "#E07A5F", "#264653"])
    axes[0].set_title("Model Fit (R²-like)"); axes[0].grid(axis="y", alpha=0.18); axes[0].tick_params(axis="x", rotation=20)
    axes[1].bar(summary_df["model"], summary_df["residual_moran_I"], color=["#577590", "#81B29A", "#E07A5F", "#264653"])
    axes[1].axhline(0, color="#999999", linewidth=1.0)
    axes[1].set_title("Residual Moran's I"); axes[1].grid(axis="y", alpha=0.18); axes[1].tick_params(axis="x", rotation=20)
    fig.suptitle(f"Spatial Model Comparison — {title}")
    plt.tight_layout(); fig.savefig(out_path, dpi=220, bbox_inches="tight"); plt.close(fig)


def plot_gwr(gdf_proj: gpd.GeoDataFrame, gwr_local: pd.DataFrame, out_path: Path, title: str) -> None:
    plot_df = gdf_proj.copy()
    plot_df["local_R2"] = gwr_local["local_R2"].to_numpy()
    plot_df["coef_구조노후도"] = gwr_local["coef_구조노후도"].to_numpy()
    plot_df = plot_df.to_crs(epsg=4326)
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 6.0))
    sc1 = axes[0].scatter(plot_df.geometry.x, plot_df.geometry.y, c=plot_df["local_R2"], s=22, cmap="YlOrRd", alpha=0.85, linewidths=0)
    axes[0].set_title("GWR Local R²"); fig.colorbar(sc1, ax=axes[0], fraction=0.046, pad=0.04)
    sc2 = axes[1].scatter(plot_df.geometry.x, plot_df.geometry.y, c=plot_df["coef_구조노후도"], s=22, cmap="coolwarm", alpha=0.85, linewidths=0)
    axes[1].set_title("GWR Coef: 구조노후도"); fig.colorbar(sc2, ax=axes[1], fraction=0.046, pad=0.04)
    fig.suptitle(f"GWR Local Patterns — {title}")
    plt.tight_layout(); fig.savefig(out_path, dpi=220, bbox_inches="tight"); plt.close(fig)


# ── 그룹별 실행 ──────────────────────────────────────────────────────

def run_group(df_group: pd.DataFrame, group_key: str, slug: str) -> None:
    label = "기존숙박군 (숙박업+관광숙박업)" if group_key == "기존숙박군" else "외국인관광도시민박업"
    print(f"\n{'='*65}")
    print(f"  {label}  N={len(df_group):,}")
    print("=" * 65)

    group_table_dir = TABLE_DIR / slug
    group_fig_dir = FIG_DIR / slug
    group_table_dir.mkdir(parents=True, exist_ok=True)
    group_fig_dir.mkdir(parents=True, exist_ok=True)

    df_clean = df_group.dropna(subset=FEATURES + [TARGET_COL, "위도", "경도"]).reset_index(drop=True)
    print(f"  결측 제거 후: {len(df_clean):,}행")

    coords, gdf_proj = build_projected_coords(df_clean)
    reg_metrics, reg_info = regression_holdout(df_clean)

    spatial_scaler = StandardScaler()
    X_spatial_scaled = spatial_scaler.fit_transform(df_clean[SPATIAL_FEATURES].to_numpy(dtype=float))

    w = build_weights(coords)
    moran_reg_df = moran_residuals(reg_info["residuals"], w)
    spatial_summary, spatial_moran_df, extras = run_spatial_models(
        X_spatial_scaled, reg_info["y"], coords, w
    )

    # 저장
    reg_metrics.to_csv(group_table_dir / "regularized_regression_metrics.csv", index=False, encoding="utf-8-sig")
    moran_reg_df.to_csv(group_table_dir / "regularized_residual_moran.csv", index=False, encoding="utf-8-sig")
    spatial_summary.to_csv(group_table_dir / "spatial_model_summary.csv", index=False, encoding="utf-8-sig")
    spatial_moran_df.to_csv(group_table_dir / "spatial_model_residual_moran.csv", index=False, encoding="utf-8-sig")
    extras["gwr_local"].to_csv(group_table_dir / "gwr_local_diagnostics.csv", index=False, encoding="utf-8-sig")

    metadata = {
        "group": group_key,
        "label": label,
        "rows": int(len(df_clean)),
        "features": FEATURES,
        "spatial_features": SPATIAL_FEATURES,
        "target": TARGET_COL,
        "weights": "KNN k=15 row-standardized",
        "gwr_bandwidth": float(extras["extras"]["bandwidth"]),
    }
    (group_table_dir / "spatial_pipeline_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    plot_regularized(reg_metrics, group_fig_dir / "regularized_regression_comparison.png", label)
    plot_moran(pd.concat([moran_reg_df, spatial_moran_df], ignore_index=True), group_fig_dir / "moran_residuals.png", label)
    plot_spatial_summary(spatial_summary, group_fig_dir / "spatial_model_comparison.png", label)
    plot_gwr(gdf_proj, extras["gwr_local"], group_fig_dir / "gwr_local_patterns.png", label)

    print(f"\n  [OLS/Ridge/Lasso]")
    print(reg_metrics.to_string(index=False))
    print(f"\n  [공간모델 비교]")
    print(spatial_summary.to_string(index=False))
    print(f"\n  GWR Bandwidth: {extras['extras']['bandwidth']:.0f}")
    print(f"  결과 저장: {group_table_dir}")


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset()

    for group_key, slug in GROUPS.items():
        df_group = df[df["업종그룹"] == group_key].copy()
        run_group(df_group, group_key, slug)

    print(f"\n\n{'='*65}")
    print("  전체 완료")
    print("=" * 65)


if __name__ == "__main__":
    main()
