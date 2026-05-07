# -*- coding: utf-8 -*-
"""
Run full-row GWR/MGWR from 0430/최종테이블0429.csv and export visualizations.

Usage:
    cd NJT-PJT
    python scripts/run_full_gwr_mgwr_from_final_table.py

Outputs:
    data/full_gwr_mgwr/gwr_results_full.csv
    data/full_gwr_mgwr/mgwr_results_full.csv
    data/full_gwr_mgwr/maps/*.png

Notes:
    GWR is fit once on all valid rows.
    MGWR is fit separately inside each risk cluster, because MGWR estimates
    variable-specific bandwidths and is much heavier than GWR.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from mgwr.gwr import GWR, MGWR
from mgwr.sel_bw import Sel_BW
from shapely.validation import make_valid
from sklearn.preprocessing import StandardScaler


BASE = Path(__file__).resolve().parents[1]
DEFAULT_TABLE = BASE / "0430" / "최종테이블0429.csv"
DEFAULT_BOUNDARY = BASE / "data" / "seoul_legal_dong_age_buckets_joined_0415.geojson"
DEFAULT_OUT = BASE / "data" / "full_gwr_mgwr"

TARGET = "최종위험점수_new"
COORDS = ["x_5181", "y_5181"]
LAT_LON = ["위도", "경도"]
DEFAULT_FEATURES = [
    "구조노후도",
    "단속위험도",
    "도로폭위험도",
    "최근접_소화용수_거리등급",
    "소방위험도_점수",
    "집중도",
]

GU_BY_CODE = {
    "11110": "종로구",
    "11140": "중구",
    "11170": "용산구",
    "11200": "성동구",
    
    "11440": "마포구",
    
    "11500": "강서구",
    
    "11560": "영등포구",
   
    "11650": "서초구",
    "11680": "강남구",
    "11710": "송파구",
    
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--boundary", type=Path, default=DEFAULT_BOUNDARY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--target", default=TARGET)
    parser.add_argument("--features", nargs="*", default=DEFAULT_FEATURES)
    parser.add_argument("--kernel", default="bisquare", choices=["bisquare", "gaussian"])
    parser.add_argument("--skip-gwr", action="store_true")
    parser.add_argument("--skip-mgwr", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=1)
    return parser.parse_args()


def load_model_frame(table: Path, target: str, features: list[str]) -> pd.DataFrame:
    df = pd.read_csv(table, encoding="utf-8-sig")
    required = ["구", "동", "숙소명", "cluster", "cluster_label", *LAT_LON, *COORDS, target, *features]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {table}: {missing}")

    work = df[required].copy()
    for col in [*LAT_LON, *COORDS, target, *features]:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna(subset=[*COORDS, target, *features]).reset_index(drop=True)
    if work.empty:
        raise ValueError("No valid rows after dropping missing model fields.")
    return work


def standardize_xy(df: pd.DataFrame, target: str, features: list[str]):
    coords = df[COORDS].astype(float).to_numpy()
    y = df[target].astype(float).to_numpy().reshape((-1, 1))
    x = StandardScaler().fit_transform(df[features].astype(float).to_numpy())
    return coords, y, x


def run_gwr(df: pd.DataFrame, target: str, features: list[str], out_dir: Path, kernel: str, n_jobs: int):
    coords, y, x = standardize_xy(df, target, features)
    print(f"[GWR] rows={len(df):,}, features={len(features)}")
    t0 = time.time()
    selector = Sel_BW(coords, y, x, kernel=kernel, fixed=False, n_jobs=n_jobs)
    bw = selector.search(search_method="golden_section")
    print(f"[GWR] selected BW={int(bw)} in {time.time() - t0:.1f}s")

    t0 = time.time()
    result = GWR(coords, y, x, bw=bw, kernel=kernel, fixed=False, n_jobs=n_jobs).fit()
    print(f"[GWR] fit done in {time.time() - t0:.1f}s, R2={result.R2:.4f}, adj_R2={result.adj_R2:.4f}")

    out = df[["구", "동", "숙소명", *LAT_LON, *COORDS, target]].copy()
    out["local_R2"] = np.asarray(result.localR2).reshape(-1)
    out["bandwidth"] = int(bw)
    out["residual"] = np.asarray(result.resid_response).reshape(-1)
    out["coef_intercept"] = result.params[:, 0]
    if hasattr(result, "tvalues"):
        out["tval_intercept"] = result.tvalues[:, 0]
    for i, feature in enumerate(features, start=1):
        out[f"coef_{feature}"] = result.params[:, i]
        if hasattr(result, "tvalues"):
            out[f"tval_{feature}"] = result.tvalues[:, i]
    path = out_dir / "gwr_results_full.csv"
    out.to_csv(path, index=False, encoding="utf-8-sig")
    return out, {"model": "GWR", "rows": len(out), "bandwidth": int(bw), "R2": float(result.R2), "adj_R2": float(result.adj_R2)}


def as_bandwidth_list(raw_bw, expected_len: int) -> list[float]:
    if isinstance(raw_bw, tuple):
        raw_bw = raw_bw[0]
    arr = np.asarray(raw_bw, dtype=float).reshape(-1)
    if len(arr) != expected_len:
        return [float("nan")] * expected_len
    return arr.tolist()


def fit_one_mgwr_group(
    group: pd.DataFrame,
    cluster_id,
    target: str,
    features: list[str],
    kernel: str,
    n_jobs: int,
) -> tuple[pd.DataFrame, dict]:
    coords, y, x = standardize_xy(group, target, features)
    label = group["cluster_label"].dropna().iloc[0] if group["cluster_label"].notna().any() else str(cluster_id)
    print(f"[MGWR] cluster={cluster_id} ({label}), rows={len(group):,}, features={len(features)}")
    print("[MGWR] variable-specific bandwidth search runs inside this cluster.")
    t0 = time.time()
    selector = Sel_BW(coords, y, x, multi=True, kernel=kernel, fixed=False, n_jobs=n_jobs)
    selector.search(verbose=True)
    bandwidths = as_bandwidth_list(selector.bw, len(features) + 1)
    print(f"[MGWR] cluster={cluster_id} selected BW={bandwidths} in {time.time() - t0:.1f}s")

    t0 = time.time()
    result = MGWR(coords, y, x, selector, kernel=kernel, fixed=False, n_jobs=n_jobs).fit()
    print(
        f"[MGWR] cluster={cluster_id} fit done in {time.time() - t0:.1f}s, "
        f"R2={result.R2:.4f}, adj_R2={result.adj_R2:.4f}"
    )

    out = group[["구", "동", "숙소명", "cluster", "cluster_label", *LAT_LON, *COORDS, target]].copy()
    try:
        out["local_R2"] = np.asarray(result.localR2).reshape(-1)
    except NotImplementedError:
        out["local_R2"] = np.nan
    out["residual"] = np.asarray(result.resid_response).reshape(-1)
    terms = ["intercept", *features]
    for i, term in enumerate(terms):
        out[f"coef_{term}"] = result.params[:, i]
        if hasattr(result, "tvalues"):
            out[f"tval_{term}"] = result.tvalues[:, i]
        out[f"bw_{term}"] = bandwidths[i] if i < len(bandwidths) else np.nan
    for i, feature in enumerate(features):
        z = x[:, i]
        out[f"z_{feature}"] = z
        out[f"contrib_{feature}"] = out[f"coef_{feature}"] * z
    metrics = {
        "model": "MGWR",
        "cluster": int(cluster_id) if pd.notna(cluster_id) else None,
        "cluster_label": str(label),
        "rows": len(out),
        "bandwidths": bandwidths,
        "R2": float(result.R2),
        "adj_R2": float(result.adj_R2),
    }
    return out, metrics


def run_mgwr_clusterwise(
    df: pd.DataFrame,
    target: str,
    features: list[str],
    out_dir: Path,
    kernel: str,
    n_jobs: int,
):
    print("[MGWR] cluster-wise mode: fit one MGWR per risk cluster, using all rows in that cluster.")
    outputs = []
    metrics = []
    for cluster_id in sorted(df["cluster"].dropna().unique()):
        group = df[df["cluster"] == cluster_id].reset_index(drop=True)
        if len(group) <= len(features) + 2:
            print(f"[MGWR] skip cluster={cluster_id}: too few rows ({len(group):,})")
            continue
        out, metric = fit_one_mgwr_group(group, cluster_id, target, features, kernel, n_jobs)
        outputs.append(out)
        metrics.append(metric)
    if not outputs:
        raise RuntimeError("MGWR produced no cluster outputs.")
    result = pd.concat(outputs, ignore_index=True)
    path = out_dir / "mgwr_results_full.csv"
    result.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[MGWR] saved cluster-wise combined results: {path} rows={len(result):,}")
    return result, metrics


def load_boundary(path: Path) -> gpd.GeoDataFrame:
    boundary = gpd.read_file(path)
    boundary["구"] = boundary["EMD_CD"].astype(str).str[:5].map(GU_BY_CODE)
    boundary["동"] = boundary["EMD_KOR_NM"]
    boundary = boundary.to_crs(epsg=5179)
    boundary["geometry"] = boundary.geometry.apply(make_valid).buffer(0)
    return boundary[boundary.geometry.notna() & ~boundary.geometry.is_empty].copy()


def rank_0_100(s: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=s.index, dtype=float)
    valid = s.dropna()
    if valid.empty:
        return out
    if valid.max() == valid.min():
        out.loc[valid.index] = 50.0
    else:
        out.loc[valid.index] = 100 * (valid.rank(method="average") - 1) / (len(valid) - 1)
    return out


def fill_idw(gdf: gpd.GeoDataFrame, value_col: str, out_col: str, k: int = 5) -> gpd.GeoDataFrame:
    gdf[out_col] = gdf[value_col]
    points = gdf.geometry.representative_point()
    gdf["_x"] = points.x
    gdf["_y"] = points.y
    valid = gdf[gdf[value_col].notna()]
    missing = gdf[gdf[value_col].isna()]
    if valid.empty:
        gdf[out_col] = 0.0
        return gdf
    vx, vy, vv = valid["_x"].to_numpy(), valid["_y"].to_numpy(), valid[value_col].to_numpy(dtype=float)
    for idx, row in missing.iterrows():
        dist = np.sqrt((vx - row["_x"]) ** 2 + (vy - row["_y"]) ** 2)
        order = np.argsort(dist)[: min(k, len(dist))]
        d = np.maximum(dist[order], 1.0)
        weights = 1 / (d**2)
        gdf.at[idx, out_col] = float(np.sum(weights * vv[order]) / np.sum(weights))
    return gdf


def plot_variable_maps(
    boundary: gpd.GeoDataFrame,
    model_outputs: dict[str, pd.DataFrame],
    features: list[str],
    maps_dir: Path,
):
    maps_dir.mkdir(parents=True, exist_ok=True)
    matplotlib.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    gu_boundary = boundary.dissolve(by="구", as_index=False, method="unary", grid_size=0.05)
    cmap = matplotlib.colormaps["YlOrRd"]
    norm = Normalize(vmin=0, vmax=100)

    for feature in features:
        map_frames = {}
        for model_name, out in model_outputs.items():
            if model_name == "GWR":
                raw_col = f"strength_{feature}"
                out = out.copy()
                t_col = f"tval_{feature}"
                if t_col in out.columns:
                    out[raw_col] = (out[f"coef_{feature}"] * out[t_col]).abs()
                else:
                    out[raw_col] = out[f"coef_{feature}"].abs()
            else:
                raw_col = f"strength_{feature}"
                out = out.copy()
                contrib = f"contrib_{feature}"
                if contrib in out.columns:
                    out[raw_col] = out[contrib].abs()
                else:
                    out[raw_col] = out[f"coef_{feature}"].abs()

            agg = out.groupby(["구", "동"], dropna=False).agg(raw_strength=(raw_col, "mean")).reset_index()
            agg["strength"] = rank_0_100(agg["raw_strength"])
            gdf = boundary.merge(agg, on=["구", "동"], how="left")
            gdf = fill_idw(gdf, "strength", "strength_full")
            map_frames[model_name] = gdf

        fig, axes = plt.subplots(1, len(map_frames), figsize=(20, 11.5), dpi=180)
        if len(map_frames) == 1:
            axes = [axes]
        fig.patch.set_facecolor("#f7f9fc")
        for ax, (model_name, gdf) in zip(axes, map_frames.items()):
            ax.set_facecolor("#f7f9fc")
            gdf.plot(ax=ax, column="strength_full", cmap=cmap, norm=norm, edgecolor="#c9d1dc", linewidth=0.14)
            gu_boundary.boundary.plot(ax=ax, color="#2f3642", linewidth=0.8, alpha=0.9)
            for _, row in gu_boundary.iterrows():
                point = row.geometry.representative_point()
                ax.text(point.x, point.y, row["구"], ha="center", va="center", fontsize=6, weight="bold")
            ax.set_axis_off()
            ax.set_title(f"{model_name} {feature} 상대강도", fontsize=16, weight="bold", loc="left")

        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.012)
        cbar.set_label("모형 내 상대강도 0-100", fontsize=9)
        fig.suptitle(f"{feature}: GWR 전체 vs MGWR 군집별 전체 비교", fontsize=22, weight="bold", x=0.03, ha="left")
        fig.text(0.03, 0.035, "직접값 없는 법정동은 인접 법정동 IDW로 전체 경계를 채움", fontsize=9, color="#667085")
        fig.savefig(maps_dir / f"{feature}_gwr_mgwr_full.png", bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    maps_dir = args.out_dir / "maps"
    df = load_model_frame(args.table, args.target, args.features)
    print(f"[DATA] valid rows={len(df):,} from {args.table}")

    metadata = {
        "table": str(args.table),
        "target": args.target,
        "features": args.features,
        "rows": int(len(df)),
        "kernel": args.kernel,
    }
    outputs = {}
    metrics = []
    if not args.skip_gwr:
        gwr_out, gwr_metrics = run_gwr(df, args.target, args.features, args.out_dir, args.kernel, args.n_jobs)
        outputs["GWR"] = gwr_out
        metrics.append(gwr_metrics)
    if not args.skip_mgwr:
        mgwr_out, mgwr_metrics = run_mgwr_clusterwise(
            df,
            args.target,
            args.features,
            args.out_dir,
            args.kernel,
            args.n_jobs,
        )
        outputs["MGWR"] = mgwr_out
        metrics.extend(mgwr_metrics)

    metadata["metrics"] = metrics
    (args.out_dir / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.boundary.exists() and outputs:
        boundary = load_boundary(args.boundary)
        plot_variable_maps(boundary, outputs, args.features, maps_dir)
        print(f"[MAP] saved maps to {maps_dir}")
    else:
        print("[MAP] boundary not found or no model output; skipped maps.")


if __name__ == "__main__":
    main()
