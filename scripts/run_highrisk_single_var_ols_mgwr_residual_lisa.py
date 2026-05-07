# -*- coding: utf-8 -*-
"""
High-risk-cluster single-variable OLS vs MGWR residual LISA maps.

Variables:
    구조노후도, 단속위험도, 최근접_소화용수_거리등급, 도로폭위험도

For each variable, this script:
    1. filters facilities to 고위험군,
    2. fits single-variable OLS,
    3. fits single-variable MGWR inside the high-risk cluster,
    4. aggregates residuals by legal dong,
    5. runs LISA on high-risk legal dongs,
    6. exports one OLS/MGWR comparison PNG.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from esda.moran import Moran, Moran_Local
from libpysal.weights import Queen
from mgwr.gwr import MGWR
from mgwr.sel_bw import Sel_BW
from shapely.validation import make_valid
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


BASE = Path(__file__).resolve().parents[1]
TABLE_PATH = next((BASE / "0430").glob("*0429.csv"))
BOUNDARY_PATH = BASE / "data" / "seoul_legal_dong_age_buckets_joined_0415.geojson"
OUT_DIR = BASE / "data" / "highrisk_single_var_residual_lisa"

TARGET = "최종위험점수_new"
VARIABLES = ["구조노후도", "단속위험도", "최근접_소화용수_거리등급", "도로폭위험도"]
COORDS = ["x_5181", "y_5181"]
ID_COLS = [
    "구",
    "동",
    "숙소명",
    "cluster",
    "cluster_label",
    "위도",
    "경도",
    *COORDS,
    TARGET,
]

GU_BY_CODE = {
    "11110": "종로구",
    "11140": "중구",
    "11170": "용산구",
    "11200": "성동구",
    "11215": "광진구",
    "11230": "동대문구",
    "11260": "중랑구",
    "11290": "성북구",
    "11305": "강북구",
    "11320": "도봉구",
    "11350": "노원구",
    "11380": "은평구",
    "11410": "서대문구",
    "11440": "마포구",
    "11470": "양천구",
    "11500": "강서구",
    "11530": "구로구",
    "11545": "금천구",
    "11560": "영등포구",
    "11590": "동작구",
    "11620": "관악구",
    "11650": "서초구",
    "11680": "강남구",
    "11710": "송파구",
    "11740": "강동구",
}

LISA_COLORS = {
    "HH": "#d7191c",
    "LL": "#2c7bb6",
    "HL": "#fdae61",
    "LH": "#abd9e9",
    "Not Sig": "#d9dee7",
    "No Data": "#f2f4f7",
}
LISA_LABELS = {
    "HH": "HH: 양의 잔차 집중",
    "LL": "LL: 음의 잔차 집중",
    "HL": "HL: 국지적 양의 잔차",
    "LH": "LH: 국지적 음의 잔차",
    "Not Sig": "유의하지 않음",
    "No Data": "고위험군 자료 없음",
}


def load_frame() -> pd.DataFrame:
    df = pd.read_csv(TABLE_PATH, encoding="utf-8-sig")
    required = [*ID_COLS, *VARIABLES]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    work = df[required].copy()
    for col in [*COORDS, TARGET, *VARIABLES]:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna(subset=[*COORDS, TARGET, *VARIABLES]).reset_index(drop=True)

    high = work[work["cluster_label"].eq("고위험군")].copy()
    if high.empty:
        high = work[work["cluster"].eq(work["cluster"].max())].copy()
    if high.empty:
        raise ValueError("No high-risk rows found.")
    return high.reset_index(drop=True)


def load_boundary() -> gpd.GeoDataFrame:
    gdf = gpd.read_file(BOUNDARY_PATH)
    gdf["구_매칭"] = gdf["EMD_CD"].astype(str).str[:5].map(GU_BY_CODE)
    if "구" in gdf.columns:
        gdf["구_매칭"] = gdf["구"].fillna(gdf["구_매칭"])
    gdf["동_매칭"] = gdf.get("법정동명", gdf["EMD_KOR_NM"]).fillna(gdf["EMD_KOR_NM"])
    gdf = gdf.to_crs(epsg=5179)
    gdf["geometry"] = gdf.geometry.apply(make_valid).buffer(0)
    return gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()


def run_ols(df: pd.DataFrame, variable: str) -> tuple[pd.DataFrame, dict]:
    x = StandardScaler().fit_transform(df[[variable]].astype(float).to_numpy())
    y = df[TARGET].astype(float).to_numpy()
    model = LinearRegression().fit(x, y)
    pred = model.predict(x)
    out = df[ID_COLS].copy()
    out["prediction"] = pred
    out["residual"] = y - pred
    out[f"coef_{variable}"] = float(model.coef_[0])
    return out, {
        "model": "OLS",
        "variable": variable,
        "rows": int(len(out)),
        "R2": float(model.score(x, y)),
    }


def as_bandwidth_list(raw_bw, expected_len: int) -> list[float]:
    if isinstance(raw_bw, tuple):
        raw_bw = raw_bw[0]
    arr = np.asarray(raw_bw, dtype=float).reshape(-1)
    if len(arr) != expected_len:
        return [float("nan")] * expected_len
    return arr.tolist()


def run_mgwr(
    df: pd.DataFrame, variable: str, n_jobs: int = 1
) -> tuple[pd.DataFrame, dict]:
    coords = df[COORDS].astype(float).to_numpy()
    y = df[TARGET].astype(float).to_numpy().reshape((-1, 1))
    x = StandardScaler().fit_transform(df[[variable]].astype(float).to_numpy())

    print(f"[MGWR-HIGH] {variable}, rows={len(df):,}")
    t0 = time.time()
    selector = Sel_BW(
        coords, y, x, multi=True, kernel="bisquare", fixed=False, n_jobs=n_jobs
    )
    selector.search(verbose=True)
    bandwidths = as_bandwidth_list(selector.bw, 2)
    print(f"[MGWR-HIGH] {variable} BW={bandwidths} search={time.time() - t0:.1f}s")

    t0 = time.time()
    result = MGWR(
        coords, y, x, selector, kernel="bisquare", fixed=False, n_jobs=n_jobs
    ).fit()
    print(f"[MGWR-HIGH] {variable} fit={time.time() - t0:.1f}s R2={result.R2:.4f}")

    out = df[ID_COLS].copy()
    out["residual"] = np.asarray(result.resid_response).reshape(-1)
    out["coef_intercept"] = result.params[:, 0]
    out[f"coef_{variable}"] = result.params[:, 1]
    out["bw_intercept"] = bandwidths[0] if len(bandwidths) > 0 else np.nan
    out[f"bw_{variable}"] = bandwidths[1] if len(bandwidths) > 1 else np.nan
    return out, {
        "model": "MGWR",
        "variable": variable,
        "rows": int(len(out)),
        "bandwidths": bandwidths,
        "R2": float(result.R2),
        "adj_R2": float(result.adj_R2),
    }


def lisa_category(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    sig = p < 0.05
    cats = np.full(len(q), "Not Sig", dtype=object)
    cats[(q == 1) & sig] = "HH"
    cats[(q == 2) & sig] = "LH"
    cats[(q == 3) & sig] = "LL"
    cats[(q == 4) & sig] = "HL"
    return cats


def residual_lisa(
    boundary: gpd.GeoDataFrame, result: pd.DataFrame, model_name: str, variable: str
):
    dong = (
        result.groupby(["구", "동"], dropna=False)
        .agg(residual=("residual", "mean"), sample_count=("residual", "size"))
        .reset_index()
    )
    base = boundary.merge(
        dong, left_on=["구_매칭", "동_매칭"], right_on=["구", "동"], how="left"
    )
    data = base[base["residual"].notna()].copy().reset_index(drop=True)
    if len(data) < 5:
        raise ValueError(f"Too few high-risk dongs for LISA: {len(data)}")

    weights = Queen.from_dataframe(data, use_index=False)
    weights.transform = "r"
    y = data["residual"].to_numpy(dtype=float)
    moran = Moran(y, weights, permutations=999)
    local = Moran_Local(y, weights, permutations=999, seed=42)
    data["lisa_cat"] = lisa_category(local.p_sim, local.q)
    data["local_i"] = local.Is
    data["p_sim"] = local.p_sim
    data["quadrant"] = local.q

    out = base.merge(
        data[["구_매칭", "동_매칭", "lisa_cat", "local_i", "p_sim", "quadrant"]],
        on=["구_매칭", "동_매칭"],
        how="left",
    )
    out["lisa_cat"] = out["lisa_cat"].fillna("No Data")
    metrics = {
        "model": model_name,
        "variable": variable,
        "dong_count": int(len(data)),
        "global_moran_i": float(moran.I),
        "global_moran_p": float(moran.p_sim),
        "HH": int((data["lisa_cat"] == "HH").sum()),
        "LL": int((data["lisa_cat"] == "LL").sum()),
        "HL": int((data["lisa_cat"] == "HL").sum()),
        "LH": int((data["lisa_cat"] == "LH").sum()),
        "Not Sig": int((data["lisa_cat"] == "Not Sig").sum()),
    }
    return out, metrics


def plot_lisa(
    variable: str, model_maps: dict[str, gpd.GeoDataFrame], metrics: list[dict]
) -> Path:
    matplotlib.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(20, 9.8), dpi=180)
    fig.patch.set_facecolor("#f7f9fc")
    metric_by_model = {m["model"]: m for m in metrics}
    gu_boundary = next(iter(model_maps.values())).dissolve(
        by="구_매칭", as_index=False, method="unary", grid_size=0.05
    )

    for ax, (model_name, gdf) in zip(axes, model_maps.items()):
        ax.set_facecolor("#f7f9fc")
        gdf["plot_color"] = (
            gdf["lisa_cat"].map(LISA_COLORS).fillna(LISA_COLORS["No Data"])
        )
        gdf.plot(ax=ax, color=gdf["plot_color"], edgecolor="#c9d1dc", linewidth=0.18)
        gu_boundary.boundary.plot(ax=ax, color="#303744", linewidth=0.75, alpha=0.9)
        for _, row in gu_boundary.iterrows():
            if row.geometry.is_empty:
                continue
            point = row.geometry.representative_point()
            ax.text(
                point.x,
                point.y,
                row["구_매칭"],
                ha="center",
                va="center",
                fontsize=5.8,
                weight="bold",
            )
        m = metric_by_model[model_name]
        ax.set_title(
            f"{model_name} 고위험군 {variable} 잔차 LISA\nMoran's I={m['global_moran_i']:.3f}, p={m['global_moran_p']:.3f}",
            fontsize=15,
            weight="bold",
            loc="left",
        )
        ax.set_axis_off()

    handles = [
        mpatches.Patch(color=LISA_COLORS[key], label=LISA_LABELS[key])
        for key in ["HH", "LL", "HL", "LH", "Not Sig", "No Data"]
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=True,
        framealpha=0.96,
        fontsize=9,
    )
    fig.suptitle(
        f"고위험군 {variable}: OLS 단일변수 잔차 LISA vs MGWR 잔차 LISA",
        fontsize=22,
        weight="bold",
        x=0.03,
        ha="left",
    )
    fig.text(
        0.03,
        0.04,
        f"고위험군 시설만 사용. 사용 변수: {variable} 1개. 색은 법정동 평균 잔차의 LISA 유형.",
        fontsize=9,
        color="#667085",
    )
    out_path = OUT_DIR / f"고위험군_{variable}_ols_mgwr_residual_lisa.png"
    fig.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_frame()
    boundary = load_boundary()
    all_metrics = []

    print(f"[DATA] high-risk rows={len(df):,}, variables={VARIABLES}")
    for variable in VARIABLES:
        print(f"\n[VARIABLE] {variable}")
        ols_out, ols_model_metric = run_ols(df, variable)
        mgwr_out, mgwr_model_metric = run_mgwr(df, variable, n_jobs=1)

        ols_out.to_csv(
            OUT_DIR / f"고위험군_{variable}_ols_results.csv",
            index=False,
            encoding="utf-8-sig",
        )
        mgwr_out.to_csv(
            OUT_DIR / f"고위험군_{variable}_mgwr_results.csv",
            index=False,
            encoding="utf-8-sig",
        )

        ols_map, ols_lisa = residual_lisa(boundary, ols_out, "OLS", variable)
        mgwr_map, mgwr_lisa = residual_lisa(boundary, mgwr_out, "MGWR", variable)
        ols_map.to_csv(
            OUT_DIR / f"고위험군_{variable}_ols_residual_lisa_by_dong.csv",
            index=False,
            encoding="utf-8-sig",
        )
        mgwr_map.to_csv(
            OUT_DIR / f"고위험군_{variable}_mgwr_residual_lisa_by_dong.csv",
            index=False,
            encoding="utf-8-sig",
        )

        path = plot_lisa(
            variable, {"OLS": ols_map, "MGWR": mgwr_map}, [ols_lisa, mgwr_lisa]
        )
        print(f"[SAVED] {path}")
        all_metrics.extend(
            [{**ols_model_metric, **ols_lisa}, {**mgwr_model_metric, **mgwr_lisa}]
        )

    pd.DataFrame(all_metrics).to_csv(
        OUT_DIR / "highrisk_single_var_residual_lisa_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (OUT_DIR / "run_metadata.json").write_text(
        json.dumps(all_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("[DONE]")
    print(pd.DataFrame(all_metrics).to_string(index=False))


if __name__ == "__main__":
    main()
