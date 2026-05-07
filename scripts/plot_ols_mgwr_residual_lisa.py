# -*- coding: utf-8 -*-
"""
Create OLS residual LISA and MGWR residual LISA maps.

OLS residuals are refit from 0430/최종테이블0429.csv using the same six
variables used for the full GWR/MGWR run. MGWR residuals are read from
data/full_gwr_mgwr/mgwr_results_full.csv.
"""

from __future__ import annotations

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
from shapely.validation import make_valid
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


BASE = Path(__file__).resolve().parents[1]
TABLE_PATH = next((BASE / "0430").glob("*0429.csv"))
MGWR_PATH = BASE / "data" / "full_gwr_mgwr" / "mgwr_results_full.csv"
BOUNDARY_PATH = BASE / "data" / "seoul_legal_dong_age_buckets_joined_0415.geojson"
OUT_DIR = BASE / "data" / "full_gwr_mgwr" / "residual_lisa"

TARGET = "최종위험점수_new"
FEATURES = [
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
    "HH": "High-High: 양의 잔차 집중",
    "LL": "Low-Low: 음의 잔차 집중",
    "HL": "High-Low: 국지적 양의 잔차",
    "LH": "Low-High: 국지적 음의 잔차",
    "Not Sig": "유의하지 않음",
    "No Data": "자료 없음",
}


def load_boundary() -> gpd.GeoDataFrame:
    gdf = gpd.read_file(BOUNDARY_PATH)
    gdf["구_매칭"] = gdf["EMD_CD"].astype(str).str[:5].map(GU_BY_CODE)
    if "구" in gdf.columns:
        gdf["구_매칭"] = gdf["구"].fillna(gdf["구_매칭"])
    gdf["동_매칭"] = gdf.get("법정동명", gdf["EMD_KOR_NM"]).fillna(gdf["EMD_KOR_NM"])
    gdf = gdf.to_crs(epsg=5179)
    gdf["geometry"] = gdf.geometry.apply(make_valid).buffer(0)
    return gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()


def fit_ols_residuals() -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(TABLE_PATH, encoding="utf-8-sig")
    required = ["구", "동", TARGET, *FEATURES]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {TABLE_PATH}: {missing}")

    work = df[required].copy()
    for col in [TARGET, *FEATURES]:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna(subset=[TARGET, *FEATURES]).reset_index(drop=True)
    x = StandardScaler().fit_transform(work[FEATURES].to_numpy(dtype=float))
    y = work[TARGET].to_numpy(dtype=float)
    model = LinearRegression().fit(x, y)
    pred = model.predict(x)
    work["residual"] = y - pred
    metrics = {
        "model": "OLS",
        "rows": int(len(work)),
        "r2": float(model.score(x, y)),
    }
    return work[["구", "동", "residual"]], metrics


def load_mgwr_residuals() -> tuple[pd.DataFrame, dict]:
    mgwr = pd.read_csv(MGWR_PATH, encoding="utf-8-sig")
    required = ["구", "동", "residual"]
    missing = [c for c in required if c not in mgwr.columns]
    if missing:
        raise ValueError(f"Missing columns in {MGWR_PATH}: {missing}")
    return mgwr[required].copy(), {"model": "MGWR", "rows": int(len(mgwr))}


def aggregate_to_dong(residuals: pd.DataFrame, model_name: str) -> pd.DataFrame:
    out = (
        residuals.groupby(["구", "동"], dropna=False)
        .agg(residual=("residual", "mean"), sample_count=("residual", "size"))
        .reset_index()
    )
    out["model"] = model_name
    return out


def classify_lisa(values: np.ndarray, p: np.ndarray, q: np.ndarray) -> np.ndarray:
    sig = p < 0.05
    cats = np.full(len(values), "Not Sig", dtype=object)
    cats[(q == 1) & sig] = "HH"
    cats[(q == 2) & sig] = "LH"
    cats[(q == 3) & sig] = "LL"
    cats[(q == 4) & sig] = "HL"
    return cats


def run_lisa(
    boundary: gpd.GeoDataFrame, dong_values: pd.DataFrame, model_name: str
) -> tuple[gpd.GeoDataFrame, dict]:
    gdf = boundary.merge(
        dong_values, left_on=["구_매칭", "동_매칭"], right_on=["구", "동"], how="left"
    )
    data = gdf[gdf["residual"].notna()].copy().reset_index(drop=True)
    if len(data) < 5:
        raise ValueError(
            f"Too few legal dongs with residuals for {model_name}: {len(data)}"
        )

    weights = Queen.from_dataframe(data, use_index=False)
    weights.transform = "r"
    y = data["residual"].to_numpy(dtype=float)
    moran = Moran(y, weights, permutations=999)
    local = Moran_Local(y, weights, permutations=999, seed=42)
    data["lisa_cat"] = classify_lisa(y, local.p_sim, local.q)
    data["local_i"] = local.Is
    data["p_sim"] = local.p_sim
    data["quadrant"] = local.q

    result = gdf.merge(
        data[
            [
                "구_매칭",
                "동_매칭",
                "lisa_cat",
                "local_i",
                "p_sim",
                "quadrant",
                "residual",
                "sample_count",
            ]
        ],
        on=["구_매칭", "동_매칭"],
        how="left",
        suffixes=("", "_lisa"),
    )
    result["lisa_cat"] = result["lisa_cat"].fillna("No Data")
    metrics = {
        "model": model_name,
        "dong_count": int(len(data)),
        "global_moran_i": float(moran.I),
        "global_moran_p": float(moran.p_sim),
        "hh_count": int((data["lisa_cat"] == "HH").sum()),
        "ll_count": int((data["lisa_cat"] == "LL").sum()),
        "hl_count": int((data["lisa_cat"] == "HL").sum()),
        "lh_count": int((data["lisa_cat"] == "LH").sum()),
        "not_sig_count": int((data["lisa_cat"] == "Not Sig").sum()),
    }
    return result, metrics


def plot_maps(model_maps: dict[str, gpd.GeoDataFrame], metrics: list[dict]) -> Path:
    matplotlib.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 2, figsize=(20, 9.8), dpi=180)
    fig.patch.set_facecolor("#f7f9fc")

    gu_boundary = next(iter(model_maps.values())).dissolve(
        by="구_매칭", as_index=False, method="unary", grid_size=0.05
    )
    metric_by_model = {m["model"]: m for m in metrics}

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
            f"{model_name} 잔차 LISA\nMoran's I={m['global_moran_i']:.3f}, p={m['global_moran_p']:.3f}",
            fontsize=16,
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
        "OLS 잔차 LISA vs MGWR 잔차 LISA", fontsize=23, weight="bold", x=0.03, ha="left"
    )
    fig.text(
        0.03,
        0.04,
        "법정동 평균 잔차 기준. HH는 모델이 낮게 예측한 양의 잔차가 주변과 함께 몰린 곳, LL은 높게 예측한 음의 잔차가 주변과 함께 몰린 곳.",
        fontsize=9,
        color="#667085",
    )
    out_path = OUT_DIR / "ols_mgwr_residual_lisa_maps.png"
    fig.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    boundary = load_boundary()

    ols_resid, ols_metric = fit_ols_residuals()
    mgwr_resid, mgwr_metric = load_mgwr_residuals()
    model_inputs = {
        "OLS": aggregate_to_dong(ols_resid, "OLS"),
        "MGWR": aggregate_to_dong(mgwr_resid, "MGWR"),
    }

    model_maps = {}
    metrics = []
    for model_name, dong_values in model_inputs.items():
        gdf, lisa_metric = run_lisa(boundary, dong_values, model_name)
        model_maps[model_name] = gdf
        base_metric = ols_metric if model_name == "OLS" else mgwr_metric
        metrics.append({**base_metric, **lisa_metric})
        export_cols = [
            "구_매칭",
            "동_매칭",
            "residual",
            "sample_count",
            "lisa_cat",
            "local_i",
            "p_sim",
            "quadrant",
        ]
        gdf[export_cols].to_csv(
            OUT_DIR / f"{model_name.lower()}_residual_lisa_by_dong.csv",
            index=False,
            encoding="utf-8-sig",
        )

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(
        OUT_DIR / "residual_lisa_summary.csv", index=False, encoding="utf-8-sig"
    )
    out_path = plot_maps(model_maps, metrics)
    print(f"Saved: {out_path}")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
