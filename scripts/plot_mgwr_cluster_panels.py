# -*- coding: utf-8 -*-
"""
Plot cluster-separated MGWR maps from precomputed full MGWR results.

This script does not fit GWR/MGWR again. It reads:
    data/full_gwr_mgwr/mgwr_results_full.csv

and exports one PNG per variable with three panels:
    저위험군 / 중위험군 / 고위험군
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from shapely.validation import make_valid


BASE = Path(__file__).resolve().parents[1]
MGWR_PATH = BASE / "data" / "full_gwr_mgwr" / "mgwr_results_full.csv"
BOUNDARY_PATH = BASE / "data" / "seoul_legal_dong_age_buckets_joined_0415.geojson"
OUT_DIR = BASE / "data" / "full_gwr_mgwr" / "cluster_maps"

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


def load_boundary() -> gpd.GeoDataFrame:
    boundary = gpd.read_file(BOUNDARY_PATH)
    boundary["구_매칭"] = boundary["EMD_CD"].astype(str).str[:5].map(GU_BY_CODE)
    if "구" in boundary.columns:
        boundary["구_매칭"] = boundary["구"].fillna(boundary["구_매칭"])
    boundary["동_매칭"] = boundary.get("법정동명", boundary["EMD_KOR_NM"]).fillna(boundary["EMD_KOR_NM"])
    boundary = boundary.to_crs(epsg=5179)
    boundary["geometry"] = boundary.geometry.apply(make_valid).buffer(0)
    return boundary[boundary.geometry.notna() & ~boundary.geometry.is_empty].copy()


def load_mgwr() -> tuple[pd.DataFrame, list[str]]:
    mgwr = pd.read_csv(MGWR_PATH, encoding="utf-8-sig")
    features = [c.removeprefix("coef_") for c in mgwr.columns if c.startswith("coef_") and c != "coef_intercept"]
    if not features:
        raise ValueError("No coef_* variable columns found in MGWR result.")
    return mgwr, features


def build_dong_cluster_values(mgwr: pd.DataFrame, feature: str) -> pd.DataFrame:
    raw_col = f"strength_{feature}"
    contrib_col = f"contrib_{feature}"
    coef_col = f"coef_{feature}"
    bw_col = f"bw_{feature}"

    work = mgwr.copy()
    if contrib_col in work.columns:
        work[raw_col] = work[contrib_col].abs()
        metric_name = "기여도 절댓값"
    else:
        work[raw_col] = work[coef_col].abs()
        metric_name = "계수 절댓값"

    agg = (
        work.groupby(["구", "동", "cluster", "cluster_label"], dropna=False)
        .agg(
            raw_strength=(raw_col, "mean"),
            bandwidth=(bw_col, "mean"),
            sample_count=(raw_col, "size"),
        )
        .reset_index()
    )
    agg["strength"] = agg.groupby("cluster", group_keys=False)["raw_strength"].apply(rank_0_100)
    agg["metric_name"] = metric_name
    return agg


def plot_feature(boundary: gpd.GeoDataFrame, agg: pd.DataFrame, feature: str) -> Path:
    matplotlib.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    clusters = (
        agg[["cluster", "cluster_label"]]
        .drop_duplicates()
        .sort_values("cluster")
        .to_records(index=False)
        .tolist()
    )
    city = boundary.dissolve(method="unary", grid_size=0.05)
    gu_boundary = boundary.dissolve(by="구_매칭", as_index=False, method="unary", grid_size=0.05)
    cmap = matplotlib.colormaps["viridis"]
    norm = Normalize(vmin=0, vmax=100)

    fig, axes = plt.subplots(1, len(clusters), figsize=(21, 8.6), dpi=180)
    if len(clusters) == 1:
        axes = [axes]
    fig.patch.set_facecolor("#f7f9fc")

    summary_rows = []
    for ax, (cluster_id, label) in zip(axes, clusters):
        part = agg[agg["cluster"] == cluster_id].copy()
        gdf = boundary.merge(part, left_on=["구_매칭", "동_매칭"], right_on=["구", "동"], how="left")
        display = gdf[gdf["strength"].notna()].copy()
        bw = part["bandwidth"].dropna().mean()
        dong_count = int(display[["구_매칭", "동_매칭"]].drop_duplicates().shape[0])
        sample_count = int(part["sample_count"].sum())
        summary_rows.append(
            {
                "feature": feature,
                "cluster": int(cluster_id),
                "cluster_label": label,
                "bandwidth": bw,
                "dong_count": dong_count,
                "sample_count": sample_count,
            }
        )

        ax.set_facecolor("#f7f9fc")
        city.plot(ax=ax, color="#eef2f7", edgecolor="#d0d7e2", linewidth=0.25)
        display.plot(
            ax=ax,
            column="strength",
            cmap=cmap,
            norm=norm,
            edgecolor="#c9d1dc",
            linewidth=0.18,
            missing_kwds={"color": "#eef2f7"},
        )
        if not display.empty:
            display.dissolve(method="unary", grid_size=0.05).boundary.plot(
                ax=ax,
                color="#111827",
                linewidth=2.2,
                alpha=0.95,
                zorder=5,
            )
        gu_boundary.boundary.plot(ax=ax, color="#303744", linewidth=0.75, alpha=0.85)
        for _, row in gu_boundary.iterrows():
            if row.geometry.is_empty:
                continue
            point = row.geometry.representative_point()
            ax.text(point.x, point.y, row["구_매칭"], ha="center", va="center", fontsize=5.6, weight="bold")
        ax.set_axis_off()
        ax.set_title(
            f"{label} MGWR\nBW {bw:.0f} · 시설 {sample_count:,}개",
            fontsize=15,
            weight="bold",
            loc="left",
        )

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.012)
    cbar.set_label("군집 내부 상대강도 0-100", fontsize=9)

    fig.suptitle(f"{feature}: MGWR 군집별 분리 표시", fontsize=22, weight="bold", x=0.03, ha="left")
    fig.text(
        0.03,
        0.035,
        "각 패널은 해당 위험군 안에서만 상대강도를 계산함. 회색 지역은 그 패널의 MGWR 추정 대상 군집이 아님.",
        fontsize=9,
        color="#667085",
    )
    out_path = OUT_DIR / f"{feature}_mgwr_cluster_panels.png"
    fig.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path, summary_rows


def plot_cluster_zone_map(boundary: gpd.GeoDataFrame, mgwr: pd.DataFrame) -> Path:
    cluster_colors = {
        "저위험군": "#2f80ed",
        "중위험군": "#f2b84b",
        "고위험군": "#e84a4a",
    }
    cluster_order = ["저위험군", "중위험군", "고위험군"]

    counts = (
        mgwr.groupby(["구", "동", "cluster", "cluster_label"], dropna=False)
        .size()
        .rename("facility_count")
        .reset_index()
    )
    dominant = (
        counts.sort_values(["구", "동", "facility_count"], ascending=[True, True, False])
        .drop_duplicates(["구", "동"])
        .copy()
    )

    gdf = boundary.merge(dominant, left_on=["구_매칭", "동_매칭"], right_on=["구", "동"], how="left")
    gdf["cluster_color"] = gdf["cluster_label"].map(cluster_colors).fillna("#eef2f7")
    zone = gdf[gdf["cluster_label"].notna()].copy()
    dissolved = zone.dissolve(by="cluster_label", as_index=False, method="unary", grid_size=0.05)
    gu_boundary = boundary.dissolve(by="구_매칭", as_index=False, method="unary", grid_size=0.05)

    matplotlib.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(13, 10), dpi=180)
    fig.patch.set_facecolor("#f7f9fc")
    ax.set_facecolor("#f7f9fc")

    gdf.plot(ax=ax, color=gdf["cluster_color"], edgecolor="#c9d1dc", linewidth=0.18)
    for _, row in dissolved.iterrows():
        label = row["cluster_label"]
        gpd.GeoDataFrame([row], geometry="geometry", crs=gdf.crs).boundary.plot(
            ax=ax,
            color=cluster_colors.get(label, "#111827"),
            linewidth=3.0,
            alpha=0.98,
            zorder=5,
        )
    gu_boundary.boundary.plot(ax=ax, color="#303744", linewidth=0.8, alpha=0.9)
    for _, row in gu_boundary.iterrows():
        if row.geometry.is_empty:
            continue
        point = row.geometry.representative_point()
        ax.text(point.x, point.y, row["구_매칭"], ha="center", va="center", fontsize=6, weight="bold")

    handles = [
        plt.Line2D([0], [0], marker="s", color="none", markerfacecolor=cluster_colors[label], markersize=11, label=label)
        for label in cluster_order
    ]
    ax.legend(handles=handles, title="대표 군집", loc="lower left", frameon=True, framealpha=0.95)
    ax.set_axis_off()
    ax.set_title("서울시 법정동별 대표 위험군 3개 구역", fontsize=22, weight="bold", loc="left", pad=12)
    fig.text(
        0.04,
        0.035,
        "법정동 안 시설의 최빈 위험군으로 대표 군집을 지정함. 굵은 선은 저위험군/중위험군/고위험군 권역 경계.",
        fontsize=9,
        color="#667085",
    )

    out_path = OUT_DIR / "mgwr_dominant_cluster_3zones.png"
    fig.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    dominant.to_csv(OUT_DIR / "mgwr_dominant_cluster_by_dong.csv", index=False, encoding="utf-8-sig")
    return out_path


def dominant_cluster_by_dong(mgwr: pd.DataFrame) -> pd.DataFrame:
    counts = (
        mgwr.groupby(["구", "동", "cluster", "cluster_label"], dropna=False)
        .size()
        .rename("facility_count")
        .reset_index()
    )
    return (
        counts.sort_values(["구", "동", "facility_count"], ascending=[True, True, False])
        .drop_duplicates(["구", "동"])
        .copy()
    )


def plot_mgwr_maps_with_cluster_boundaries(boundary: gpd.GeoDataFrame, mgwr: pd.DataFrame, features: list[str]) -> list[Path]:
    cluster_colors = {
        "저위험군": "#2f80ed",
        "중위험군": "#f2b84b",
        "고위험군": "#e84a4a",
    }
    matplotlib.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    dominant = dominant_cluster_by_dong(mgwr)
    base = boundary.merge(dominant, left_on=["구_매칭", "동_매칭"], right_on=["구", "동"], how="left")
    cluster_zone = base[base["cluster_label"].notna()].dissolve(
        by="cluster_label", as_index=False, method="unary", grid_size=0.05
    )
    gu_boundary = boundary.dissolve(by="구_매칭", as_index=False, method="unary", grid_size=0.05)
    cmap = matplotlib.colormaps["viridis"]
    norm = Normalize(vmin=0, vmax=100)
    paths = []

    for feature in features:
        raw_col = f"strength_{feature}"
        contrib_col = f"contrib_{feature}"
        coef_col = f"coef_{feature}"
        work = mgwr.copy()
        if contrib_col in work.columns:
            work[raw_col] = work[contrib_col].abs()
            metric_label = "MGWR 기여도 절댓값"
        else:
            work[raw_col] = work[coef_col].abs()
            metric_label = "MGWR 계수 절댓값"

        agg = (
            work.groupby(["구", "동"], dropna=False)
            .agg(raw_strength=(raw_col, "mean"), sample_count=(raw_col, "size"))
            .reset_index()
        )
        agg["strength"] = rank_0_100(agg["raw_strength"])
        gdf = boundary.merge(agg, left_on=["구_매칭", "동_매칭"], right_on=["구", "동"], how="left")

        fig, ax = plt.subplots(figsize=(13.6, 10), dpi=180)
        fig.patch.set_facecolor("#f7f9fc")
        ax.set_facecolor("#f7f9fc")
        gdf.plot(
            ax=ax,
            column="strength",
            cmap=cmap,
            norm=norm,
            edgecolor="#c9d1dc",
            linewidth=0.16,
            missing_kwds={"color": "#eef2f7"},
        )
        gu_boundary.boundary.plot(ax=ax, color="#303744", linewidth=0.65, alpha=0.85)

        for _, row in cluster_zone.iterrows():
            label = row["cluster_label"]
            one = gpd.GeoDataFrame([row], geometry="geometry", crs=base.crs)
            one.boundary.plot(
                ax=ax,
                color=cluster_colors.get(label, "#111827"),
                linewidth=3.2,
                alpha=0.98,
                zorder=5,
            )
            point = row.geometry.representative_point()
            ax.text(
                point.x,
                point.y,
                label,
                ha="center",
                va="center",
                fontsize=12,
                weight="bold",
                color="#111827",
                bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": cluster_colors.get(label, "#111827"), "linewidth": 1.5, "alpha": 0.9},
                zorder=6,
            )

        for _, row in gu_boundary.iterrows():
            if row.geometry.is_empty:
                continue
            point = row.geometry.representative_point()
            ax.text(point.x, point.y, row["구_매칭"], ha="center", va="center", fontsize=5.4, weight="bold", alpha=0.85)

        handles = [
            plt.Line2D([0], [0], color=color, linewidth=3.2, label=label)
            for label, color in cluster_colors.items()
        ]
        ax.legend(handles=handles, title="군집 경계", loc="lower left", frameon=True, framealpha=0.95)
        ax.set_axis_off()
        ax.set_title(f"{feature}: MGWR 영향력 + 3군집 경계", fontsize=22, weight="bold", loc="left", pad=12)
        fig.text(0.04, 0.045, f"색: {metric_label}의 법정동 평균 상대강도 0-100 / 선: 법정동 대표 위험군 3개 경계", fontsize=9, color="#667085")

        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.015)
        cbar.set_label("변수 영향력 상대강도 0-100", fontsize=9)

        out_path = OUT_DIR / f"{feature}_mgwr_with_3cluster_boundaries.png"
        fig.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        paths.append(out_path)
    return paths


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    boundary = load_boundary()
    mgwr, features = load_mgwr()

    cluster_map = plot_cluster_zone_map(boundary, mgwr)
    boundary_paths = plot_mgwr_maps_with_cluster_boundaries(boundary, mgwr, features)
    all_summary = []
    paths = []
    for feature in features:
        agg = build_dong_cluster_values(mgwr, feature)
        path, rows = plot_feature(boundary, agg, feature)
        paths.append(path)
        all_summary.extend(rows)

    summary = pd.DataFrame(all_summary)
    summary.to_csv(OUT_DIR / "mgwr_cluster_panel_summary.csv", index=False, encoding="utf-8-sig")
    print(f"Saved cluster zone map: {cluster_map}")
    print(f"Saved {len(boundary_paths)} MGWR maps with 3-cluster boundaries")
    for path in boundary_paths:
        print(path)
    print(f"Saved {len(paths)} cluster-panel maps to {OUT_DIR}")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
