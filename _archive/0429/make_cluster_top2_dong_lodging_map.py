# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from shapely.geometry import Point


BASE = Path(__file__).resolve().parents[2]
OUT_DIR = BASE / "NJT-PJT" / "0429"
DATA_DIR = BASE / "NJT-PJT" / "0424" / "data" / "cluster3_spatial_pipeline_fire_count_150m_0428"
BOUNDARY_PATH = BASE / "NJT-PJT" / "data" / "법정동별_사용승인구간_공간정보0415.geojson"

TARGETS = {
    0: [("마포구", "동교동"), ("강서구", "화곡동")],
    1: [("용산구", "이태원동"), ("성동구", "성동")],
    2: [("마포구", "연남동"), ("마포구", "서교동")],
}

COLORS = {
    0: "#2563EB",
    1: "#059669",
    2: "#F97316",
}


def setup_font() -> None:
    font_path = Path("C:/Windows/Fonts/malgun.ttf")
    if font_path.exists():
        fm.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = fm.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams["axes.unicode_minus"] = False


def read_source() -> pd.DataFrame:
    data_path = max([p for p in DATA_DIR.glob("*.csv") if p.stat().st_size > 100000], key=lambda p: p.stat().st_size)
    df = pd.read_csv(data_path, encoding="utf-8-sig", low_memory=False)
    df.columns = df.columns.str.strip()
    for col in ["위도", "경도", "cluster", "최종_화재위험점수"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["구", "동", "위도", "경도", "cluster"]).copy()


def select_target_facilities(df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for cluster, areas in TARGETS.items():
        for gu, dong in areas:
            part = df[
                df["cluster"].astype(int).eq(cluster)
                & df["구"].astype(str).str.strip().eq(gu)
                & df["동"].astype(str).str.strip().eq(dong)
            ].copy()
            part["target_label"] = f"C{cluster} {gu} {dong}"
            part["target_gu"] = gu
            part["target_dong"] = dong
            parts.append(part)
    return pd.concat(parts, ignore_index=True)


def load_boundaries() -> gpd.GeoDataFrame:
    gdf = gpd.read_file(BOUNDARY_PATH)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    else:
        gdf = gdf.to_crs("EPSG:4326")
    return gdf


def main() -> None:
    setup_font()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = read_source()
    selected = select_target_facilities(df)
    points = gpd.GeoDataFrame(
        selected,
        geometry=[Point(xy) for xy in zip(selected["경도"], selected["위도"])],
        crs="EPSG:4326",
    )
    boundary = load_boundaries()
    target_area_keys = {(gu, dong) for areas in TARGETS.values() for gu, dong in areas}
    target_boundary = boundary[
        boundary.apply(
            lambda row: (str(row.get("구", "")).strip(), str(row.get("법정동명", row.get("EMD_KOR_NM", ""))).strip())
            in target_area_keys,
            axis=1,
        )
    ].copy()

    summary = (
        selected.groupby(["cluster", "target_gu", "target_dong"], as_index=False)
        .agg(
            시설수=("숙소명", "size"),
            평균_최종위험점수=("최종_화재위험점수", "mean"),
            최대_최종위험점수=("최종_화재위험점수", "max"),
        )
        .sort_values(["cluster", "시설수"], ascending=[True, False])
    )
    summary_path = OUT_DIR / "cluster_top2_dongs_lodging_facility_map_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    minx, miny, maxx, maxy = points.total_bounds
    pad_x = max((maxx - minx) * 0.18, 0.012)
    pad_y = max((maxy - miny) * 0.18, 0.012)

    fig = plt.figure(figsize=(16, 10), dpi=180)
    fig.patch.set_facecolor("#F8FAFC")
    ax = fig.add_axes([0.04, 0.10, 0.67, 0.78])
    ax.set_facecolor("#F8FAFC")

    boundary.plot(ax=ax, facecolor="#EEF2F7", edgecolor="white", linewidth=0.45, alpha=0.95)
    if not target_boundary.empty:
        target_boundary.plot(ax=ax, facecolor="#FEF3C7", edgecolor="#92400E", linewidth=1.1, alpha=0.42)

    for cluster, color in COLORS.items():
        sub = points[points["cluster"].astype(int).eq(cluster)]
        ax.scatter(
            sub.geometry.x,
            sub.geometry.y,
            s=34,
            color=color,
            edgecolor="white",
            linewidth=0.45,
            alpha=0.86,
            label=f"Cluster {cluster}",
            zorder=5,
        )

    # Mark each selected dong group at its median facility location without callout labels.
    for (cluster, gu, dong), sub in points.groupby(["cluster", "target_gu", "target_dong"]):
        x = sub.geometry.x.median()
        y = sub.geometry.y.median()
        ax.scatter([x], [y], s=120, marker="X", color=COLORS[int(cluster)], edgecolor="white", linewidth=1.2, zorder=7)

    ax.set_xlim(minx - pad_x, maxx + pad_x)
    ax.set_ylim(miny - pad_y, maxy + pad_y)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color="#CBD5E1", linewidth=0.6, alpha=0.6)
    ax.set_xlabel("경도", color="#475569", fontsize=10)
    ax.set_ylabel("위도", color="#475569", fontsize=10)
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.text(0.04, 0.94, "클러스터별 상위 2개 동 숙박시설 위치", fontsize=22, weight="bold", color="#111827")
    fig.text(
        0.04,
        0.905,
        "기준: cluster3_fire_count_150m_spatial_distribution 표의 각 클러스터 상위 2개 동 · 점 = 해당 클러스터에 속한 숙박시설",
        fontsize=11,
        color="#475569",
    )

    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS[c], markeredgecolor="white", markersize=9, label=f"Cluster {c}")
        for c in [0, 1, 2]
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=True, framealpha=0.95, facecolor="white", edgecolor="#E2E8F0")

    # Right side summary table
    side = fig.add_axes([0.74, 0.14, 0.23, 0.72])
    side.axis("off")
    side.text(0.0, 0.98, "지도 표시 대상", fontsize=15, weight="bold", color="#111827", va="top")
    y = 0.91
    for cluster in [0, 1, 2]:
        side.text(0.0, y, f"Cluster {cluster}", fontsize=13, weight="bold", color=COLORS[cluster], va="top")
        side.plot([0.0, 0.88], [y - 0.025, y - 0.025], color=COLORS[cluster], linewidth=3, solid_capstyle="round")
        y -= 0.07
        csum = summary[summary["cluster"].astype(int).eq(cluster)]
        for row in csum.itertuples(index=False):
            side.text(0.02, y, f"{row.target_gu} {row.target_dong}", fontsize=10.5, color="#111827", va="center")
            side.text(0.88, y, f"{int(row.시설수):,}개", fontsize=10.5, color="#111827", weight="bold", ha="right", va="center")
            side.text(0.02, y - 0.035, f"평균 위험점수 {row.평균_최종위험점수:.1f}", fontsize=8.7, color="#64748B", va="center")
            y -= 0.095
        y -= 0.025

    fig.text(
        0.04,
        0.045,
        f"총 표시 시설 {len(points):,}개 · 경계 강조는 법정동 경계가 매칭된 경우만 표시 · 요약 CSV: {summary_path.name}",
        fontsize=9.5,
        color="#64748B",
    )

    out_path = OUT_DIR / "cluster_top2_dongs_lodging_facility_map.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(out_path)
    print(summary_path)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
