# -*- coding: utf-8 -*-
"""Create six static PNG visualizations from the fire-dispatch CSV."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CSV = BASE_DIR / "data" / "화재출동" / "화재출동_2021_2024.csv"
DEFAULT_OUT = BASE_DIR / "reports" / "fire_visualizations_png"

WEEKDAY_ORDER = ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"]
FIG_DPI = 180


def setup_style() -> None:
    font_names = {f.name for f in fm.fontManager.ttflist}
    selected_font = "Malgun Gothic"
    for font in ("Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR"):
        if font in font_names:
            selected_font = font
            break
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["font.family"] = selected_font
    plt.rcParams["font.sans-serif"] = [selected_font, "DejaVu Sans", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"


def read_csv_robust(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "cp949", "euc-kr", "utf-8"):
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, low_memory=False)


def clean_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype("string").str.strip()
        df[col] = df[col].replace("", pd.NA)
    return df


def prepare_data(csv_path: Path, start_year: int | None, end_year: int | None) -> pd.DataFrame:
    df = clean_text_columns(read_csv_robust(csv_path))

    numeric_cols = [
        "발생연도",
        "발생월",
        "발생일",
        "발생시",
        "발생분",
        "사망자수",
        "부상자수",
        "인명피해계",
        "재산피해액(천원)",
        "출동소요시간",
        "진압소요시간",
        "경도",
        "위도",
        "현장거리(km)",
        "안전센터_현장거리(km)",
        "출동대_현장거리(km)",
        "기온(℃)",
        "강수량(mm)",
        "풍속(m/s)",
        "습도(%)",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    date_text = df["발생일자"].astype("string").str.replace(r"\.0$", "", regex=True).str.zfill(8)
    df["발생일자_dt"] = pd.to_datetime(date_text, format="%Y%m%d", errors="coerce")
    df["발생연월"] = df["발생일자_dt"].dt.to_period("M").dt.to_timestamp()
    df["출동소요시간_분"] = df["출동소요시간"] / 60
    df["진압소요시간_분"] = df["진압소요시간"] / 60
    df["재산피해액_백만원"] = df["재산피해액(천원)"] / 1000

    if start_year is not None:
        df = df[df["발생연도"] >= start_year]
    if end_year is not None:
        df = df[df["발생연도"] <= end_year]
    return df.reset_index(drop=True)


def save_fig(fig: plt.Figure, out_path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def add_title(ax: plt.Axes, title: str, subtitle: str | None = None) -> None:
    ax.set_title(title, loc="left", fontsize=17, fontweight="bold", pad=34 if subtitle else 16)
    if subtitle:
        ax.text(
            0,
            1.015,
            subtitle,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=10.5,
            color="#4b5563",
        )


def viz_01_monthly_trend(df: pd.DataFrame, out_dir: Path) -> Path:
    monthly = (
        df.dropna(subset=["발생연월"])
        .groupby("발생연월")
        .size()
        .reset_index(name="화재건수")
        .sort_values("발생연월")
    )
    fig, ax = plt.subplots(figsize=(13.5, 6.2))
    ax.plot(monthly["발생연월"], monthly["화재건수"], color="#2563eb", linewidth=2.5)
    ax.scatter(monthly["발생연월"], monthly["화재건수"], color="#dc2626", s=28, zorder=3)
    add_title(ax, "연도-월별 화재 발생 추세", f"총 {len(df):,}건")
    ax.set_xlabel("발생 연월")
    ax.set_ylabel("화재 건수")
    ax.grid(axis="y", color="#e5e7eb")
    ax.grid(axis="x", visible=False)
    return save_fig(fig, out_dir / "01_year_month_fire_trend.png")


def viz_02_weekday_hour_heatmap(df: pd.DataFrame, out_dir: Path) -> Path:
    heat = df.dropna(subset=["발생요일", "발생시"]).copy()
    heat = heat[heat["발생요일"].isin(WEEKDAY_ORDER)]
    heat["발생시"] = heat["발생시"].astype(int)
    pivot = (
        heat.groupby(["발생요일", "발생시"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=WEEKDAY_ORDER, columns=range(24), fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(14, 5.8))
    sns.heatmap(
        pivot,
        ax=ax,
        cmap="YlOrRd",
        linewidths=0.35,
        linecolor="#fff7ed",
        cbar_kws={"label": "화재 건수"},
    )
    add_title(ax, "요일 × 시간대 화재 발생 히트맵")
    ax.set_xlabel("발생 시간")
    ax.set_ylabel("발생 요일")
    ax.set_xticklabels([f"{i}시" for i in range(24)], rotation=0)
    ax.set_yticklabels(WEEKDAY_ORDER, rotation=0)
    return save_fig(fig, out_dir / "02_weekday_hour_fire_heatmap.png")


def valid_geo_rows(df: pd.DataFrame) -> pd.DataFrame:
    geo = df.dropna(subset=["위도", "경도"]).copy()
    return geo[(geo["위도"].between(37.4, 37.8)) & (geo["경도"].between(126.7, 127.3))]


def viz_03_district_map(df: pd.DataFrame, out_dir: Path) -> Path:
    geo = valid_geo_rows(df)
    gu_stats = (
        geo.dropna(subset=["발생시군구"])
        .groupby("발생시군구")
        .agg(
            화재건수=("화재번호", "count"),
            위도=("위도", "median"),
            경도=("경도", "median"),
            평균출동분=("출동소요시간_분", "mean"),
            총재산피해백만원=("재산피해액_백만원", "sum"),
        )
        .reset_index()
        .sort_values("화재건수", ascending=False)
    )
    gu_stats.to_csv(out_dir / "03_district_fire_summary.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(10.8, 10.8))
    ax.scatter(
        geo["경도"],
        geo["위도"],
        s=4,
        color="#9ca3af",
        alpha=0.18,
        linewidths=0,
        label="개별 화재 위치",
    )
    sizes = 180 + 1200 * (gu_stats["화재건수"] / gu_stats["화재건수"].max())
    bubbles = ax.scatter(
        gu_stats["경도"],
        gu_stats["위도"],
        s=sizes,
        c=gu_stats["화재건수"],
        cmap="YlOrRd",
        edgecolor="#7f1d1d",
        linewidth=1.2,
        alpha=0.82,
        label="구별 발생 건수",
    )
    for _, row in gu_stats.iterrows():
        ax.text(
            row["경도"],
            row["위도"],
            f"{row['발생시군구']}\n{int(row['화재건수']):,}",
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="#111827",
        )
    add_title(ax, "구별 화재 발생 지도", "점은 개별 화재 위치, 원 크기와 색은 구별 발생 건수")
    ax.set_xlabel("경도")
    ax.set_ylabel("위도")
    ax.set_xlim(126.75, 127.18)
    ax.set_ylim(37.43, 37.68)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color="#e5e7eb")
    cbar = fig.colorbar(bubbles, ax=ax, shrink=0.74)
    cbar.set_label("화재 건수")
    return save_fig(fig, out_dir / "03_district_fire_map.png")


def viz_04_cause_top10(df: pd.DataFrame, out_dir: Path) -> Path:
    cause = (
        df.dropna(subset=["발화요인_대분류"])
        .groupby("발화요인_대분류")
        .size()
        .reset_index(name="화재건수")
        .sort_values("화재건수", ascending=False)
        .head(10)
        .sort_values("화재건수")
    )
    fig, ax = plt.subplots(figsize=(11.5, 6.8))
    colors = sns.color_palette("crest", len(cause))
    ax.barh(cause["발화요인_대분류"], cause["화재건수"], color=colors)
    for i, value in enumerate(cause["화재건수"]):
        ax.text(value + cause["화재건수"].max() * 0.012, i, f"{int(value):,}건", va="center", fontsize=10)
    add_title(ax, "발화요인 대분류 TOP 10")
    ax.set_xlabel("화재 건수")
    ax.set_ylabel("발화요인")
    ax.set_xlim(0, cause["화재건수"].max() * 1.18)
    ax.grid(axis="x", color="#e5e7eb")
    ax.grid(axis="y", visible=False)
    return save_fig(fig, out_dir / "04_top10_fire_causes.png")


def viz_05_response_vs_distance(df: pd.DataFrame, out_dir: Path) -> Path:
    scatter = df.dropna(subset=["현장거리(km)", "출동소요시간_분"]).copy()
    scatter = scatter[(scatter["현장거리(km)"] >= 0) & (scatter["출동소요시간_분"] > 0)]
    distance_cap = scatter["현장거리(km)"].quantile(0.99)
    response_cap = scatter["출동소요시간_분"].quantile(0.99)
    plot_df = scatter[
        (scatter["현장거리(km)"] <= distance_cap) & (scatter["출동소요시간_분"] <= response_cap)
    ].copy()
    if len(plot_df) > 9000:
        plot_df = plot_df.sample(9000, random_state=42)

    corr = plot_df["현장거리(km)"].corr(plot_df["출동소요시간_분"])
    fig, ax = plt.subplots(figsize=(11.5, 7))
    sns.scatterplot(
        data=plot_df,
        x="현장거리(km)",
        y="출동소요시간_분",
        hue="발생시군구",
        palette="tab20",
        s=20,
        alpha=0.5,
        linewidth=0,
        legend=False,
        ax=ax,
    )
    if len(plot_df) >= 2:
        slope, intercept = np.polyfit(plot_df["현장거리(km)"], plot_df["출동소요시간_분"], 1)
        x_line = np.linspace(plot_df["현장거리(km)"].min(), plot_df["현장거리(km)"].max(), 100)
        ax.plot(x_line, slope * x_line + intercept, color="#111827", linewidth=2.4, label="선형 추세")
    add_title(ax, "출동소요시간 vs 현장거리", f"상위 1% 이상치 제외, 상관계수 r = {corr:.3f}")
    ax.set_xlabel("현장거리(km)")
    ax.set_ylabel("출동소요시간(분)")
    ax.grid(color="#e5e7eb")
    ax.legend(loc="upper left", frameon=True)
    return save_fig(fig, out_dir / "05_response_time_vs_distance.png")


def viz_06_high_damage_map(df: pd.DataFrame, out_dir: Path) -> Path:
    geo = valid_geo_rows(df)
    top = (
        geo.dropna(subset=["재산피해액(천원)"])
        .query("`재산피해액(천원)` > 0")
        .sort_values("재산피해액(천원)", ascending=False)
        .head(100)
        .copy()
    )
    top["재산피해액_백만원"] = top["재산피해액(천원)"] / 1000

    ranking_cols = [
        "화재번호",
        "발생일자",
        "발생시군구",
        "화재유형",
        "발화요인_대분류",
        "발화요인_소분류",
        "발화장소_대분류",
        "발화장소_중분류",
        "재산피해액(천원)",
        "재산피해액_백만원",
        "사망자수",
        "부상자수",
        "인명피해계",
        "위도",
        "경도",
    ]
    top[[c for c in ranking_cols if c in top.columns]].to_csv(
        out_dir / "06_high_damage_ranking_top100.csv", index=False, encoding="utf-8-sig"
    )

    fig = plt.figure(figsize=(15.5, 8.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.08, 1.12], wspace=0.34)
    ax_map = fig.add_subplot(gs[0, 0])
    ax_rank = fig.add_subplot(gs[0, 1])

    ax_map.scatter(geo["경도"], geo["위도"], s=3, color="#d1d5db", alpha=0.16, linewidths=0)
    sizes = 40 + 420 * np.log10(top["재산피해액(천원)"] + 1) / np.log10(top["재산피해액(천원)"].max() + 1)
    ax_map.scatter(
        top["경도"],
        top["위도"],
        s=sizes,
        color="#fca5a5",
        edgecolor="#7f1d1d",
        linewidth=0.8,
        alpha=0.74,
    )
    for rank, (_, row) in enumerate(top.head(10).iterrows(), start=1):
        ax_map.text(row["경도"], row["위도"], str(rank), ha="center", va="center", fontsize=8, fontweight="bold")
    add_title(ax_map, "재산피해액 TOP 100 화재 위치", "숫자 표기는 피해액 상위 10건")
    ax_map.set_xlabel("경도")
    ax_map.set_ylabel("위도")
    ax_map.set_xlim(126.75, 127.18)
    ax_map.set_ylim(37.43, 37.68)
    ax_map.set_aspect("equal", adjustable="box")
    ax_map.grid(color="#e5e7eb")

    rank = top.head(15).copy()
    rank["순위"] = range(1, len(rank) + 1)
    labels = rank.apply(
        lambda r: f"{int(r['순위'])}. {r.get('발생시군구', '')} | {str(r.get('발생일자', ''))}",
        axis=1,
    )
    y_pos = np.arange(len(rank))
    ax_rank.barh(y_pos, rank["재산피해액_백만원"], color="#dc2626", alpha=0.82)
    ax_rank.set_yticks(y_pos)
    ax_rank.set_yticklabels(labels, fontsize=8.6)
    ax_rank.invert_yaxis()
    ax_rank.set_xscale("log")
    ax_rank.set_xlabel("재산피해액(백만원, 로그축)")
    ax_rank.set_title("재산피해액 TOP 15", loc="left", fontsize=15, fontweight="bold", pad=14)
    ax_rank.grid(axis="x", color="#e5e7eb")
    ax_rank.grid(axis="y", visible=False)
    for y, value in zip(y_pos, rank["재산피해액_백만원"]):
        ax_rank.text(value * 1.08, y, f"{value:,.1f}", va="center", fontsize=8.5)
    ax_rank.set_xlim(max(rank["재산피해액_백만원"].min() * 0.72, 1), rank["재산피해액_백만원"].max() * 1.55)

    return save_fig(fig, out_dir / "06_high_damage_fire_map_and_ranking.png")


def write_summary(df: pd.DataFrame, csv_path: Path, out_dir: Path, outputs: list[Path]) -> Path:
    summary = {
        "source_csv": str(csv_path),
        "row_count": int(len(df)),
        "year_min": int(df["발생연도"].min()) if df["발생연도"].notna().any() else None,
        "year_max": int(df["발생연도"].max()) if df["발생연도"].notna().any() else None,
        "fire_count_by_year": {
            str(int(k)): int(v)
            for k, v in df.groupby("발생연도").size().sort_index().items()
            if pd.notna(k)
        },
        "top_districts": df.groupby("발생시군구").size().sort_values(ascending=False).head(10).to_dict(),
        "top_causes": df.groupby("발화요인_대분류").size().sort_values(ascending=False).head(10).to_dict(),
        "outputs": [p.name for p in outputs],
    }
    out_path = out_dir / "summary.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build six static PNG fire-dispatch visualizations.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Input fire dispatch CSV path")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output directory")
    parser.add_argument("--start-year", type=int, default=None, help="Optional inclusive start year")
    parser.add_argument("--end-year", type=int, default=None, help="Optional inclusive end year")
    return parser.parse_args()


def main() -> None:
    setup_style()
    args = parse_args()
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    df = prepare_data(args.csv, args.start_year, args.end_year)
    outputs = [
        viz_01_monthly_trend(df, out_dir),
        viz_02_weekday_hour_heatmap(df, out_dir),
        viz_03_district_map(df, out_dir),
        viz_04_cause_top10(df, out_dir),
        viz_05_response_vs_distance(df, out_dir),
        viz_06_high_damage_map(df, out_dir),
    ]
    summary_path = write_summary(df, args.csv, out_dir, outputs)

    print(f"rows={len(df):,}")
    print(f"years={int(df['발생연도'].min())}-{int(df['발생연도'].max())}")
    print(f"output_dir={out_dir}")
    for path in outputs:
        print(f"created={path}")
    print(f"created={summary_path}")


if __name__ == "__main__":
    main()
