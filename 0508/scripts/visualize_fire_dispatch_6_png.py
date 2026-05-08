# -*- coding: utf-8 -*-
"""
화재출동 CSV 로부터 정적(PNG) 6종 시각화 일괄 생성 + 요약 JSON 저장.

생성 파일:
    01_year_month_fire_trend.png     — 월별 화재 발생 추세 라인
    02_weekday_hour_fire_heatmap.png — 요일×시간대 히트맵
    03_district_fire_map.png         — 구별 발생 위치 지도(점+버블)
    04_top10_fire_causes.png         — 발화요인 TOP10 가로 막대
    05_response_time_vs_distance.png — 출동시간 vs 현장거리 산점도+추세선
    06_high_damage_fire_map_and_ranking.png — 재산피해 TOP100 지도 + TOP15 막대

부가 산출물:
    03_district_fire_summary.csv
    06_high_damage_ranking_top100.csv
    summary.json (rows/연도범위/구별 분포/원인/출력파일 목록)

사용:
    python scripts/visualize_fire_dispatch_6_png.py [--csv PATH] [--out DIR] [--start-year N] [--end-year N]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

# 헤드리스 PNG 저장
matplotlib.use("Agg")

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# 기본 입출력 경로
BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CSV = BASE_DIR / "data" / "화재출동" / "화재출동_2021_2024.csv"
DEFAULT_OUT = BASE_DIR / "reports" / "fire_visualizations_png"

# 요일 정렬 순서
WEEKDAY_ORDER = ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"]
FIG_DPI = 180


def setup_style() -> None:
    """seaborn 테마 + 한글 폰트(가용한 것 우선)를 설정."""
    font_names = {f.name for f in fm.fontManager.ttflist}
    selected_font = "Malgun Gothic"
    # OS별 우선순위 — Windows 맑은고딕 → macOS AppleGothic → 나눔/Noto
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
    """다중 인코딩 시도 — utf-8-sig → cp949 → euc-kr → utf-8 순."""
    for encoding in ("utf-8-sig", "cp949", "euc-kr", "utf-8"):
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, low_memory=False)


def clean_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """컬럼명/문자열 컬럼의 공백 정리 + 빈 문자열을 NaN 으로 통일."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype("string").str.strip()
        df[col] = df[col].replace("", pd.NA)
    return df


def prepare_data(
    csv_path: Path, start_year: int | None, end_year: int | None
) -> pd.DataFrame:
    """원본 CSV 로드 + 숫자형 변환 + 날짜/파생 컬럼 생성 + 연도 필터."""
    df = clean_text_columns(read_csv_robust(csv_path))

    # 자주 쓰는 숫자형 컬럼 — 한 번에 강제 변환
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

    # 발생일자 — '20240101.0' 같은 표기 정리 후 datetime 변환
    date_text = (
        df["발생일자"]
        .astype("string")
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(8)
    )
    df["발생일자_dt"] = pd.to_datetime(date_text, format="%Y%m%d", errors="coerce")
    # 월 단위 첫날 — 시계열 그루핑용
    df["발생연월"] = df["발생일자_dt"].dt.to_period("M").dt.to_timestamp()
    # 초 → 분 변환
    df["출동소요시간_분"] = df["출동소요시간"] / 60
    df["진압소요시간_분"] = df["진압소요시간"] / 60
    # 천원 → 백만원 단위 (가독성)
    df["재산피해액_백만원"] = df["재산피해액(천원)"] / 1000

    # 연도 범위 필터 (옵션)
    if start_year is not None:
        df = df[df["발생연도"] >= start_year]
    if end_year is not None:
        df = df[df["발생연도"] <= end_year]
    return df.reset_index(drop=True)


def save_fig(fig: plt.Figure, out_path: Path) -> Path:
    """tight_layout + PNG 저장 + figure 닫기."""
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def add_title(ax: plt.Axes, title: str, subtitle: str | None = None) -> None:
    """좌측 정렬 제목 + 옵션 부제 — 부제 있으면 패딩을 더 둠."""
    ax.set_title(
        title, loc="left", fontsize=17, fontweight="bold", pad=34 if subtitle else 16
    )
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
    """01: 연도-월별 화재 발생 추세 라인 + 점."""
    monthly = (
        df.dropna(subset=["발생연월"])
        .groupby("발생연월")
        .size()
        .reset_index(name="화재건수")
        .sort_values("발생연월")
    )
    fig, ax = plt.subplots(figsize=(13.5, 6.2))
    ax.plot(monthly["발생연월"], monthly["화재건수"], color="#2563eb", linewidth=2.5)
    # 각 월에 점 강조 — 시계열 정점 인식 용이
    ax.scatter(
        monthly["발생연월"], monthly["화재건수"], color="#dc2626", s=28, zorder=3
    )
    add_title(ax, "연도-월별 화재 발생 추세", f"총 {len(df):,}건")
    ax.set_xlabel("발생 연월")
    ax.set_ylabel("화재 건수")
    ax.grid(axis="y", color="#e5e7eb")
    ax.grid(axis="x", visible=False)
    return save_fig(fig, out_dir / "01_year_month_fire_trend.png")


def viz_02_weekday_hour_heatmap(df: pd.DataFrame, out_dir: Path) -> Path:
    """02: 요일×시간대 히트맵 — YlOrRd 컬러맵."""
    heat = df.dropna(subset=["발생요일", "발생시"]).copy()
    heat = heat[heat["발생요일"].isin(WEEKDAY_ORDER)]
    heat["발생시"] = heat["발생시"].astype(int)
    # 요일×시(0~23) 피벗 — 결측 0
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
    """서울 범위 내 위경도만 — bounding box 필터."""
    geo = df.dropna(subset=["위도", "경도"]).copy()
    return geo[(geo["위도"].between(37.4, 37.8)) & (geo["경도"].between(126.7, 127.3))]


def viz_03_district_map(df: pd.DataFrame, out_dir: Path) -> Path:
    """03: 구별 발생 지도 — 회색 점은 개별, 컬러 버블은 구별 합계."""
    geo = valid_geo_rows(df)
    # 구별 발생 위치 중간값 + 화재수/평균출동시간/총피해 계산
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
    # 구별 요약 CSV 저장
    gu_stats.to_csv(
        out_dir / "03_district_fire_summary.csv", index=False, encoding="utf-8-sig"
    )

    fig, ax = plt.subplots(figsize=(10.8, 10.8))
    # 개별 화재 — 회색 점 (배경 레이어)
    ax.scatter(
        geo["경도"],
        geo["위도"],
        s=4,
        color="#9ca3af",
        alpha=0.18,
        linewidths=0,
        label="개별 화재 위치",
    )
    # 구별 버블 크기 — 화재수에 비례
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
    # 구별 라벨 — 이름 + 건수
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
    add_title(
        ax, "구별 화재 발생 지도", "점은 개별 화재 위치, 원 크기와 색은 구별 발생 건수"
    )
    ax.set_xlabel("경도")
    ax.set_ylabel("위도")
    # 서울 범위로 축 고정
    ax.set_xlim(126.75, 127.18)
    ax.set_ylim(37.43, 37.68)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color="#e5e7eb")
    cbar = fig.colorbar(bubbles, ax=ax, shrink=0.74)
    cbar.set_label("화재 건수")
    return save_fig(fig, out_dir / "03_district_fire_map.png")


def viz_04_cause_top10(df: pd.DataFrame, out_dir: Path) -> Path:
    """04: 발화요인 대분류 TOP 10 가로 막대."""
    cause = (
        df.dropna(subset=["발화요인_대분류"])
        .groupby("발화요인_대분류")
        .size()
        .reset_index(name="화재건수")
        .sort_values("화재건수", ascending=False)
        .head(10)
        .sort_values("화재건수")  # 가로 막대용 오름차순
    )
    fig, ax = plt.subplots(figsize=(11.5, 6.8))
    colors = sns.color_palette("crest", len(cause))
    ax.barh(cause["발화요인_대분류"], cause["화재건수"], color=colors)
    # 막대 끝 숫자 라벨
    for i, value in enumerate(cause["화재건수"]):
        ax.text(
            value + cause["화재건수"].max() * 0.012,
            i,
            f"{int(value):,}건",
            va="center",
            fontsize=10,
        )
    add_title(ax, "발화요인 대분류 TOP 10")
    ax.set_xlabel("화재 건수")
    ax.set_ylabel("발화요인")
    # 라벨 잘림 방지 — 우측 여유 18%
    ax.set_xlim(0, cause["화재건수"].max() * 1.18)
    ax.grid(axis="x", color="#e5e7eb")
    ax.grid(axis="y", visible=False)
    return save_fig(fig, out_dir / "04_top10_fire_causes.png")


def viz_05_response_vs_distance(df: pd.DataFrame, out_dir: Path) -> Path:
    """05: 출동소요시간 vs 현장거리 산점도 + 선형 추세선 + 상관계수."""
    scatter = df.dropna(subset=["현장거리(km)", "출동소요시간_분"]).copy()
    scatter = scatter[(scatter["현장거리(km)"] >= 0) & (scatter["출동소요시간_분"] > 0)]
    # 상위 1% 이상치 제거 — 분포 시각화 안정화
    distance_cap = scatter["현장거리(km)"].quantile(0.99)
    response_cap = scatter["출동소요시간_분"].quantile(0.99)
    plot_df = scatter[
        (scatter["현장거리(km)"] <= distance_cap)
        & (scatter["출동소요시간_분"] <= response_cap)
    ].copy()
    # 점이 너무 많으면 9000개로 다운샘플 (가독성)
    if len(plot_df) > 9000:
        plot_df = plot_df.sample(9000, random_state=42)

    corr = plot_df["현장거리(km)"].corr(plot_df["출동소요시간_분"])
    fig, ax = plt.subplots(figsize=(11.5, 7))
    # 자치구별로 색상 구분 (범례 숨김 — 너무 많아서)
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
    # 선형 추세선 — np.polyfit
    if len(plot_df) >= 2:
        slope, intercept = np.polyfit(
            plot_df["현장거리(km)"], plot_df["출동소요시간_분"], 1
        )
        x_line = np.linspace(
            plot_df["현장거리(km)"].min(), plot_df["현장거리(km)"].max(), 100
        )
        ax.plot(
            x_line,
            slope * x_line + intercept,
            color="#111827",
            linewidth=2.4,
            label="선형 추세",
        )
    add_title(
        ax, "출동소요시간 vs 현장거리", f"상위 1% 이상치 제외, 상관계수 r = {corr:.3f}"
    )
    ax.set_xlabel("현장거리(km)")
    ax.set_ylabel("출동소요시간(분)")
    ax.grid(color="#e5e7eb")
    ax.legend(loc="upper left", frameon=True)
    return save_fig(fig, out_dir / "05_response_time_vs_distance.png")


def viz_06_high_damage_map(df: pd.DataFrame, out_dir: Path) -> Path:
    """06: 재산피해 TOP100 위치 지도 + TOP15 가로 막대 (좌우 패널)."""
    geo = valid_geo_rows(df)
    # 피해액 > 0 인 화재만 추려 상위 100개
    top = (
        geo.dropna(subset=["재산피해액(천원)"])
        .query("`재산피해액(천원)` > 0")
        .sort_values("재산피해액(천원)", ascending=False)
        .head(100)
        .copy()
    )
    top["재산피해액_백만원"] = top["재산피해액(천원)"] / 1000

    # TOP100 CSV 저장 (랭킹 보고용)
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

    # 좌우 분할 figure — 좌: 지도, 우: TOP15 막대
    fig = plt.figure(figsize=(15.5, 8.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.08, 1.12], wspace=0.34)
    ax_map = fig.add_subplot(gs[0, 0])
    ax_rank = fig.add_subplot(gs[0, 1])

    # 좌측 — 배경 회색 점 + 빨간 버블
    ax_map.scatter(
        geo["경도"], geo["위도"], s=3, color="#d1d5db", alpha=0.16, linewidths=0
    )
    # 버블 크기 — log 스케일 (피해액 분포가 매우 우편향)
    sizes = 40 + 420 * np.log10(top["재산피해액(천원)"] + 1) / np.log10(
        top["재산피해액(천원)"].max() + 1
    )
    ax_map.scatter(
        top["경도"],
        top["위도"],
        s=sizes,
        color="#fca5a5",
        edgecolor="#7f1d1d",
        linewidth=0.8,
        alpha=0.74,
    )
    # TOP10 만 순위 숫자로 표기 — 너무 많으면 가독성 저하
    for rank, (_, row) in enumerate(top.head(10).iterrows(), start=1):
        ax_map.text(
            row["경도"],
            row["위도"],
            str(rank),
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
        )
    add_title(ax_map, "재산피해액 TOP 100 화재 위치", "숫자 표기는 피해액 상위 10건")
    ax_map.set_xlabel("경도")
    ax_map.set_ylabel("위도")
    ax_map.set_xlim(126.75, 127.18)
    ax_map.set_ylim(37.43, 37.68)
    ax_map.set_aspect("equal", adjustable="box")
    ax_map.grid(color="#e5e7eb")

    # 우측 — TOP15 가로 막대 (로그축, 피해액 절댓값 차이가 매우 큼)
    rank = top.head(15).copy()
    rank["순위"] = range(1, len(rank) + 1)
    labels = rank.apply(
        lambda r: (
            f"{int(r['순위'])}. {r.get('발생시군구', '')} | {str(r.get('발생일자', ''))}"
        ),
        axis=1,
    )
    y_pos = np.arange(len(rank))
    ax_rank.barh(y_pos, rank["재산피해액_백만원"], color="#dc2626", alpha=0.82)
    ax_rank.set_yticks(y_pos)
    ax_rank.set_yticklabels(labels, fontsize=8.6)
    ax_rank.invert_yaxis()  # 1위가 위에 오도록
    ax_rank.set_xscale("log")
    ax_rank.set_xlabel("재산피해액(백만원, 로그축)")
    ax_rank.set_title(
        "재산피해액 TOP 15", loc="left", fontsize=15, fontweight="bold", pad=14
    )
    ax_rank.grid(axis="x", color="#e5e7eb")
    ax_rank.grid(axis="y", visible=False)
    # 막대 끝 숫자
    for y, value in zip(y_pos, rank["재산피해액_백만원"]):
        ax_rank.text(value * 1.08, y, f"{value:,.1f}", va="center", fontsize=8.5)
    # x축 — 최소값 1 보장 + 최댓값에 여유
    ax_rank.set_xlim(
        max(rank["재산피해액_백만원"].min() * 0.72, 1),
        rank["재산피해액_백만원"].max() * 1.55,
    )

    return save_fig(fig, out_dir / "06_high_damage_fire_map_and_ranking.png")


def write_summary(
    df: pd.DataFrame, csv_path: Path, out_dir: Path, outputs: list[Path]
) -> Path:
    """전체 요약 JSON — 행수/연도/구별 분포/원인 TOP10/생성 파일 목록."""
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
        "top_districts": df.groupby("발생시군구")
        .size()
        .sort_values(ascending=False)
        .head(10)
        .to_dict(),
        "top_causes": df.groupby("발화요인_대분류")
        .size()
        .sort_values(ascending=False)
        .head(10)
        .to_dict(),
        "outputs": [p.name for p in outputs],
    }
    out_path = out_dir / "summary.json"
    out_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return out_path


def parse_args() -> argparse.Namespace:
    """CLI 인자 — 입력 CSV/출력 폴더/연도 범위 옵션."""
    parser = argparse.ArgumentParser(
        description="Build six static PNG fire-dispatch visualizations."
    )
    parser.add_argument(
        "--csv", type=Path, default=DEFAULT_CSV, help="Input fire dispatch CSV path"
    )
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUT, help="Output directory"
    )
    parser.add_argument(
        "--start-year", type=int, default=None, help="Optional inclusive start year"
    )
    parser.add_argument(
        "--end-year", type=int, default=None, help="Optional inclusive end year"
    )
    return parser.parse_args()


def main() -> None:
    """메인 — 스타일 설정 → 데이터 준비 → 6개 시각화 → 요약 저장."""
    setup_style()
    args = parse_args()
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    df = prepare_data(args.csv, args.start_year, args.end_year)
    # 6개 PNG 순차 생성 — 각 함수가 경로 반환
    outputs = [
        viz_01_monthly_trend(df, out_dir),
        viz_02_weekday_hour_heatmap(df, out_dir),
        viz_03_district_map(df, out_dir),
        viz_04_cause_top10(df, out_dir),
        viz_05_response_vs_distance(df, out_dir),
        viz_06_high_damage_map(df, out_dir),
    ]
    summary_path = write_summary(df, args.csv, out_dir, outputs)

    # 콘솔 출력 — 실행 성공 후 요약
    print(f"rows={len(df):,}")
    print(f"years={int(df['발생연도'].min())}-{int(df['발생연도'].max())}")
    print(f"output_dir={out_dir}")
    for path in outputs:
        print(f"created={path}")
    print(f"created={summary_path}")


if __name__ == "__main__":
    main()
