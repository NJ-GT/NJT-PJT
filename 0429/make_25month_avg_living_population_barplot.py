from __future__ import annotations

from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "새 폴더" / "날짜별_concat" / "생활인구수_한글컬럼_방문4시간그룹.csv"
OUTPUT_PNG = BASE_DIR / "새 폴더" / "날짜별_concat" / "구별_25개월평균_생활인구수_top10_barplot.png"
OUTPUT_CSV = BASE_DIR / "새 폴더" / "날짜별_concat" / "구별_25개월평균_생활인구수_top10.csv"


def set_korean_font() -> None:
    candidates = ["Malgun Gothic", "AppleGothic", "NanumGothic"]
    available = {font.name for font in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False


def main() -> None:
    set_korean_font()

    df = pd.read_csv(INPUT_PATH, encoding="utf-8-sig", dtype=str)
    for col in ["월평균_외국인_상주생활인구수", "월평균_외국인_방문생활인구수"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["생활인구수"] = df["월평균_외국인_상주생활인구수"] + df["월평균_외국인_방문생활인구수"]

    monthly_gu = (
        df.groupby(["파일기준년월", "시군구명"], as_index=False)["생활인구수"]
        .sum()
        .rename(columns={"생활인구수": "월별_구합계_생활인구수"})
    )

    top10 = (
        monthly_gu.groupby("시군구명", as_index=False)["월별_구합계_생활인구수"]
        .mean()
        .rename(columns={"월별_구합계_생활인구수": "25개월평균_생활인구수"})
        .sort_values("25개월평균_생활인구수", ascending=False)
        .head(10)
    )
    top10["순위"] = range(1, len(top10) + 1)
    top10 = top10[["순위", "시군구명", "25개월평균_생활인구수"]]
    top10.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    plot_df = top10.sort_values("25개월평균_생활인구수", ascending=True)
    colors = ["#2563EB" if rank <= 3 else "#94A3B8" for rank in plot_df["순위"]]

    fig, ax = plt.subplots(figsize=(11.8, 7.0), dpi=180)
    fig.patch.set_facecolor("#F8FAFC")
    ax.set_facecolor("#F8FAFC")

    bars = ax.barh(plot_df["시군구명"], plot_df["25개월평균_생활인구수"], color=colors, height=0.62)

    max_value = plot_df["25개월평균_생활인구수"].max()
    for bar, value in zip(bars, plot_df["25개월평균_생활인구수"]):
        ax.text(
            value + max_value * 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"{value:,.0f}",
            va="center",
            ha="left",
            fontsize=10.5,
            color="#0F172A",
            weight="bold",
        )

    ax.set_title("구별 25개월 평균 생활인구수 TOP 10", loc="left", fontsize=19, weight="bold", color="#111827", pad=16)
    ax.text(
        0,
        1.02,
        "기준: 2024.03~2026.03 월별 구 합계 평균 · 생활인구수 = 외국인 상주생활인구수 + 방문생활인구수",
        transform=ax.transAxes,
        fontsize=10.5,
        color="#64748B",
    )

    ax.set_xlabel("25개월 평균 생활인구수", fontsize=10.5, color="#475569", labelpad=10)
    ax.set_ylabel("")
    ax.grid(axis="x", color="#E2E8F0", linewidth=0.9)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="x", colors="#64748B", labelsize=9.5)
    ax.tick_params(axis="y", colors="#111827", labelsize=11)
    ax.set_xlim(0, max_value * 1.16)

    fig.tight_layout(pad=2.0)
    fig.savefig(OUTPUT_PNG, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    print(OUTPUT_PNG)
    print(OUTPUT_CSV)
    print(top10.to_string(index=False))


if __name__ == "__main__":
    main()
