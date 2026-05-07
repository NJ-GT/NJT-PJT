# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE = Path(__file__).resolve().parents[1]
SRC = BASE / "원본데이터" / "서울시 외국인관광도시민박업 인허가 정보.csv"
OUT_PNG = BASE / "data" / "foreign_homestay_10gu_2021_2025_barplot.png"
OUT_CSV = BASE / "data" / "foreign_homestay_10gu_2021_2025_counts.csv"

YEARS = list(range(2021, 2026))
GU_10 = [
    "강남구",
    "강서구",
    "마포구",
    "서초구",
    "성동구",
    "송파구",
    "영등포구",
    "용산구",
    "종로구",
    "중구",
]
PALETTE = [
    "#66C2A5",
    "#FC8D62",
    "#8DA0CB",
    "#E78AC3",
    "#A6D854",
    "#FFD92F",
    "#E5C494",
    "#B3B3B3",
    "#A6CEE3",
    "#FDBF6F",
]


def set_korean_font() -> None:
    for font in fm.findSystemFonts():
        if "malgun" in font.lower():
            plt.rcParams["font.family"] = fm.FontProperties(fname=font).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False


def load_counts() -> pd.DataFrame:
    df = pd.read_csv(SRC, encoding="utf-8-sig", low_memory=False)
    df["년도"] = pd.to_datetime(df["인허가일자"], errors="coerce").dt.year
    df["구"] = (
        df["지번주소"].fillna(df["도로명주소"]).str.extract(r"서울특별시\s+(\S+구)")
    )
    filtered = df[df["년도"].isin(YEARS) & df["구"].isin(GU_10)].copy()

    counts = filtered.groupby(["년도", "구"]).size().rename("갯수").reset_index()
    full_index = pd.MultiIndex.from_product([YEARS, GU_10], names=["년도", "구"])
    return (
        counts.set_index(["년도", "구"]).reindex(full_index, fill_value=0).reset_index()
    )


def plot(counts: pd.DataFrame) -> None:
    gu_order = (
        counts.groupby("구")["갯수"].sum().sort_values(ascending=False).index.tolist()
    )
    pivot = counts.pivot(index="년도", columns="구", values="갯수").reindex(
        index=YEARS, columns=gu_order
    )

    fig, ax = plt.subplots(figsize=(18, 8), facecolor="#F8F9FC")
    ax.set_facecolor("white")

    x = np.arange(len(YEARS))
    width = 0.09
    offsets = (np.arange(len(gu_order)) - (len(gu_order) - 1) / 2) * width

    for idx, gu in enumerate(gu_order):
        values = pivot[gu].to_numpy()
        bars = ax.bar(
            x + offsets[idx],
            values,
            width=width,
            label=gu,
            color=PALETTE[idx],
            alpha=0.9,
            edgecolor="white",
            linewidth=0.6,
        )
        for bar, value in zip(bars, values):
            if value >= 10:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1,
                    f"{int(value)}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="#1F2937",
                )

    ax.set_title(
        "외국인관광도시민박업 신규 인허가 건수 (2021-2025, 10개구)",
        fontsize=18,
        fontweight="bold",
        pad=18,
    )
    ax.set_xlabel("년도", fontsize=12, fontweight="bold")
    ax.set_ylabel("신규 인허가 건수", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(YEARS, fontsize=11)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.legend(
        title="구",
        ncol=5,
        loc="upper left",
        bbox_to_anchor=(0, 1.03),
        frameon=True,
        fontsize=18,
        title_fontsize=20,
    )
    ax.spines[["top", "right"]].set_visible(False)

    ymax = max(10, int(pivot.to_numpy().max()))
    ax.set_ylim(0, ymax * 1.18)
    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    set_korean_font()
    counts = load_counts()
    gu_order = (
        counts.groupby("구")["갯수"].sum().sort_values(ascending=False).index.tolist()
    )
    counts["구"] = pd.Categorical(counts["구"], categories=gu_order, ordered=True)
    counts = counts.sort_values(["년도", "구"]).reset_index(drop=True)
    counts["구"] = counts["구"].astype(str)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    counts.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    plot(counts)
    print(f"saved: {OUT_PNG}")
    print(f"saved: {OUT_CSV}")


if __name__ == "__main__":
    main()
