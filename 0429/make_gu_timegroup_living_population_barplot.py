from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "새 폴더" / "날짜별_concat" / "생활인구수_한글컬럼_방문4시간그룹.csv"
OUTPUT_PNG = BASE_DIR / "새 폴더" / "날짜별_concat" / "구별_시간대별_25개월평균_방문생활인구수_barplot.png"
OUTPUT_CSV = BASE_DIR / "새 폴더" / "날짜별_concat" / "구별_시간대별_25개월평균_방문생활인구수.csv"

TIME_COLS = [
    "방문생활인구수_00_03시",
    "방문생활인구수_04_07시",
    "방문생활인구수_08_11시",
    "방문생활인구수_12_15시",
    "방문생활인구수_16_19시",
    "방문생활인구수_20_23시",
]

TIME_LABELS = {
    "방문생활인구수_00_03시": "00-03시",
    "방문생활인구수_04_07시": "04-07시",
    "방문생활인구수_08_11시": "08-11시",
    "방문생활인구수_12_15시": "12-15시",
    "방문생활인구수_16_19시": "16-19시",
    "방문생활인구수_20_23시": "20-23시",
}


def set_korean_font() -> None:
    available = {font.name for font in fm.fontManager.ttflist}
    for name in ["Malgun Gothic", "AppleGothic", "NanumGothic"]:
        if name in available:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False


def main() -> None:
    set_korean_font()
    df = pd.read_csv(INPUT_PATH, encoding="utf-8-sig", dtype=str)

    for col in TIME_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    monthly = (
        df.groupby(["파일기준년월", "시군구명"], as_index=False)[TIME_COLS]
        .sum()
        .sort_values(["파일기준년월", "시군구명"])
    )

    avg = (
        monthly.groupby("시군구명", as_index=False)[TIME_COLS]
        .mean()
    )
    avg["총방문생활인구수"] = avg[TIME_COLS].sum(axis=1)
    avg = avg.sort_values("총방문생활인구수", ascending=False)
    gu_order = avg["시군구명"].tolist()
    avg.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(14.5, 7.5), dpi=180)
    fig.patch.set_facecolor("#F8FAFC")
    ax.set_facecolor("#F8FAFC")

    x = np.arange(len(gu_order))
    width = 0.12
    palette = list(plt.get_cmap("Set2").colors[: len(TIME_COLS)])
    offsets = (np.arange(len(TIME_COLS)) - (len(TIME_COLS) - 1) / 2) * width

    for idx, col in enumerate(TIME_COLS):
        values = avg.set_index("시군구명").loc[gu_order, col].to_numpy()
        ax.bar(
            x + offsets[idx],
            values,
            width=width,
            label=TIME_LABELS[col],
            color=palette[idx],
            alpha=0.9,
        )

    ax.set_title("구별 시간대별 25개월 평균 방문생활인구수 (단기체류외국인)", loc="left", fontsize=19, weight="bold", color="#111827", pad=20)
    ax.text(
        0,
        1.01,
        "기준: 2024.03~2026.03",
        transform=ax.transAxes,
        fontsize=10.5,
        color="#64748B",
    )

    ax.set_xlabel("구", fontsize=11, color="#475569", labelpad=10)
    ax.set_ylabel("25개월 평균 방문생활인구수", fontsize=11, color="#475569", labelpad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(gu_order, rotation=35, ha="right")
    ax.grid(axis="y", color="#E2E8F0", linewidth=0.9)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.legend(
        title="시간대",
        ncol=3,
        frameon=True,
        facecolor="white",
        edgecolor="#CBD5E1",
        loc="upper right",
        fontsize=9.5,
        title_fontsize=10,
    )

    ax.tick_params(axis="x", labelsize=10.5, colors="#111827")
    ax.tick_params(axis="y", labelsize=9.5, colors="#64748B")
    ax.margins(x=0.02)
    fig.tight_layout(pad=2.0, rect=[0, 0, 1, 0.93])
    fig.savefig(OUTPUT_PNG, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    print(OUTPUT_PNG)
    print(OUTPUT_CSV)
    print(avg[["시군구명", *TIME_COLS]].to_string(index=False))


if __name__ == "__main__":
    main()
