from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "새 폴더" / "날짜별_concat" / "동별_25개월평균_방문생활인구수_빠른추정.csv"
OUTPUT_PNG = BASE_DIR / "새 폴더" / "날짜별_concat" / "중구_동별_top10_시간대별_25개월평균_방문생활인구수_barplot.png"
OUTPUT_CSV = BASE_DIR / "새 폴더" / "날짜별_concat" / "중구_동별_top10_시간대별_25개월평균_방문생활인구수.csv"

TIME_COLS = [
    "방문생활인구수_00_03시",
    "방문생활인구수_04_07시",
    "방문생활인구수_08_11시",
    "방문생활인구수_12_15시",
    "방문생활인구수_16_19시",
    "방문생활인구수_20_23시",
]

TIME_LABELS = ["00-03시", "04-07시", "08-11시", "12-15시", "16-19시", "20-23시"]
DISPLAY_NAME_FIX = {
    "산림동": "신당동",
    "신림동": "신당동",
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
    df = pd.read_csv(INPUT_PATH, encoding="utf-8-sig")
    junggu = df[df["추정_구"].eq("중구")].copy()
    for col in TIME_COLS:
        junggu[col] = pd.to_numeric(junggu[col], errors="coerce").fillna(0)

    junggu = junggu.sort_values("25개월평균_방문생활인구수", ascending=False).head(10)
    junggu.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    junggu["동표시"] = junggu["추정_동"].replace(DISPLAY_NAME_FIX)
    dong_order = junggu["동표시"].tolist()
    x = np.arange(len(dong_order))
    width = min(0.12, 0.72 / len(TIME_COLS))
    offsets = (np.arange(len(TIME_COLS)) - (len(TIME_COLS) - 1) / 2) * width
    palette = list(plt.get_cmap("Set2").colors[: len(TIME_COLS)])

    fig_width = max(13.5, len(dong_order) * 0.78)
    fig, ax = plt.subplots(figsize=(fig_width, 7.8), dpi=180)
    fig.patch.set_facecolor("#F8FAFC")
    ax.set_facecolor("#F8FAFC")

    values_by_dong = junggu.set_index("동표시")
    for idx, (col, label) in enumerate(zip(TIME_COLS, TIME_LABELS)):
        ax.bar(
            x + offsets[idx],
            values_by_dong.loc[dong_order, col].to_numpy(),
            width=width,
            label=label,
            color=palette[idx],
            alpha=0.9,
        )

    ax.set_title("중구 동별 TOP 10 시간대별 25개월 평균 방문생활인구수 (단기체류외국인)", loc="left", fontsize=19, weight="bold", color="#111827", pad=20)
    ax.text(
        0,
        1.01,
        "기준: 2024.03~2026.03",
        transform=ax.transAxes,
        fontsize=10.5,
        color="#64748B",
    )
    ax.set_xlabel("동", fontsize=11, color="#475569", labelpad=10)
    ax.set_ylabel("25개월 평균 방문생활인구수", fontsize=11, color="#475569", labelpad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(dong_order, rotation=45, ha="right")
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
    ax.tick_params(axis="x", labelsize=9.5, colors="#111827")
    ax.tick_params(axis="y", labelsize=9.5, colors="#64748B")
    ax.margins(x=0.01)
    fig.tight_layout(pad=2.0, rect=[0, 0, 1, 0.93])
    fig.savefig(OUTPUT_PNG, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    print(OUTPUT_PNG)
    print(OUTPUT_CSV)
    print(junggu[["추정_동", *TIME_COLS, "25개월평균_방문생활인구수"]].to_string(index=False))


if __name__ == "__main__":
    main()
