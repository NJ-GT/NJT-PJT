from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "새 폴더" / "날짜별_concat"

GU_INPUT = DATA_DIR / "구별_시간대별_25개월평균_방문생활인구수.csv"
JUNGGU_INPUT = DATA_DIR / "중구_동별_시간대별_25개월평균_방문생활인구수.csv"

GU_OUTPUT = DATA_DIR / "시간대별_구별_25개월평균_방문생활인구수_barplot.png"
JUNGGU_OUTPUT = DATA_DIR / "시간대별_중구동별_top10_25개월평균_방문생활인구수_barplot.png"

TIME_COLS = [
    "방문생활인구수_00_03시",
    "방문생활인구수_04_07시",
    "방문생활인구수_08_11시",
    "방문생활인구수_12_15시",
    "방문생활인구수_16_19시",
    "방문생활인구수_20_23시",
]
TIME_LABELS = ["00-03시", "04-07시", "08-11시", "12-15시", "16-19시", "20-23시"]


def set_korean_font() -> None:
    available = {font.name for font in fm.fontManager.ttflist}
    for name in ["Malgun Gothic", "AppleGothic", "NanumGothic"]:
        if name in available:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False


def grouped_time_plot(
    df: pd.DataFrame,
    entity_col: str,
    title: str,
    subtitle: str,
    output_path: Path,
    top_n: int | None = None,
) -> None:
    work = df.copy()
    for col in TIME_COLS:
        work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0)

    work["총방문생활인구수"] = work[TIME_COLS].sum(axis=1)
    work = work.sort_values("총방문생활인구수", ascending=False)
    if top_n is not None:
        work = work.head(top_n)

    entities = work[entity_col].tolist()
    x = np.arange(len(TIME_LABELS))
    width = min(0.08, 0.78 / max(len(entities), 1))
    offsets = (np.arange(len(entities)) - (len(entities) - 1) / 2) * width

    palette = [
        "#2563EB", "#0F9F6E", "#F59E0B", "#DC2626", "#7C3AED",
        "#64748B", "#14B8A6", "#F97316", "#DB2777", "#475569",
    ]

    fig, ax = plt.subplots(figsize=(14.5, 7.6), dpi=180)
    fig.patch.set_facecolor("#F8FAFC")
    ax.set_facecolor("#F8FAFC")

    values = work.set_index(entity_col)
    for idx, entity in enumerate(entities):
        ax.bar(
            x + offsets[idx],
            values.loc[entity, TIME_COLS].to_numpy(dtype=float),
            width=width,
            label=entity,
            color=palette[idx % len(palette)],
            alpha=0.92,
        )

    ax.set_title(title, loc="left", fontsize=19, weight="bold", color="#111827", pad=18)
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=10.5, color="#64748B")
    ax.set_xlabel("시간대", fontsize=11, color="#475569", labelpad=10)
    ax.set_ylabel("25개월 평균 방문생활인구수", fontsize=11, color="#475569", labelpad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(TIME_LABELS)
    ax.grid(axis="y", color="#E2E8F0", linewidth=0.9)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="x", labelsize=10.5, colors="#111827")
    ax.tick_params(axis="y", labelsize=9.5, colors="#64748B")
    ax.legend(
        title=entity_col.replace("추정_", ""),
        ncol=2,
        frameon=True,
        facecolor="white",
        edgecolor="#CBD5E1",
        loc="upper right",
        fontsize=8.8,
        title_fontsize=10,
    )
    ax.margins(x=0.04)
    fig.tight_layout(pad=2.0)
    fig.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    set_korean_font()
    gu = pd.read_csv(GU_INPUT, encoding="utf-8-sig")
    junggu = pd.read_csv(JUNGGU_INPUT, encoding="utf-8-sig")

    grouped_time_plot(
        gu,
        entity_col="시군구명",
        title="시간대별 구별 25개월 평균 방문생활인구수",
        subtitle="기준: 2024.03~2026.03 · x축 시간대 정렬 · hue=구",
        output_path=GU_OUTPUT,
    )
    grouped_time_plot(
        junggu,
        entity_col="추정_동",
        title="시간대별 중구 동별 25개월 평균 방문생활인구수 TOP 10",
        subtitle="기준: 2024.03~2026.03 · 상권 중심점 기준 법정동 빠른 추정 · hue=동",
        output_path=JUNGGU_OUTPUT,
        top_n=10,
    )

    print(GU_OUTPUT)
    print(JUNGGU_OUTPUT)


if __name__ == "__main__":
    main()
