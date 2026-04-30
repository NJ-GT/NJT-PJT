from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
GU_INPUT = BASE_DIR / "카드이용건수_이미지추정_구별.csv"
DONG_INPUT = BASE_DIR / "카드이용건수_이미지추정_동별.csv"
OUTPUT_CSV = BASE_DIR / "카드이용건수_이미지추정_구별_동별합계기준보정.csv"
OUTPUT_PNG = BASE_DIR / "카드이용건수_이미지추정_구별_동별합계기준보정_barplot.png"

TIME_COLS = ["T1", "T2", "T3", "T4", "T5", "T6"]


def set_korean_font() -> None:
    available = {font.name for font in fm.fontManager.ttflist}
    for name in ["Malgun Gothic", "AppleGothic", "NanumGothic"]:
        if name in available:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False


def main() -> None:
    set_korean_font()
    gu = pd.read_csv(GU_INPUT, encoding="utf-8-sig")
    dong = pd.read_csv(DONG_INPUT, encoding="utf-8-sig")

    for col in TIME_COLS:
        gu[col] = pd.to_numeric(gu[col], errors="coerce")
        dong[col] = pd.to_numeric(dong[col], errors="coerce")

    junggu_sum = dong[TIME_COLS].sum()
    junggu_original = gu.loc[gu["구"].eq("중구"), TIME_COLS].iloc[0]
    scale = (junggu_sum / junggu_original).replace([np.inf, -np.inf], np.nan).fillna(1)

    corrected = gu.copy()
    for col in TIME_COLS:
        corrected[col] = (corrected[col] * scale[col]).round(2)

    corrected.loc[corrected["구"].eq("중구"), TIME_COLS] = junggu_sum.round(2).to_numpy()
    corrected["비고"] = "동별_중구합계_기준_시간대별_스케일보정"
    corrected.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    x = np.arange(len(corrected))
    width = 0.12
    offsets = (np.arange(len(TIME_COLS)) - (len(TIME_COLS) - 1) / 2) * width
    palette = list(plt.get_cmap("Set2").colors[: len(TIME_COLS)])

    fig, ax = plt.subplots(figsize=(12.8, 7.0), dpi=180)
    fig.patch.set_facecolor("#F8FAFC")
    ax.set_facecolor("#F8FAFC")

    for i, col in enumerate(TIME_COLS):
        ax.bar(
            x + offsets[i],
            corrected[col].to_numpy(dtype=float),
            width=width,
            color=palette[i],
            label=col,
            alpha=0.92,
        )

    ax.set_title("구별 시간대별 평균 카드이용건수", loc="left", fontsize=18, weight="bold", color="#111827", pad=18)
    ax.set_xlabel("구", fontsize=11, color="#475569", labelpad=10)
    ax.set_ylabel("카드이용건수", fontsize=11, color="#475569", labelpad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(corrected["구"], rotation=35, ha="right")
    ax.grid(axis="y", color="#E2E8F0", linewidth=0.9)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(title="시간대구간", ncol=3, frameon=True, facecolor="white", edgecolor="#CBD5E1", loc="upper right")
    fig.tight_layout(pad=2.0, rect=[0, 0, 1, 0.94])
    fig.savefig(OUTPUT_PNG, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    print(OUTPUT_CSV)
    print(OUTPUT_PNG)
    print("scale")
    print(scale.round(3).to_string())
    print(corrected.to_string(index=False))


if __name__ == "__main__":
    main()
