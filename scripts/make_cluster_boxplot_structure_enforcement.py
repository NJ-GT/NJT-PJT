# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "0430" / "최종테이블0429.csv"
OUT_PATH = ROOT / "0430" / "군집별_구조노후도_단속위험도_boxplot.png"

ORDER = ["저위험군", "중위험군", "고위험군"]
PALETTE = {
    "저위험군": "#60A5FA",
    "중위험군": "#FBBF24",
    "고위험군": "#EF4444",
}


def set_korean_font() -> str:
    candidates = [
        r"C:\Windows\Fonts\malgun.ttf",
        r"C:\Windows\Fonts\NanumGothic.ttf",
        r"C:\Windows\Fonts\NotoSansKR-Regular.otf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            fm.fontManager.addfont(candidate)
            font_name = fm.FontProperties(fname=candidate).get_name()
            plt.rcParams["font.family"] = font_name
            return font_name
    plt.rcParams["axes.unicode_minus"] = False
    return "sans-serif"


def main() -> None:
    font_name = set_korean_font()
    sns.set_theme(style="whitegrid", rc={"font.family": font_name})
    plt.rcParams["font.family"] = font_name
    plt.rcParams["axes.unicode_minus"] = False

    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()

    use = df[["cluster_label", "구조노후도", "단속위험도"]].copy()
    use["cluster_label"] = pd.Categorical(
        use["cluster_label"], categories=ORDER, ordered=True
    )
    for col in ["구조노후도", "단속위험도"]:
        use[col] = pd.to_numeric(use[col], errors="coerce")
    use = use.dropna(subset=["cluster_label", "구조노후도", "단속위험도"])

    summary = (
        use.groupby("cluster_label", observed=True)[["구조노후도", "단속위험도"]]
        .agg(["count", "mean", "median"])
        .reindex(ORDER)
    )

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 7.2), dpi=180)
    fig.patch.set_facecolor("#f6f8fb")

    plot_specs = [
        ("구조노후도", "구조노후도", "건물 노후 위험 지표"),
        ("단속위험도", "단속위험도", "불법주정차/단속 위험 지표"),
    ]

    for ax, (col, title, subtitle) in zip(axes, plot_specs):
        ax.set_facecolor("#ffffff")
        sns.boxplot(
            data=use,
            x="cluster_label",
            y=col,
            order=ORDER,
            palette=PALETTE,
            width=0.55,
            linewidth=1.35,
            fliersize=2.0,
            ax=ax,
        )
        sns.stripplot(
            data=use.sample(min(len(use), 900), random_state=42),
            x="cluster_label",
            y=col,
            order=ORDER,
            color="#111827",
            alpha=0.13,
            size=2.2,
            jitter=0.22,
            ax=ax,
        )
        ax.set_title(title, fontsize=18, fontweight="bold", color="#111827", pad=18)
        ax.text(
            0.5,
            1.01,
            subtitle,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=10.5,
            color="#667085",
        )
        ax.set_xlabel("")
        ax.set_ylabel("원본 지표값", fontsize=12, color="#344054")
        ax.tick_params(axis="x", labelsize=12)
        ax.tick_params(axis="y", labelsize=10.5)
        ax.grid(axis="y", alpha=0.25)
        ax.grid(axis="x", visible=False)
        for spine in ax.spines.values():
            spine.set_color("#d0d5dd")

        ymax = use[col].quantile(0.985)
        for idx, label in enumerate(ORDER):
            row = summary.loc[label, col]
            ax.text(
                idx,
                ymax,
                f"n={int(row['count']):,}\n평균 {row['mean']:.3f}\n중앙 {row['median']:.3f}",
                ha="center",
                va="top",
                fontsize=9.2,
                color="#1f2937",
                bbox=dict(
                    boxstyle="round,pad=0.28",
                    facecolor="#ffffff",
                    edgecolor="#e5e7eb",
                    alpha=0.88,
                ),
            )

    fig.suptitle(
        "군집별 구조노후도 · 단속위험도 분포",
        fontsize=24,
        fontweight="bold",
        color="#101828",
        y=0.98,
    )
    fig.text(
        0.5,
        0.925,
        "최종테이블0429 기준 | box=사분위범위(IQR), 중앙선=중앙값, 점=표본 일부",
        ha="center",
        fontsize=11.5,
        color="#667085",
    )
    fig.tight_layout(rect=[0.03, 0.04, 0.97, 0.89])
    fig.savefig(OUT_PATH, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(OUT_PATH)


if __name__ == "__main__":
    main()
