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
OUT_PATH = ROOT / "0430" / "군집별_도로폭위험도_boxplot.png"

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
    return "sans-serif"


def main() -> None:
    font_name = set_korean_font()
    sns.set_theme(style="whitegrid", rc={"font.family": font_name})
    plt.rcParams["font.family"] = font_name
    plt.rcParams["axes.unicode_minus"] = False

    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()

    use = df[["cluster_label", "도로폭위험도"]].copy()
    use["cluster_label"] = pd.Categorical(
        use["cluster_label"], categories=ORDER, ordered=True
    )
    use["도로폭위험도"] = pd.to_numeric(use["도로폭위험도"], errors="coerce")
    use = use.dropna()

    summary = (
        use.groupby("cluster_label", observed=True)["도로폭위험도"]
        .agg(["count", "mean", "median"])
        .reindex(ORDER)
    )

    fig, ax = plt.subplots(figsize=(10.8, 7.4), dpi=180)
    fig.patch.set_facecolor("#f6f8fb")
    ax.set_facecolor("#ffffff")

    sns.boxplot(
        data=use,
        x="cluster_label",
        y="도로폭위험도",
        order=ORDER,
        hue="cluster_label",
        palette=PALETTE,
        width=0.52,
        linewidth=1.45,
        fliersize=2.3,
        legend=False,
        ax=ax,
    )
    sns.stripplot(
        data=use.sample(min(len(use), 1000), random_state=42),
        x="cluster_label",
        y="도로폭위험도",
        order=ORDER,
        color="#111827",
        alpha=0.13,
        size=2.2,
        jitter=0.22,
        ax=ax,
    )

    ymax = use["도로폭위험도"].quantile(0.985)
    for idx, label in enumerate(ORDER):
        row = summary.loc[label]
        ax.text(
            idx,
            ymax,
            f"n={int(row['count']):,}\n평균 {row['mean']:.3f}\n중앙 {row['median']:.3f}",
            ha="center",
            va="top",
            fontsize=10.4,
            color="#1f2937",
            bbox=dict(
                boxstyle="round,pad=0.32",
                facecolor="#ffffff",
                edgecolor="#e5e7eb",
                alpha=0.9,
            ),
        )

    ax.set_title(
        "군집별 도로폭위험도 분포",
        fontsize=24,
        fontweight="bold",
        color="#101828",
        pad=25,
    )
    ax.text(
        0.5,
        1.015,
        "도로폭이 좁을수록 위험도가 높게 반영된 지표 | box=IQR, 중앙선=중앙값, 점=표본 일부",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=11.5,
        color="#667085",
    )
    ax.set_xlabel("")
    ax.set_ylabel("원본 지표값", fontsize=13, color="#344054")
    ax.tick_params(axis="x", labelsize=13)
    ax.tick_params(axis="y", labelsize=11)
    ax.grid(axis="y", alpha=0.25)
    ax.grid(axis="x", visible=False)
    for spine in ax.spines.values():
        spine.set_color("#d0d5dd")

    fig.tight_layout(rect=[0.03, 0.03, 0.97, 0.93])
    fig.savefig(OUT_PATH, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(OUT_PATH)


if __name__ == "__main__":
    main()
