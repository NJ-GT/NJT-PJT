# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
SRC = BASE / "0424" / "data" / "분석변수_최종테이블0428.csv"
OUT = BASE / "0424" / "data" / "구별_소화용수_거리등급_시각화.png"

GRADE_LABELS = {
    0: "20m 이내",
    1: "20~40m",
    2: "40m 초과",
}
COLORS = {
    0: "#16A34A",
    1: "#F59E0B",
    2: "#DC2626",
}


def main() -> None:
    df = pd.read_csv(SRC, encoding="utf-8-sig")
    df["최근접_소화용수_거리등급"] = pd.to_numeric(df["최근접_소화용수_거리등급"], errors="coerce")
    df = df.dropna(subset=["구", "최근접_소화용수_거리등급"]).copy()
    df["최근접_소화용수_거리등급"] = df["최근접_소화용수_거리등급"].astype(int)

    counts = (
        df.groupby(["구", "최근접_소화용수_거리등급"])
        .size()
        .rename("시설수")
        .reset_index()
    )
    total = df.groupby("구").size().rename("전체시설수")
    counts = counts.merge(total, on="구")
    counts["비율"] = counts["시설수"] / counts["전체시설수"] * 100

    pivot_pct = (
        counts.pivot(index="구", columns="최근접_소화용수_거리등급", values="비율")
        .fillna(0)
        .reindex(columns=[0, 1, 2], fill_value=0)
    )
    gu_order = pivot_pct.sort_values([0, 1], ascending=[True, True]).index.tolist()

    summary = (
        df.groupby("구")
        .agg(
            평균등급=("최근접_소화용수_거리등급", "mean"),
            시설수=("숙소명", "size"),
            사십미터이내비율=("최근접_소화용수_거리등급", lambda s: s.le(1).mean() * 100),
        )
        .reindex(gu_order)
    )

    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig = plt.figure(figsize=(16, 9.8), facecolor="#F8FAFC")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.9, 1], wspace=0.18)
    ax = fig.add_subplot(gs[0, 0], facecolor="white")
    ax2 = fig.add_subplot(gs[0, 1], facecolor="white")

    y = range(len(gu_order))
    left = [0.0] * len(gu_order)
    for grade in [0, 1, 2]:
        vals = pivot_pct.loc[gu_order, grade].to_list()
        bars = ax.barh(
            y,
            vals,
            left=left,
            color=COLORS[grade],
            edgecolor="white",
            linewidth=1.2,
            height=0.72,
            label=GRADE_LABELS[grade],
        )
        for idx, (bar, val) in enumerate(zip(bars, vals)):
            if val >= 7:
                ax.text(
                    left[idx] + val / 2,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="white",
                    fontweight="bold",
                )
        left = [a + b for a, b in zip(left, vals)]

    ax.set_yticks(list(y), gu_order, fontsize=12)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("구성비 (%)", fontsize=11, color="#475569")
    ax.set_title("구별 거리등급 구성", loc="left", fontsize=16, fontweight="bold", pad=22)
    ax.grid(axis="x", color="#E2E8F0", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#CBD5E1")
    ax.legend(
        ncols=3,
        loc="lower left",
        bbox_to_anchor=(0, 1.055),
        frameon=False,
        fontsize=11,
        columnspacing=1.4,
        handlelength=1.2,
    )

    avg = summary["평균등급"].to_list()
    bar_colors = ["#16A34A" if v < 1.0 else "#F59E0B" if v < 1.2 else "#DC2626" for v in avg]
    bars2 = ax2.barh(y, avg, color=bar_colors, edgecolor="white", linewidth=1.2, height=0.72)
    for bar, val, within40 in zip(bars2, avg, summary["사십미터이내비율"]):
        ax2.text(
            val + 0.035,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}",
            va="center",
            ha="left",
            fontsize=10,
            color="#0F172A",
            fontweight="bold",
        )
        ax2.text(
            0.02,
            bar.get_y() + bar.get_height() / 2,
            f"40m 이내 {within40:.0f}%",
            va="center",
            ha="left",
            fontsize=9,
            color="white",
            fontweight="bold",
        )

    ax2.set_yticks(list(y), [""] * len(gu_order))
    ax2.invert_yaxis()
    ax2.set_xlim(0, 2.25)
    ax2.set_xlabel("평균 등급", fontsize=11, color="#475569")
    ax2.set_title("평균 거리등급", loc="left", fontsize=16, fontweight="bold", pad=22)
    ax2.grid(axis="x", color="#E2E8F0", linewidth=0.8)
    ax2.set_axisbelow(True)
    ax2.spines[["top", "right", "left"]].set_visible(False)
    ax2.spines["bottom"].set_color("#CBD5E1")

    fig.suptitle("구별 소화용수 접근성 등급", x=0.055, y=0.975, ha="left", fontsize=21, fontweight="bold", color="#0F172A")
    fig.text(
        0.055,
        0.935,
        "0=20m 이내, 1=20~40m, 2=40m 초과 | 낮을수록 가까움",
        ha="left",
        fontsize=12,
        color="#475569",
    )
    fig.text(
        0.055,
        0.035,
        "초록 비중이 클수록 소화용수 접근성이 좋고, 빨강 비중이 클수록 최근접 소화용수가 40m를 초과합니다.",
        ha="left",
        fontsize=11,
        color="#64748B",
    )
    fig.subplots_adjust(top=0.8, bottom=0.1, left=0.13, right=0.97)
    fig.savefig(OUT, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"saved={OUT}")


if __name__ == "__main__":
    main()
