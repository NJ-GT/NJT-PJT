# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
SRC = BASE / "0424" / "data" / "최종_화재위험_분석결과_0428.csv"
OUT = BASE / "0424" / "data" / "군집별_최고위험시설_TOP10_0428.png"

RISK_LABELS = ["저위험군", "중위험군", "고위험군"]
COLORS = {
    "저위험군": "#15803D",
    "중위험군": "#F59E0B",
    "고위험군": "#DC2626",
}


def trim_name(name: object, max_len: int = 15) -> str:
    text = str(name)
    return text if len(text) <= max_len else text[: max_len - 1] + "…"


def main() -> None:
    df = pd.read_csv(SRC, encoding="utf-8-sig")
    cluster_col = "cluster_k3" if "cluster_k3" in df.columns else "cluster"

    cluster_order = df.groupby(cluster_col)["최종_화재위험점수"].mean().sort_values().index.tolist()
    label_map = {cluster: label for cluster, label in zip(cluster_order, RISK_LABELS)}
    df["위험군"] = df[cluster_col].map(label_map)

    top_by_group = {
        label: (
            df[df["위험군"].eq(label)]
            .sort_values("최종_화재위험점수", ascending=False)
            .head(10)
            .sort_values("최종_화재위험점수", ascending=True)
        )
        for label in RISK_LABELS
    }

    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 3, figsize=(18, 8.4), facecolor="#F8FAFC")
    for ax, label in zip(axes, RISK_LABELS):
        top = top_by_group[label]
        y_labels = [trim_name(v) for v in top["숙소명"]]
        scores = top["최종_화재위험점수"]
        bars = ax.barh(
            y_labels,
            scores,
            color=COLORS[label],
            edgecolor="white",
            linewidth=1.1,
            height=0.72,
        )

        for bar, (_, row) in zip(bars, top.iterrows()):
            ax.text(
                bar.get_width() + 0.35,
                bar.get_y() + bar.get_height() / 2,
                f"{row['최종_화재위험점수']:.1f} | {row['구']}",
                va="center",
                ha="left",
                fontsize=9.5,
                color="#0F172A",
                fontweight="bold",
            )

        mean_score = df.loc[df["위험군"].eq(label), "최종_화재위험점수"].mean()
        count = df["위험군"].eq(label).sum()
        ax.set_title(
            f"{label}\nN={count:,} | 평균 {mean_score:.1f}점",
            loc="left",
            fontsize=16,
            fontweight="bold",
            color="#0F172A",
            pad=14,
        )
        ax.set_xlabel("최종 화재위험점수", fontsize=11, color="#475569")
        ax.grid(axis="x", color="#E2E8F0", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#CBD5E1")
        ax.tick_params(axis="y", labelsize=10)
        ax.tick_params(axis="x", labelsize=10, colors="#475569")
        ax.set_xlim(0, max(65, scores.max() + 7))

    fig.suptitle(
        "군집별 최고위험시설 TOP 10",
        x=0.055,
        y=0.965,
        ha="left",
        fontsize=25,
        fontweight="bold",
        color="#0F172A",
    )
    fig.text(
        0.055,
        0.915,
        "각 군집 내부에서 최종 화재위험점수가 높은 숙소 10개를 표시했습니다. 군집명은 평균 위험점수 기준으로 재정렬했습니다.",
        ha="left",
        fontsize=12,
        color="#475569",
    )
    fig.text(
        0.055,
        0.045,
        "막대 오른쪽 표기는 위험점수와 자치구입니다.",
        ha="left",
        fontsize=11,
        color="#64748B",
    )
    fig.subplots_adjust(top=0.82, bottom=0.12, left=0.11, right=0.97, wspace=0.5)
    fig.savefig(OUT, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"saved={OUT}")


if __name__ == "__main__":
    main()
