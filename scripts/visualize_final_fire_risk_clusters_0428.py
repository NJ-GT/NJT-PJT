# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


BASE = Path(__file__).resolve().parents[1]
SRC = BASE / "0424" / "data" / "최종_화재위험_분석결과_0428.csv"
OUT = BASE / "0424" / "data" / "최종_화재위험_군집화_대시보드_0428.png"

FEATURES = [
    "구조노후도",
    "단속위험도",
    "도로폭위험도",
    "최근접_소화용수_거리등급",
    "소방위험도_점수",
    "연면적",
    "집중도",
    "주변건물수",
    "총층수",
]

FEATURE_LABELS = {
    "구조노후도": "구조\n노후",
    "단속위험도": "단속\n위험",
    "도로폭위험도": "도로폭\n위험",
    "최근접_소화용수_거리등급": "소화용수\n거리",
    "소방위험도_점수": "소방\n위험",
    "연면적": "연면적",
    "집중도": "집중도",
    "주변건물수": "주변\n건물",
    "총층수": "총층수",
}

RISK_LABELS = ["저위험군", "중위험군", "고위험군"]
COLORS = {
    "저위험군": "#15803D",
    "중위험군": "#F59E0B",
    "고위험군": "#DC2626",
}


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(SRC, encoding="utf-8-sig")
    cluster_order = df.groupby("cluster_k3")["최종_화재위험점수"].mean().sort_values().index.tolist()
    label_map = {cluster: label for cluster, label in zip(cluster_order, RISK_LABELS)}
    df["위험군"] = df["cluster_k3"].map(label_map)
    df["위험군"] = pd.Categorical(df["위험군"], categories=RISK_LABELS, ordered=True)

    x = df[FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0)
    scaled = pd.DataFrame(MinMaxScaler().fit_transform(x), columns=FEATURES, index=df.index)
    scaled["위험군"] = df["위험군"]
    return df, scaled


def main() -> None:
    df, scaled = load_data()
    summary = (
        df.groupby("위험군", observed=True)
        .agg(
            시설수=("숙소명", "size"),
            평균점수=("최종_화재위험점수", "mean"),
            중앙점수=("최종_화재위험점수", "median"),
            최고점수=("최종_화재위험점수", "max"),
        )
        .reindex(RISK_LABELS)
    )
    profile = scaled.groupby("위험군", observed=True)[FEATURES].mean().reindex(RISK_LABELS)
    gu_cluster = (
        df.groupby(["구", "위험군"], observed=True)
        .size()
        .rename("시설수")
        .reset_index()
    )
    gu_total = df.groupby("구").size().rename("전체")
    gu_cluster = gu_cluster.merge(gu_total, on="구")
    gu_cluster["비율"] = gu_cluster["시설수"] / gu_cluster["전체"] * 100
    high_share = (
        gu_cluster[gu_cluster["위험군"].eq("고위험군")]
        .set_index("구")["비율"]
        .sort_values()
    )
    top10 = df.nlargest(10, "최종_화재위험점수").sort_values("최종_화재위험점수")

    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig = plt.figure(figsize=(18, 10.5), facecolor="#F8FAFC")
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.08, 1.1], height_ratios=[1.0, 1.08], hspace=0.35, wspace=0.28)
    ax_summary = fig.add_subplot(gs[0, 0], facecolor="white")
    ax_box = fig.add_subplot(gs[0, 1], facecolor="white")
    ax_heat = fig.add_subplot(gs[0, 2], facecolor="white")
    ax_gu = fig.add_subplot(gs[1, 0], facecolor="white")
    ax_map = fig.add_subplot(gs[1, 1], facecolor="white")
    ax_top = fig.add_subplot(gs[1, 2], facecolor="white")

    # Summary cards as bars
    xs = np.arange(len(RISK_LABELS))
    bars = ax_summary.bar(xs, summary["시설수"], color=[COLORS[l] for l in RISK_LABELS], width=0.56, edgecolor="white", linewidth=1.5)
    ax_summary2 = ax_summary.twinx()
    ax_summary2.plot(xs, summary["평균점수"], color="#0F172A", linewidth=2.4, marker="o", markersize=7)
    for i, label in enumerate(RISK_LABELS):
        ax_summary.text(i, summary.loc[label, "시설수"] + 24, f"{summary.loc[label, '시설수']:,}개", ha="center", va="bottom", fontsize=12, fontweight="bold")
        ax_summary2.text(i, summary.loc[label, "평균점수"] + 0.8, f"{summary.loc[label, '평균점수']:.1f}점", ha="center", va="bottom", fontsize=11, fontweight="bold", color="#0F172A")
    ax_summary.set_xticks(xs, RISK_LABELS, fontsize=12, fontweight="bold")
    ax_summary.set_title("군집 규모와 평균 점수", loc="left", fontsize=15, fontweight="bold", pad=12)
    ax_summary.set_ylabel("시설수", color="#475569")
    ax_summary2.set_ylabel("평균 점수", color="#475569")
    ax_summary.grid(axis="y", color="#E2E8F0")
    ax_summary.set_axisbelow(True)
    ax_summary.spines[["top", "right", "left"]].set_visible(False)
    ax_summary2.spines[["top", "left"]].set_visible(False)

    # Boxplot
    box_data = [df.loc[df["위험군"].eq(label), "최종_화재위험점수"] for label in RISK_LABELS]
    bp = ax_box.boxplot(box_data, patch_artist=True, labels=RISK_LABELS, widths=0.55, showfliers=False)
    for patch, label in zip(bp["boxes"], RISK_LABELS):
        patch.set_facecolor(COLORS[label])
        patch.set_alpha(0.82)
        patch.set_edgecolor("white")
    for element in ["whiskers", "caps", "medians"]:
        for item in bp[element]:
            item.set_color("#334155")
            item.set_linewidth(1.3)
    ax_box.set_title("위험점수 분포", loc="left", fontsize=15, fontweight="bold", pad=12)
    ax_box.set_ylabel("최종 화재위험점수", color="#475569")
    ax_box.grid(axis="y", color="#E2E8F0")
    ax_box.spines[["top", "right"]].set_visible(False)

    # Heatmap profile
    im = ax_heat.imshow(profile.to_numpy(), cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    ax_heat.set_xticks(np.arange(len(FEATURES)), [FEATURE_LABELS[f] for f in FEATURES], fontsize=9)
    ax_heat.set_yticks(np.arange(len(RISK_LABELS)), RISK_LABELS, fontsize=11, fontweight="bold")
    ax_heat.set_title("군집별 변수 프로파일", loc="left", fontsize=15, fontweight="bold", pad=12)
    for y in range(profile.shape[0]):
        for x in range(profile.shape[1]):
            value = profile.iloc[y, x]
            ax_heat.text(x, y, f"{value:.2f}", ha="center", va="center", fontsize=8.5, fontweight="bold", color="white" if value > 0.52 else "#111827")
    for spine in ax_heat.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.04, pad=0.02)
    cbar.set_label("정규화 평균", color="#475569")

    # Gu high-risk share
    colors_gu = ["#F59E0B" if v < high_share.median() else "#DC2626" for v in high_share]
    ax_gu.barh(high_share.index, high_share.values, color=colors_gu, edgecolor="white", linewidth=1.0)
    for idx, value in enumerate(high_share.values):
        ax_gu.text(value + 0.8, idx, f"{value:.1f}%", va="center", fontsize=9.5, fontweight="bold", color="#0F172A")
    ax_gu.set_title("구별 고위험군 비율", loc="left", fontsize=15, fontweight="bold", pad=12)
    ax_gu.set_xlabel("고위험군 비율 (%)", color="#475569")
    ax_gu.grid(axis="x", color="#E2E8F0")
    ax_gu.set_axisbelow(True)
    ax_gu.spines[["top", "right", "left"]].set_visible(False)

    # Spatial scatter
    for label in RISK_LABELS:
        sub = df[df["위험군"].eq(label)]
        ax_map.scatter(
            sub["경도"],
            sub["위도"],
            s=14 if label != "고위험군" else 22,
            c=COLORS[label],
            alpha=0.42 if label != "고위험군" else 0.75,
            label=f"{label} ({len(sub):,})",
            linewidths=0,
        )
    ax_map.set_title("공간 분포", loc="left", fontsize=15, fontweight="bold", pad=12)
    ax_map.set_xlabel("경도", color="#475569")
    ax_map.set_ylabel("위도", color="#475569")
    ax_map.grid(color="#E2E8F0")
    ax_map.legend(frameon=False, loc="lower left", fontsize=9.5)
    ax_map.spines[["top", "right"]].set_visible(False)

    # Top 10
    ax_top.barh(top10["숙소명"].str.slice(0, 18), top10["최종_화재위험점수"], color=top10["위험군"].map(COLORS), edgecolor="white", linewidth=1.0)
    for idx, (_, row) in enumerate(top10.iterrows()):
        ax_top.text(row["최종_화재위험점수"] + 0.4, idx, f"{row['최종_화재위험점수']:.1f} | {row['구']}", va="center", fontsize=9.5, fontweight="bold", color="#0F172A")
    ax_top.set_title("최고 위험 시설 TOP 10", loc="left", fontsize=15, fontweight="bold", pad=12)
    ax_top.set_xlabel("최종 화재위험점수", color="#475569")
    ax_top.grid(axis="x", color="#E2E8F0")
    ax_top.spines[["top", "right", "left"]].set_visible(False)
    ax_top.tick_params(axis="y", labelsize=9)

    fig.suptitle("최종 화재위험 군집화 대시보드", x=0.055, y=0.982, ha="left", fontsize=22, fontweight="bold", color="#0F172A")
    fig.text(
        0.055,
        0.946,
        "AHP 가중치 기반 최종 화재위험점수 + KMeans(K=3) | 군집명은 평균 위험점수 기준으로 재정렬",
        ha="left",
        fontsize=12,
        color="#475569",
    )
    fig.text(
        0.055,
        0.032,
        "변수 프로파일은 Min-Max 정규화 평균입니다. 색상: 초록=저위험군, 주황=중위험군, 빨강=고위험군.",
        ha="left",
        fontsize=11,
        color="#64748B",
    )
    fig.subplots_adjust(top=0.86, bottom=0.08, left=0.08, right=0.97)
    fig.savefig(OUT, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"saved={OUT}")
    print(summary.to_string())


if __name__ == "__main__":
    main()
