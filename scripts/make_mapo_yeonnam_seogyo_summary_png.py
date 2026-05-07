# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "0430" / "최종테이블0429.csv"
OUT_PATH = ROOT / "0430" / "마포구_연남동_서교동_위험특성비교.png"

TARGET_DONGS = ["연남동", "서교동"]
RISK_ORDER = ["저위험군", "중위험군", "고위험군"]
RISK_COLORS = {"저위험군": "#60A5FA", "중위험군": "#FBBF24", "고위험군": "#EF4444"}
MAIN_COLORS = {"연남동": "#2563EB", "서교동": "#F97316"}
FEATURES = [
    "구조노후도",
    "단속위험도",
    "도로폭위험도",
    "최근접_소화용수_거리등급",
    "소방위험도_점수",
    "집중도",
    "주변건물수",
]
FEATURE_LABELS = {
    "구조노후도": "구조\n노후도",
    "단속위험도": "단속\n위험도",
    "도로폭위험도": "도로폭\n위험도",
    "최근접_소화용수_거리등급": "소화용수\n거리등급",
    "소방위험도_점수": "소방\n위험도",
    "집중도": "집중도",
    "주변건물수": "주변\n건물수",
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
    subset = df[df["구"].eq("마포구") & df["동"].isin(TARGET_DONGS)].copy()

    numeric_cols = ["최종위험점수_new", *FEATURES, "승인연도", "연면적", "총층수"]
    for col in numeric_cols:
        subset[col] = pd.to_numeric(subset[col], errors="coerce")

    score_summary = subset.groupby("동")["최종위험점수_new"].agg(["count", "mean", "median", "max"]).reindex(TARGET_DONGS)
    risk_counts = pd.crosstab(subset["동"], subset["cluster_label"]).reindex(index=TARGET_DONGS, columns=RISK_ORDER, fill_value=0)
    mean_features = subset.groupby("동")[FEATURES].mean().reindex(TARGET_DONGS)

    scaler = MinMaxScaler()
    scaled = pd.DataFrame(scaler.fit_transform(mean_features.T), index=FEATURES, columns=TARGET_DONGS)
    # Scaling across the two dongs makes relative differences explicit for the heatmap.

    fig = plt.figure(figsize=(17.2, 10.4), dpi=180)
    fig.patch.set_facecolor("#f6f8fb")
    gs = fig.add_gridspec(3, 4, height_ratios=[0.52, 1.18, 1.36], hspace=0.48, wspace=0.38)

    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis("off")
    ax_title.text(0.01, 0.86, "마포구 연남동 · 서교동 위험 특성 비교", fontsize=25, fontweight="bold", color="#101828", va="top")
    ax_title.text(
        0.01,
        0.43,
        "최종테이블0429 기준 | 위험군 구성, 최종위험점수, 주요 위험 변수 평균 비교",
        fontsize=12.4,
        color="#667085",
        va="top",
    )

    key_text = (
        f"연남동 평균위험도 {score_summary.loc['연남동', 'mean']:.2f}점, "
        f"고위험군 {int(risk_counts.loc['연남동', '고위험군'])}개\n"
        f"서교동 평균위험도 {score_summary.loc['서교동', 'mean']:.2f}점, "
        f"고위험군 {int(risk_counts.loc['서교동', '고위험군'])}개"
    )
    ax_title.text(
        0.68,
        0.78,
        key_text,
        fontsize=12.3,
        color="#7a2e0e",
        va="top",
        bbox=dict(boxstyle="round,pad=0.55,rounding_size=0.12", facecolor="#fff4e5", edgecolor="#ffd8a8"),
    )

    ax_counts = fig.add_subplot(gs[1, 0])
    bottom = np.zeros(len(TARGET_DONGS))
    x = np.arange(len(TARGET_DONGS))
    for risk in RISK_ORDER:
        vals = risk_counts[risk].to_numpy()
        ax_counts.bar(x, vals, bottom=bottom, color=RISK_COLORS[risk], edgecolor="white", linewidth=1.2, label=risk)
        for xi, val, bot in zip(x, vals, bottom):
            if val > 18:
                ax_counts.text(xi, bot + val / 2, f"{int(val)}", ha="center", va="center", fontsize=10.5, fontweight="bold", color="#111827")
        bottom += vals
    ax_counts.set_xticks(x, TARGET_DONGS)
    ax_counts.set_title("위험군 구성", fontsize=15, fontweight="bold")
    ax_counts.set_ylabel("시설 수")
    ax_counts.legend(loc="upper left", frameon=True, fontsize=9)
    ax_counts.set_facecolor("#ffffff")
    ax_counts.grid(axis="y", alpha=0.25)

    ax_score = fig.add_subplot(gs[1, 1])
    bars = ax_score.bar(TARGET_DONGS, score_summary["mean"], color=[MAIN_COLORS[d] for d in TARGET_DONGS], edgecolor="white", linewidth=1.2)
    for bar, dong in zip(bars, TARGET_DONGS):
        ax_score.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.7,
            f"{score_summary.loc[dong, 'mean']:.2f}",
            ha="center",
            fontsize=12,
            fontweight="bold",
            color="#101828",
        )
    ax_score.set_ylim(0, max(score_summary["mean"]) * 1.25)
    ax_score.set_title("평균 최종위험점수", fontsize=15, fontweight="bold")
    ax_score.set_ylabel("점수")
    ax_score.set_facecolor("#ffffff")
    ax_score.grid(axis="y", alpha=0.25)

    ax_box = fig.add_subplot(gs[1, 2:])
    sns.boxplot(
        data=subset,
        x="동",
        y="최종위험점수_new",
        order=TARGET_DONGS,
        hue="동",
        palette=MAIN_COLORS,
        width=0.5,
        linewidth=1.3,
        fliersize=2,
        legend=False,
        ax=ax_box,
    )
    sns.stripplot(
        data=subset.sample(min(len(subset), 450), random_state=42),
        x="동",
        y="최종위험점수_new",
        order=TARGET_DONGS,
        color="#111827",
        alpha=0.13,
        size=2.2,
        jitter=0.2,
        ax=ax_box,
    )
    ax_box.set_title("최종위험점수 분포", fontsize=15, fontweight="bold")
    ax_box.set_xlabel("")
    ax_box.set_ylabel("점수")
    ax_box.set_facecolor("#ffffff")
    ax_box.grid(axis="y", alpha=0.25)

    ax_heat = fig.add_subplot(gs[2, 0:2])
    heat = scaled.rename(index=FEATURE_LABELS)
    sns.heatmap(
        heat,
        cmap=sns.color_palette(["#E8F3EF", "#8BD3C7", "#276FBF"], as_cmap=True),
        annot=mean_features.T.rename(index=FEATURE_LABELS),
        fmt=".2f",
        linewidths=1.0,
        linecolor="#ffffff",
        cbar_kws={"label": "두 동 간 상대 수준"},
        ax=ax_heat,
    )
    ax_heat.set_title("주요 변수 평균 비교", fontsize=15, fontweight="bold", pad=12)
    ax_heat.set_xlabel("")
    ax_heat.set_ylabel("")

    ax_delta = fig.add_subplot(gs[2, 2:])
    delta = (mean_features.loc["연남동"] - mean_features.loc["서교동"]).sort_values()
    colors = ["#F97316" if v < 0 else "#2563EB" for v in delta]
    ax_delta.barh([FEATURE_LABELS[i].replace("\n", " ") for i in delta.index], delta.values, color=colors, edgecolor="white")
    ax_delta.axvline(0, color="#344054", linewidth=1.1)
    for y, (idx, val) in enumerate(delta.items()):
        ax_delta.text(val + (0.015 if val >= 0 else -0.015), y, f"{val:+.2f}", va="center", ha="left" if val >= 0 else "right", fontsize=10.5)
    ax_delta.set_title("연남동 - 서교동 평균 차이", fontsize=15, fontweight="bold", pad=12)
    ax_delta.set_xlabel("양수면 연남동이 더 높음")
    ax_delta.set_facecolor("#ffffff")
    ax_delta.grid(axis="x", alpha=0.25)

    fig.text(
        0.018,
        0.018,
        "결론: 연남동은 평균 위험점수와 고위험군 수가 더 높고, 특히 소방위험도·집중도·주변건물수·구조노후도가 서교동보다 높다. 서교동은 도로폭위험도와 총층수/연면적 측면이 상대적으로 높다.",
        fontsize=11.5,
        color="#344054",
    )
    fig.savefig(OUT_PATH, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(OUT_PATH)


if __name__ == "__main__":
    main()
