# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


BASE = Path(__file__).resolve().parents[1]
SRC = BASE / "0424" / "data" / "분석변수_최종테이블0428.csv"
OUT_CSV = BASE / "0424" / "data" / "최종_화재위험_분석결과_0428_연면적제외.csv"
OUT_PNG = BASE / "0424" / "data" / "군집별_최고위험시설_TOP10_0428_연면적제외.png"

WEIGHTS = {
    "구조노후도": 0.24,
    "단속위험도": 0.16,
    "도로폭위험도": 0.14,
    "최근접_소화용수_거리등급": 0.12,
    "소방위험도_점수": 0.11,
    "집중도": 0.07,
    "주변건물수": 0.05,
    "총층수": 0.02,
}

RISK_LABELS = ["저위험군", "중위험군", "고위험군"]
COLORS = {
    "저위험군": "#15803D",
    "중위험군": "#F59E0B",
    "고위험군": "#DC2626",
}


def trim_name(name: object, max_len: int = 15) -> str:
    text = str(name)
    return text if len(text) <= max_len else text[: max_len - 1] + "…"


def build_result() -> pd.DataFrame:
    df = pd.read_csv(SRC, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()

    features = list(WEIGHTS.keys())
    x = df[features].apply(pd.to_numeric, errors="coerce").fillna(0)
    scaled = pd.DataFrame(MinMaxScaler().fit_transform(x), columns=features, index=df.index)

    # Renormalize weights after excluding area so the score remains on a 0-100 scale.
    weight_sum = sum(WEIGHTS.values())
    weights = pd.Series({k: v / weight_sum for k, v in WEIGHTS.items()})
    df["최종_화재위험점수_연면적제외"] = (scaled[features] * weights).sum(axis=1) * 100

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df["cluster_k3_연면적제외"] = kmeans.fit_predict(scaled[features])

    order = (
        df.groupby("cluster_k3_연면적제외")["최종_화재위험점수_연면적제외"]
        .mean()
        .sort_values()
        .index
        .tolist()
    )
    label_map = {cluster: label for cluster, label in zip(order, RISK_LABELS)}
    df["위험군_연면적제외"] = df["cluster_k3_연면적제외"].map(label_map)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    return df


def main() -> None:
    df = build_result()
    top_by_group = {
        label: (
            df[df["위험군_연면적제외"].eq(label)]
            .sort_values("최종_화재위험점수_연면적제외", ascending=False)
            .head(10)
            .sort_values("최종_화재위험점수_연면적제외", ascending=True)
        )
        for label in RISK_LABELS
    }

    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 3, figsize=(18, 8.4), facecolor="#F8FAFC")
    for ax, label in zip(axes, RISK_LABELS):
        top = top_by_group[label]
        labels = [trim_name(v) for v in top["숙소명"]]
        scores = top["최종_화재위험점수_연면적제외"]
        bars = ax.barh(
            labels,
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
                f"{row['최종_화재위험점수_연면적제외']:.1f} | {row['구']}",
                va="center",
                ha="left",
                fontsize=9.5,
                color="#0F172A",
                fontweight="bold",
            )

        mean_score = df.loc[df["위험군_연면적제외"].eq(label), "최종_화재위험점수_연면적제외"].mean()
        count = int(df["위험군_연면적제외"].eq(label).sum())
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
        ax.set_xlim(0, max(72, float(scores.max()) + 7))

    fig.suptitle(
        "군집별 최고위험시설 TOP 10 - 연면적 제외",
        x=0.055,
        y=0.965,
        ha="left",
        fontsize=24,
        fontweight="bold",
        color="#0F172A",
    )
    fig.text(
        0.055,
        0.915,
        "연면적을 제외하고 가중치를 재정규화한 뒤, KMeans(K=3)를 다시 수행했습니다.",
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
    fig.savefig(OUT_PNG, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"saved_png={OUT_PNG}")
    print(f"saved_csv={OUT_CSV}")
    print(
        df.groupby("위험군_연면적제외")["최종_화재위험점수_연면적제외"]
        .agg(["count", "mean", "max"])
        .reindex(RISK_LABELS)
        .to_string()
    )


if __name__ == "__main__":
    main()
