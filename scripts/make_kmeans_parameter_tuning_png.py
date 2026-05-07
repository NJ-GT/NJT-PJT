# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import MinMaxScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "0430" / "최종테이블0429.csv"
OUT_PATH = ROOT / "0430" / "kmeans_군집파라미터_튜닝과정.png"


FEATURES = [
    "구조노후도",
    "단속위험도",
    "도로폭위험도",
    "최근접_소화용수_거리등급",
    "소방위험도_점수",
    "승인연도",
    "연면적",
    "집중도",
    "주변건물수",
    "총층수",
]


def set_korean_font() -> None:
    candidates = [
        r"C:\Windows\Fonts\malgun.ttf",
        r"C:\Windows\Fonts\NanumGothic.ttf",
        r"C:\Windows\Fonts\NotoSansKR-Regular.otf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            fm.fontManager.addfont(candidate)
            plt.rcParams["font.family"] = fm.FontProperties(fname=candidate).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False


def kmeans_metrics(x: np.ndarray, k_range: range) -> pd.DataFrame:
    rows: list[dict] = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10, init="k-means++")
        labels = km.fit_predict(x)
        rows.append(
            {
                "k": k,
                "inertia": float(km.inertia_),
                "silhouette": float(silhouette_score(x, labels)),
                "calinski_harabasz": float(calinski_harabasz_score(x, labels)),
                "davies_bouldin": float(davies_bouldin_score(x, labels)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    set_korean_font()
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()

    score_x = (
        df[["최종위험점수_new"]]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .to_numpy()
    )
    feature_x = df[FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0)
    feature_x = MinMaxScaler().fit_transform(feature_x)
    stored_labels = df["cluster"].astype(int).to_numpy()

    score_metrics = kmeans_metrics(score_x, range(2, 9))
    feature_metrics = kmeans_metrics(feature_x, range(2, 9))

    score_k3_labels = KMeans(
        n_clusters=3, random_state=42, n_init=10, init="k-means++"
    ).fit_predict(score_x)
    feature_k3_labels = KMeans(
        n_clusters=3, random_state=42, n_init=10, init="k-means++"
    ).fit_predict(feature_x)
    ari_score = adjusted_rand_score(stored_labels, score_k3_labels)
    ari_feature = adjusted_rand_score(stored_labels, feature_k3_labels)

    stored_score_metrics = {
        "silhouette": silhouette_score(score_x, stored_labels),
        "calinski_harabasz": calinski_harabasz_score(score_x, stored_labels),
        "davies_bouldin": davies_bouldin_score(score_x, stored_labels),
    }

    summary = (
        df.groupby(["cluster", "cluster_label"])["최종위험점수_new"]
        .agg(["count", "min", "mean", "max"])
        .reset_index()
        .sort_values("mean")
    )

    fig = plt.figure(figsize=(18, 10.2), dpi=180)
    fig.patch.set_facecolor("#f5f7fb")
    gs = fig.add_gridspec(
        3, 4, height_ratios=[0.92, 1.35, 1.15], hspace=0.42, wspace=0.28
    )

    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis("off")
    ax_title.text(
        0.012,
        0.88,
        "K-Means 군집 파라미터 및 튜닝 과정",
        fontsize=25,
        fontweight="bold",
        color="#101828",
        va="top",
    )
    ax_title.text(
        0.012,
        0.58,
        "최종 파일: 0430/최종테이블0429.csv  |  기준 점수: 최종위험점수_new  |  군집명: 평균 점수 기준 저위험군·중위험군·고위험군",
        fontsize=12.5,
        color="#475467",
        va="top",
    )

    param_lines = [
        "사용 알고리즘: sklearn.cluster.KMeans",
        "최종 군집 수: n_clusters=3",
        "초기화: init='k-means++'",
        "반복 초기화: n_init=10",
        "재현성: random_state=42",
        "전처리 후보: 최종위험점수 1D 또는 10개 변수 MinMax 정규화",
    ]
    y = 0.28
    for line in param_lines:
        ax_title.text(0.018, y, f"• {line}", fontsize=12, color="#1d2939", va="top")
        y -= 0.12

    ax_title.text(
        0.62,
        0.30,
        f"검증 메모\n저장 cluster vs 점수 1D KMeans ARI = {ari_score:.3f}\n저장 cluster vs 10변수 KMeans ARI = {ari_feature:.3f}\n=> 최종 cluster 생성 스크립트/입력공간 확인 필요",
        fontsize=12,
        color="#7a2e0e",
        bbox=dict(
            boxstyle="round,pad=0.55,rounding_size=0.12",
            facecolor="#fff4e5",
            edgecolor="#ffd8a8",
        ),
        va="top",
    )

    def lineplot(ax, data, ycol, title, ylabel, better_note, color, mark_k3=True):
        ax.plot(data["k"], data[ycol], marker="o", linewidth=2.4, color=color)
        if mark_k3:
            k3 = data.loc[data["k"].eq(3), ycol].iloc[0]
            ax.scatter([3], [k3], s=120, color="#101828", zorder=5)
            ax.annotate(
                f"K=3\n{k3:,.3f}",
                xy=(3, k3),
                xytext=(3.25, k3),
                fontsize=10,
                color="#101828",
                arrowprops=dict(arrowstyle="-", color="#667085"),
            )
        ax.set_title(title, fontsize=13, fontweight="bold", color="#101828")
        ax.set_xlabel("K 후보")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.26)
        ax.set_facecolor("#ffffff")
        ax.text(
            0.02,
            0.04,
            better_note,
            transform=ax.transAxes,
            fontsize=9.5,
            color="#667085",
        )

    ax1 = fig.add_subplot(gs[1, 0])
    lineplot(
        ax1,
        score_metrics,
        "inertia",
        "Elbow: SSE / Inertia",
        "작을수록 좋음",
        "급격한 감소 후 완만해지는 지점 확인",
        "#2563eb",
    )

    ax2 = fig.add_subplot(gs[1, 1])
    lineplot(
        ax2,
        score_metrics,
        "silhouette",
        "Silhouette Score",
        "클수록 좋음",
        "분리도와 응집도 균형",
        "#059669",
    )

    ax3 = fig.add_subplot(gs[1, 2])
    lineplot(
        ax3,
        score_metrics,
        "calinski_harabasz",
        "Calinski-Harabasz",
        "클수록 좋음",
        "군집 간 분산 / 군집 내 분산",
        "#dc6803",
    )

    ax4 = fig.add_subplot(gs[1, 3])
    lineplot(
        ax4,
        score_metrics,
        "davies_bouldin",
        "Davies-Bouldin",
        "작을수록 좋음",
        "군집 간 겹침이 작을수록 양호",
        "#c11574",
    )

    ax_table = fig.add_subplot(gs[2, 0:2])
    ax_table.axis("off")
    table_df = score_metrics.copy()
    for col in ["inertia", "silhouette", "calinski_harabasz", "davies_bouldin"]:
        table_df[col] = table_df[col].map(lambda v: f"{v:,.3f}")
    cell_text = table_df[
        ["k", "inertia", "silhouette", "calinski_harabasz", "davies_bouldin"]
    ].values
    table = ax_table.table(
        cellText=cell_text,
        colLabels=["K", "Inertia", "Silhouette", "CH", "DB"],
        loc="center",
        cellLoc="center",
        colColours=["#e7eefb"] * 5,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1, 1.45)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#d0d5dd")
        if r == 0:
            cell.set_text_props(weight="bold", color="#101828")
        elif int(table_df.iloc[r - 1]["k"]) == 3:
            cell.set_facecolor("#fff4cc")
    ax_table.set_title(
        "K 후보별 튜닝 지표: 최종위험점수_new 1D 기준",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )

    ax_bar = fig.add_subplot(gs[2, 2])
    colors = ["#60a5fa", "#fbbf24", "#ef4444"]
    ax_bar.bar(
        summary["cluster_label"],
        summary["count"],
        color=colors,
        edgecolor="#ffffff",
        linewidth=1.2,
    )
    for i, row in enumerate(summary.itertuples()):
        ax_bar.text(
            i,
            row.count + 35,
            f"{row.count:,}",
            ha="center",
            fontsize=11,
            fontweight="bold",
            color="#101828",
        )
    ax_bar.set_title("최종 저장 군집 규모", fontsize=13, fontweight="bold")
    ax_bar.set_ylabel("시설 수")
    ax_bar.grid(axis="y", alpha=0.25)
    ax_bar.set_facecolor("#ffffff")

    ax_note = fig.add_subplot(gs[2, 3])
    ax_note.axis("off")
    note = (
        "해석 포인트\n"
        "1. 최종 산출물은 K=3 체계로 저·중·고 위험군을 사용.\n"
        "2. KMeans 기본 파라미터는 n_clusters=3, random_state=42, n_init=10.\n"
        "3. 점수 1개만으로 재군집하면 저장 cluster와 완전 일치하지 않음.\n"
        "4. 따라서 최종 발표에는 'K=3 사용'과 함께 최종 생성 코드의 입력공간을 명시해야 함.\n"
        f"5. 저장 cluster 기준 점수 1D 지표: Sil={stored_score_metrics['silhouette']:.3f}, "
        f"CH={stored_score_metrics['calinski_harabasz']:.1f}, DB={stored_score_metrics['davies_bouldin']:.3f}"
    )
    ax_note.text(
        0,
        1,
        note,
        fontsize=11.4,
        linespacing=1.62,
        color="#1d2939",
        va="top",
        bbox=dict(
            boxstyle="round,pad=0.72,rounding_size=0.12",
            facecolor="#ffffff",
            edgecolor="#d0d5dd",
        ),
    )

    fig.text(
        0.012,
        0.015,
        "주의: 이 PNG는 현재 저장된 최종 CSV와 저장소 내 확인 가능한 KMeans 계열 스크립트를 기준으로 재계산한 근거 자료입니다.",
        fontsize=10.5,
        color="#667085",
    )
    fig.savefig(OUT_PATH, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(OUT_PATH)


if __name__ == "__main__":
    main()
