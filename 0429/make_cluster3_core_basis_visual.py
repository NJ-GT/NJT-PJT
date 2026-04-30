from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


BASE = Path(__file__).resolve().parents[1]
DATA_DIR = BASE / "0424" / "data" / "cluster3_spatial_pipeline_fire_count_150m_0428"
OUT_DIR = BASE / "0429"


FEATURES = [
    "구조노후도",
    "단속위험도",
    "도로폭위험도",
    "집중도",
    "주변건물수",
    "최근접_소화용수_거리등급",
    "소방위험도_점수",
    "최종_화재위험점수",
]

FEATURE_LABELS = {
    "구조노후도": "구조\n노후도",
    "단속위험도": "단속\n위험도",
    "도로폭위험도": "도로폭\n위험도",
    "집중도": "시설\n집중도",
    "주변건물수": "주변\n건물수",
    "최근접_소화용수_거리등급": "소화용수\n거리등급",
    "소방위험도_점수": "소방\n위험도",
    "최종_화재위험점수": "최종\n위험점수",
}

CLUSTER_MEMO = {
    0: "밀집도는 높지만\n소화용수 접근성 양호",
    1: "저밀도이나\n소화용수 접근 취약",
    2: "밀집·도로폭·최종위험\n동시 고위험",
}


def set_korean_font() -> None:
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.family"] = "Malgun Gothic"


def read_data() -> pd.DataFrame:
    csv_files = sorted(DATA_DIR.glob("*.csv"), key=lambda p: p.stat().st_size, reverse=True)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {DATA_DIR}")
    df = pd.read_csv(csv_files[0], encoding="utf-8-sig")
    needed = ["cluster", *FEATURES]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")
    return df[needed].dropna().copy()


def minmax_by_feature(cluster_means: pd.DataFrame) -> pd.DataFrame:
    scored = cluster_means.copy()
    for col in scored.columns:
        mean = scored[col].mean()
        value_range = scored[col].max() - scored[col].min()
        if np.isclose(mean, 0) or value_range / abs(mean) < 0.05:
            scored[col] = 0
        else:
            pct_diff = (scored[col] - mean) / abs(mean) * 100
            scored[col] = pct_diff.clip(-80, 80)
    return scored


def main() -> None:
    set_korean_font()
    df = read_data()

    counts = df.groupby("cluster").size()
    means = df.groupby("cluster")[FEATURES].mean().sort_index()
    relative = minmax_by_feature(means)
    relative.columns = [FEATURE_LABELS[c] for c in relative.columns]
    relative.index = [f"Cluster {idx}\n(n={counts.loc[idx]:,})" for idx in relative.index]

    out_png = OUT_DIR / "군집3개_핵심근거_프로파일히트맵_0429.png"
    out_csv = OUT_DIR / "군집3개_핵심근거_변수평균표_0429.csv"
    means.round(4).to_csv(out_csv, encoding="utf-8-sig")

    fig = plt.figure(figsize=(14.5, 8.2), dpi=180)
    gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[4.8, 1.7], wspace=0.07)
    ax = fig.add_subplot(gs[0, 0])
    ax_note = fig.add_subplot(gs[0, 1])

    sns.heatmap(
        relative,
        ax=ax,
        cmap=sns.diverging_palette(220, 20, as_cmap=True),
        center=0,
        vmin=-80,
        vmax=80,
        annot=means.rename(columns=FEATURE_LABELS).round(2),
        fmt=".2f",
        linewidths=1.2,
        linecolor="white",
        cbar_kws={"label": "전체 평균 대비 차이(%) · 차이 작으면 0"},
        annot_kws={"fontsize": 10, "weight": "bold"},
    )

    ax.set_title("K=3 군집화 핵심 근거: 변수 평균 프로파일", fontsize=19, weight="bold", pad=18)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelsize=10, rotation=0)
    ax.tick_params(axis="y", labelsize=11, rotation=0)

    ax_note.axis("off")
    ax_note.text(
        0,
        0.98,
        "군집 해석 요약",
        fontsize=15,
        weight="bold",
        va="top",
        transform=ax_note.transAxes,
    )

    y = 0.82
    colors = {0: "#66c2a5", 1: "#fc8d62", 2: "#8da0cb"}
    for cluster in sorted(CLUSTER_MEMO):
        ax_note.add_patch(
            plt.Rectangle(
                (0, y - 0.035),
                0.055,
                0.055,
                color=colors[cluster],
                transform=ax_note.transAxes,
                clip_on=False,
            )
        )
        ax_note.text(
            0.075,
            y,
            f"Cluster {cluster}",
            fontsize=12,
            weight="bold",
            va="center",
            transform=ax_note.transAxes,
        )
        ax_note.text(
            0.075,
            y - 0.08,
            CLUSTER_MEMO[cluster],
            fontsize=11,
            color="#283747",
            va="top",
            linespacing=1.35,
            transform=ax_note.transAxes,
        )
        y -= 0.27

    ax_note.text(
        0,
        0.04,
        "숫자: 각 군집의 원자료 평균\n색: 전체 평균 대비 차이\n구조노후도처럼 차이가 미미하면 중립색 처리",
        fontsize=9.5,
        color="#5f6b7a",
        va="bottom",
        linespacing=1.35,
        transform=ax_note.transAxes,
    )

    fig.patch.set_facecolor("white")
    fig.savefig(out_png, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(out_png)
    print(out_csv)


if __name__ == "__main__":
    main()
