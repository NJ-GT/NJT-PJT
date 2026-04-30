from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


BASE = Path(__file__).resolve().parents[1]
DATA_DIR = BASE / "0424" / "data" / "cluster3_spatial_pipeline_fire_count_150m_0428"
OUT_DIR = BASE / "0429"


def set_korean_font() -> None:
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.family"] = "Malgun Gothic"


def read_data() -> pd.DataFrame:
    csv_files = sorted(DATA_DIR.glob("*.csv"), key=lambda p: p.stat().st_size, reverse=True)
    df = pd.read_csv(csv_files[0], encoding="utf-8-sig")
    return df.dropna(subset=["경도", "위도", "cluster"]).copy()


def main() -> None:
    set_korean_font()
    df = read_data()
    target = df[df["cluster"].isin([0, 2])].copy()

    out = OUT_DIR / "클러스터0_2_겹침_밀도등고선_0429.png"
    fig, ax = plt.subplots(figsize=(11, 8.5), dpi=180)

    palette = {0: "#1b9e77", 2: "#386cb0"}
    labels = {0: "Cluster 0", 2: "Cluster 2"}

    for cluster in [0, 2]:
        part = target[target["cluster"] == cluster]
        ax.scatter(
            part["경도"],
            part["위도"],
            s=9,
            alpha=0.18,
            color=palette[cluster],
            linewidths=0,
            label=f"{labels[cluster]} 점 (n={len(part):,})",
        )

    for cluster in [0, 2]:
        part = target[target["cluster"] == cluster]
        sns.kdeplot(
            data=part,
            x="경도",
            y="위도",
            levels=7,
            color=palette[cluster],
            linewidths=1.8,
            alpha=0.92,
            ax=ax,
            label=f"{labels[cluster]} 밀도",
        )

    ax.set_title("Cluster 0 · 2 공간 겹침 비교", fontsize=18, weight="bold", pad=18)
    ax.set_xlabel("경도", fontsize=11)
    ax.set_ylabel("위도", fontsize=11)
    ax.grid(True, color="#e7ebf0", linewidth=0.8)
    ax.set_facecolor("#fbfcfe")
    ax.legend(
        loc="upper right",
        frameon=True,
        framealpha=0.96,
        facecolor="white",
        edgecolor="#d7dde5",
        fontsize=9.5,
    )

    fig.patch.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
