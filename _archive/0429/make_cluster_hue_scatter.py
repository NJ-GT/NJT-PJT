from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
DATA_DIR = BASE / "0424" / "data" / "cluster3_spatial_pipeline_fire_count_150m_0428"
OUT_DIR = BASE / "0429"


def set_korean_font() -> None:
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.family"] = "Malgun Gothic"


def read_data() -> pd.DataFrame:
    csv_files = sorted(DATA_DIR.glob("*.csv"), key=lambda p: p.stat().st_size, reverse=True)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {DATA_DIR}")
    df = pd.read_csv(csv_files[0], encoding="utf-8-sig")
    needed = ["경도", "위도", "cluster"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")
    return df.dropna(subset=needed).copy()


def main() -> None:
    set_korean_font()
    df = read_data()

    colors = {
        0: "#66c2a5",
        1: "#fc8d62",
        2: "#8da0cb",
    }

    out = OUT_DIR / "숙박시설_클러스터_hue_공간산점도_0429.png"
    fig, ax = plt.subplots(figsize=(10.8, 8.4), dpi=180)

    for cluster in sorted(df["cluster"].unique()):
        part = df[df["cluster"] == cluster]
        ax.scatter(
            part["경도"],
            part["위도"],
            s=17,
            alpha=0.78,
            linewidths=0,
            color=colors.get(cluster, "#999999"),
            label=f"Cluster {cluster} (n={len(part):,})",
        )

    ax.set_title("숙박시설 클러스터별 공간 산점도", fontsize=18, weight="bold", pad=18)
    ax.set_xlabel("경도", fontsize=11)
    ax.set_ylabel("위도", fontsize=11)
    ax.grid(True, color="#e7ebf0", linewidth=0.8)
    ax.set_facecolor("#fbfcfe")
    ax.legend(
        title="클러스터",
        loc="upper right",
        frameon=True,
        framealpha=0.95,
        facecolor="white",
        edgecolor="#d7dde5",
        fontsize=10,
        title_fontsize=10,
    )

    fig.patch.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
