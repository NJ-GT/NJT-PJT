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
    for font in ["Malgun Gothic", "AppleGothic", "NanumGothic"]:
        plt.rcParams["font.family"] = font
        break


def read_data() -> pd.DataFrame:
    csv_files = sorted(DATA_DIR.glob("*.csv"), key=lambda p: p.stat().st_size, reverse=True)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {DATA_DIR}")
    df = pd.read_csv(csv_files[0], encoding="utf-8-sig")
    needed = ["경도", "위도", "구조노후도", "cluster"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")
    return df.dropna(subset=needed).copy()


def plot_overall(df: pd.DataFrame) -> Path:
    out = OUT_DIR / "구조노후도_공간산점도_0429.png"
    fig, ax = plt.subplots(figsize=(10.5, 8.2), dpi=180)

    sc = ax.scatter(
        df["경도"],
        df["위도"],
        c=df["구조노후도"],
        cmap="magma_r",
        s=16,
        alpha=0.78,
        linewidths=0,
    )

    cbar = fig.colorbar(sc, ax=ax, pad=0.012, shrink=0.86)
    cbar.set_label("구조노후도", fontsize=11)

    ax.set_title("숙박시설별 구조노후도 공간 산점도", fontsize=18, weight="bold", pad=18)
    ax.set_xlabel("경도", fontsize=11)
    ax.set_ylabel("위도", fontsize=11)
    ax.grid(True, color="#e7ebf0", linewidth=0.8)
    ax.set_facecolor("#fbfcfe")
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def plot_by_cluster(df: pd.DataFrame) -> Path:
    out = OUT_DIR / "구조노후도_군집별_공간산점도_0429.png"
    clusters = sorted(df["cluster"].dropna().unique())
    fig, axes = plt.subplots(1, len(clusters), figsize=(15.5, 5.6), dpi=180, sharex=True, sharey=True)
    if len(clusters) == 1:
        axes = [axes]

    vmin = df["구조노후도"].min()
    vmax = df["구조노후도"].max()
    last_sc = None
    for ax, cluster in zip(axes, clusters):
        part = df[df["cluster"] == cluster]
        last_sc = ax.scatter(
            part["경도"],
            part["위도"],
            c=part["구조노후도"],
            cmap="magma_r",
            vmin=vmin,
            vmax=vmax,
            s=14,
            alpha=0.78,
            linewidths=0,
        )
        ax.set_title(f"Cluster {cluster}  n={len(part):,}", fontsize=13, weight="bold")
        ax.set_xlabel("경도", fontsize=10)
        ax.grid(True, color="#e7ebf0", linewidth=0.75)
        ax.set_facecolor("#fbfcfe")

    axes[0].set_ylabel("위도", fontsize=10)
    fig.suptitle("군집별 구조노후도 공간 산점도", fontsize=18, weight="bold", y=1.02)
    cbar = fig.colorbar(last_sc, ax=axes, pad=0.012, shrink=0.78)
    cbar.set_label("구조노후도", fontsize=11)
    fig.patch.set_facecolor("white")
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def main() -> None:
    set_korean_font()
    df = read_data()
    overall = plot_overall(df)
    by_cluster = plot_by_cluster(df)
    print(overall)
    print(by_cluster)


if __name__ == "__main__":
    main()
