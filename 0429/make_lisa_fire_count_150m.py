from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from esda.moran import Moran_Local
from libpysal.weights import KNN


ROOT = Path(__file__).resolve().parents[1]
K2_DIR = ROOT / "0429" / "cluster2_spatial_pipeline_fire_count_150m_0429"
OUT_DIR = K2_DIR / "lisa_fire_count_150m"
TARGET = "fire_count_150m"
KNN_K = 6
PERMUTATIONS = 999


def set_korean_font() -> None:
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.family"] = "Malgun Gothic"


def read_data() -> pd.DataFrame:
    csv_files = sorted(K2_DIR.glob("*cluster_k2.csv"), key=lambda p: p.stat().st_size, reverse=True)
    if not csv_files:
        raise FileNotFoundError(K2_DIR)
    df = pd.read_csv(csv_files[0], encoding="utf-8-sig")
    needed = ["구", "동", "숙소명", "경도", "위도", "x_5181", "y_5181", TARGET, "cluster_k2"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")
    for col in ["경도", "위도", "x_5181", "y_5181", TARGET, "cluster_k2"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["경도", "위도", "x_5181", "y_5181", TARGET]).reset_index(drop=True)


def classify_lisa(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    coords = df[["x_5181", "y_5181"]].to_numpy(dtype=float)
    values = df[TARGET].to_numpy(dtype=float)
    w = KNN.from_array(coords, k=min(KNN_K, len(df) - 1))
    w.transform = "r"
    lisa = Moran_Local(values, w, permutations=PERMUTATIONS, seed=42)

    out = df.copy()
    out["lisa_I"] = lisa.Is
    out["lisa_p"] = lisa.p_sim
    out["lisa_q"] = lisa.q
    out["lisa_significant"] = out["lisa_p"] < 0.05

    labels = {1: "High-High", 2: "Low-High", 3: "Low-Low", 4: "High-Low"}
    out["lisa_type"] = "Not significant"
    sig = out["lisa_significant"]
    out.loc[sig, "lisa_type"] = out.loc[sig, "lisa_q"].map(labels).fillna("Not significant")

    global_info = {
        "knn_k": KNN_K,
        "permutations": PERMUTATIONS,
        "n": int(len(out)),
        "target_mean": float(out[TARGET].mean()),
        "target_median": float(out[TARGET].median()),
        "high_high_n": int((out["lisa_type"] == "High-High").sum()),
        "low_low_n": int((out["lisa_type"] == "Low-Low").sum()),
        "high_low_n": int((out["lisa_type"] == "High-Low").sum()),
        "low_high_n": int((out["lisa_type"] == "Low-High").sum()),
        "not_significant_n": int((out["lisa_type"] == "Not significant").sum()),
    }
    return out, global_info


def plot_lisa_map(df: pd.DataFrame, out_path: Path) -> None:
    colors = {
        "High-High": "#d73027",
        "Low-Low": "#4575b4",
        "High-Low": "#fc8d59",
        "Low-High": "#91bfdb",
        "Not significant": "#c8cdd2",
    }
    sizes = {
        "High-High": 24,
        "Low-Low": 16,
        "High-Low": 22,
        "Low-High": 18,
        "Not significant": 8,
    }
    order = ["Not significant", "Low-Low", "Low-High", "High-Low", "High-High"]

    fig, ax = plt.subplots(figsize=(11.5, 8.8), dpi=180)
    for typ in order:
        part = df[df["lisa_type"] == typ]
        if part.empty:
            continue
        ax.scatter(
            part["경도"],
            part["위도"],
            s=sizes[typ],
            color=colors[typ],
            alpha=0.82 if typ != "Not significant" else 0.28,
            linewidths=0,
            label=f"{typ} (n={len(part):,})",
        )
    ax.set_title("LISA 군집도: 150m 화재건수 기준", fontsize=19, weight="bold", pad=18)
    ax.set_xlabel("경도")
    ax.set_ylabel("위도")
    ax.grid(True, color="#e7ebf0", linewidth=0.8)
    ax.set_facecolor("#fbfcfe")
    ax.legend(
        title="LISA 유형",
        loc="upper right",
        frameon=True,
        facecolor="white",
        edgecolor="#d7dde5",
        fontsize=9,
        title_fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_high_high_bar(df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    hh = df[df["lisa_type"] == "High-High"].copy()
    top = (
        hh.groupby(["구", "동"])
        .agg(
            high_high_시설수=("숙소명", "count"),
            평균_150m화재건수=(TARGET, "mean"),
            평균_lisa_I=("lisa_I", "mean"),
            고위험군_cluster1_비율=("cluster_k2", lambda s: float((s == 1).mean())),
        )
        .reset_index()
        .sort_values(["high_high_시설수", "평균_150m화재건수"], ascending=False)
    )

    plot = top.head(12).copy()
    plot["지역"] = plot["구"] + " " + plot["동"]
    fig, ax = plt.subplots(figsize=(11.5, 7.4), dpi=180)
    sns.barplot(data=plot, y="지역", x="high_high_시설수", palette="Reds_r", ax=ax)
    for i, row in plot.reset_index(drop=True).iterrows():
        ax.text(
            row["high_high_시설수"] + 0.5,
            i,
            f"평균 {row['평균_150m화재건수']:.1f}건",
            va="center",
            fontsize=9.5,
            color="#333333",
        )
    ax.set_title("LISA High-High 상위 법정동", fontsize=18, weight="bold", pad=16)
    ax.set_xlabel("High-High 시설 수")
    ax.set_ylabel("")
    ax.grid(axis="x", color="#e7ebf0", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return top


def main() -> None:
    set_korean_font()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = read_data()
    lisa_df, global_info = classify_lisa(df)
    lisa_df.to_csv(OUT_DIR / "lisa_fire_count_150m_results.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([global_info]).to_csv(OUT_DIR / "lisa_fire_count_150m_summary.csv", index=False, encoding="utf-8-sig")
    plot_lisa_map(lisa_df, OUT_DIR / "lisa_fire_count_150m_map.png")
    top = plot_high_high_bar(lisa_df, OUT_DIR / "lisa_high_high_top_dongs.png")
    top.to_csv(OUT_DIR / "lisa_high_high_top_dongs.csv", index=False, encoding="utf-8-sig")
    print(OUT_DIR)
    print(pd.DataFrame([global_info]).to_string(index=False))
    print(top.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
