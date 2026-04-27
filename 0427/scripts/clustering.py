# -*- coding: utf-8 -*-
import glob
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

sys.stdout.reconfigure(encoding="utf-8")
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

BASE = "c:/Users/USER/Documents/GitHub/기말공모전/NJT-PJT"
SRC = glob.glob(f"{BASE}/0424/*/tables/*AHP3*.csv")[0]
OUT_DIR = f"{BASE}/data"

VARS = ["구조노후도", "단속위험도", "도로폭위험도", "집중도", "주변건물수"]
COLORS = ["#27ae60", "#e67e22", "#c0392b", "#2980b9", "#8e44ad", "#f39c12", "#1abc9c"]

GROUPS = {
    "기존숙박군": "기존숙박군 (숙박업+관광숙박업)",
    "외국인관광도시민박업": "외국인관광도시민박업",
}

df_all = pd.read_csv(SRC, encoding="utf-8-sig")
for v in VARS + ["위험점수_AHP"]:
    df_all[v] = pd.to_numeric(df_all[v], errors="coerce")

results = {}

for group_key, group_label in GROUPS.items():
    print(f"\n{'='*60}")
    print(f"  {group_label}  ({(df_all['업종그룹'] == group_key).sum()}개)")
    print("=" * 60)

    df = df_all[df_all["업종그룹"] == group_key].dropna(subset=VARS).reset_index(drop=True)
    slug = "A_기존숙박군" if group_key == "기존숙박군" else "B_외국인민박"

    X = df[VARS].values
    X_scaled = StandardScaler().fit_transform(X)

    # ── 1. 최적 K 탐색 (엘보우 + 실루엣) ────────────────────────────
    inertias, silhouettes = [], []
    K_range = range(2, 8)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))

    best_k = list(K_range)[int(np.argmax(silhouettes))]
    print(f"  최적 K: {best_k}  (실루엣 {max(silhouettes):.3f})")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"군집 수 탐색 — {group_label}", fontsize=12)
    axes[0].plot(K_range, inertias, "o-", color="#2c7bb6")
    axes[0].set_xlabel("군집 수 (K)")
    axes[0].set_ylabel("Inertia")
    axes[0].set_title("엘보우 차트")
    axes[0].grid(alpha=0.3)
    axes[1].plot(K_range, silhouettes, "o-", color="#d7191c")
    axes[1].axvline(best_k, linestyle="--", color="gray", alpha=0.7, label=f"최적 K={best_k}")
    axes[1].set_xlabel("군집 수 (K)")
    axes[1].set_ylabel("실루엣 점수")
    axes[1].set_title("실루엣 점수")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/clustering_elbow_{slug}.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── 2. 최적 K로 군집화 ──────────────────────────────────────────
    km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    df["군집"] = km_final.fit_predict(X_scaled)

    # ── 3. 군집별 요약 ──────────────────────────────────────────────
    print(f"\n[군집별 특성 — K={best_k}]")
    summary = df.groupby("군집")[VARS + ["위험점수_AHP"]].mean().round(3)
    summary["시설수"] = df.groupby("군집").size()
    print(summary.to_string())

    # ── 4. 산점도 ───────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"군집 산점도 — {group_label}", fontsize=12)

    for c in range(best_k):
        mask = df["군집"] == c
        cnt = mask.sum()
        avg_risk = df.loc[mask, "위험점수_AHP"].mean()
        axes[0].scatter(
            df.loc[mask, "구조노후도"], df.loc[mask, "단속위험도"],
            c=COLORS[c], s=20, alpha=0.6,
            label=f"군집{c} ({cnt}개, AHP={avg_risk:.0f})",
        )
        axes[1].scatter(
            df.loc[mask, "주변건물수"], df.loc[mask, "집중도"],
            c=COLORS[c], s=20, alpha=0.6,
            label=f"군집{c}",
        )

    axes[0].set_xlabel("구조노후도")
    axes[0].set_ylabel("단속위험도")
    axes[0].set_title("구조노후도 vs 단속위험도")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[1].set_xlabel("주변건물수")
    axes[1].set_ylabel("집중도")
    axes[1].set_title("밀집도 vs 집중도")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/clustering_scatter_{slug}.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── 5. 저장 ─────────────────────────────────────────────────────
    out_path = f"{OUT_DIR}/clustering_result_{slug}.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n  저장: {out_path}")

    results[group_key] = df

# ── 6. 전체 합산 저장 ────────────────────────────────────────────────
df_merged = pd.concat(results.values(), ignore_index=True)
df_merged.to_csv(f"{OUT_DIR}/clustering_result_all.csv", index=False, encoding="utf-8-sig")
print(f"\n\n[전체 저장] {OUT_DIR}/clustering_result_all.csv  ({len(df_merged)}행)")
