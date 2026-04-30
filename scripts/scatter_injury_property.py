# -*- coding: utf-8 -*-
import sys
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

BASE = "c:/Users/USER/Documents/GitHub/기말공모전/NJT-PJT"
SRC  = f"{BASE}/data/화재출동/화재출동_2021_2024.csv"
OUT  = f"{BASE}/data/scatter_injury_property.png"

df = pd.read_csv(SRC, encoding="utf-8-sig", low_memory=False)
df["발화장소_대분류"] = df["발화장소_대분류"].str.strip()

TARGET_CATS = ["주거", "판매/업무시설", "생활서비스"]
COLORS      = ["#4C72B0", "#DD8452", "#55A868"]

df_inj = df[
    (df["인명피해계"] >= 1) & (df["발화장소_대분류"].isin(TARGET_CATS))
].copy()
df_inj["재산피해액(천원)"] = pd.to_numeric(df_inj["재산피해액(천원)"], errors="coerce")
df_inj = df_inj.dropna(subset=["재산피해액(천원)"])

fig, ax = plt.subplots(figsize=(10, 6))

rng = np.random.default_rng(42)

for i, (cat, color) in enumerate(zip(TARGET_CATS, COLORS)):
    sub = df_inj[df_inj["발화장소_대분류"] == cat]["재산피해액(천원)"].values
    # x축 jitter
    jitter = rng.uniform(-0.18, 0.18, size=len(sub))
    ax.scatter(
        np.full(len(sub), i + 1) + jitter,
        sub,
        color=color, alpha=0.55, s=30, linewidths=0.3,
        edgecolors="white", zorder=3,
        label=f"{cat} (N={len(sub)})",
    )
    # 중앙값 가로선
    med = np.median(sub)
    ax.hlines(med, i + 0.78, i + 1.22, colors=color, linewidths=2.5, zorder=4)
    ax.text(i + 1.25, med, f"중앙값\n{med:,.0f}천원",
            va="center", fontsize=8.5, color=color, fontweight="bold")

ax.set_yscale("log")
ax.set_xticks([1, 2, 3])
ax.set_xticklabels(
    [f"{cat}\n(N={df_inj[df_inj['발화장소_대분류']==cat].shape[0]})" for cat in TARGET_CATS],
    fontsize=11,
)
ax.set_xlim(0.5, 3.8)
ax.set_ylabel("재산피해액 (천원, 로그 스케일)", fontsize=11)
ax.set_title(
    "인명피해 1명 이상 화재 — 발화장소 대분류별 재산피해액 산점도\n(2021–2024, 서울 / 가로선 = 중앙값)",
    fontsize=13, pad=14,
)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax.grid(axis="y", alpha=0.25, linestyle="--")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(loc="upper right", fontsize=9, framealpha=0.7)

plt.tight_layout()
fig.savefig(OUT, dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"저장: {OUT}")
