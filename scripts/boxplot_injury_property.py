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
SRC = f"{BASE}/data/화재출동/화재출동_2021_2024.csv"
OUT = f"{BASE}/data/boxplot_injury_property.png"

df = pd.read_csv(SRC, encoding="utf-8-sig", low_memory=False)

# 인명피해 1명 이상 + 대분류 3개 필터
TARGET_CATS = ["주거", "판매/업무시설", "생활서비스"]
df["발화장소_대분류"] = df["발화장소_대분류"].str.strip()
df_inj = df[
    (df["인명피해계"] >= 1) & (df["발화장소_대분류"].isin(TARGET_CATS))
].copy()
df_inj["재산피해액(천원)"] = pd.to_numeric(df_inj["재산피해액(천원)"], errors="coerce")
df_inj = df_inj.dropna(subset=["재산피해액(천원)"])

# 로그 스케일을 위해 0 제외
df_inj = df_inj[df_inj["재산피해액(천원)"] > 0]
groups = [df_inj[df_inj["발화장소_대분류"] == cat]["재산피해액(천원)"].values for cat in TARGET_CATS]

# 각 그룹 요약
print("=== 그룹별 재산피해액(천원) 요약 ===")
for cat, g in zip(TARGET_CATS, groups):
    print(f"\n  [{cat}]  N={len(g)}")
    print(f"    중앙값: {np.median(g):,.0f}  평균: {np.mean(g):,.0f}  최대: {np.max(g):,.0f}")

# ── 박스플롯 ──────────────────────────────────────────────────────────
COLORS = ["#4C72B0", "#DD8452", "#55A868"]
MEDIANPROPS = dict(color="#e74c3c", linewidth=2.5)
BOXPROPS    = dict(linewidth=1.5)
WHISKERPROPS= dict(linewidth=1.4, linestyle="--")
CAPPROPS    = dict(linewidth=1.4)
FLIERPROPS  = dict(marker="o", markersize=3.5, alpha=0.45, linestyle="none")

fig, ax = plt.subplots(figsize=(10, 6))

bp = ax.boxplot(
    groups,
    patch_artist=True,
    notch=False,
    medianprops=MEDIANPROPS,
    boxprops=BOXPROPS,
    whiskerprops=WHISKERPROPS,
    capprops=CAPPROPS,
    flierprops=FLIERPROPS,
)

for patch, color in zip(bp["boxes"], COLORS):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
for flier, color in zip(bp["fliers"], COLORS):
    flier.set_markerfacecolor(color)
    flier.set_markeredgecolor(color)

# 중앙값 레이블
for i, (cat, g) in enumerate(zip(TARGET_CATS, groups), start=1):
    med = np.median(g)
    ax.text(i, med * 1.3, f"{med:,.0f}천원", ha="center", va="bottom",
            fontsize=9, color="#e74c3c", fontweight="bold")

ax.set_yscale("log")
ax.set_xticks([1, 2, 3])
ax.set_xticklabels(
    [f"{cat}\n(N={len(g)})" for cat, g in zip(TARGET_CATS, groups)],
    fontsize=11,
)
ax.set_ylabel("재산피해액 (천원, 로그 스케일)", fontsize=11)
ax.set_title("인명피해 1명 이상 화재 — 발화장소 대분류별 재산피해액 분포\n(2021–2024, 서울)", fontsize=13, pad=14)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax.grid(axis="y", alpha=0.3, linestyle="--")
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
fig.savefig(OUT, dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"\n저장: {OUT}")
