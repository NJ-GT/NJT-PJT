# -*- coding: utf-8 -*-
"""
인명피해 1명 이상 화재 — 발화장소 대분류별 재산피해액 분포 박스플롯.

목적:
    {주거 / 판매·업무시설 / 생활서비스} 세 그룹의 재산피해액(천원) 분포를
    로그 스케일 박스플롯으로 비교하고, 중앙값을 명시 라벨로 표시한다.

산출:
    NJT-PJT/data/boxplot_injury_property.png
"""
import sys
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# Windows 콘솔 한글 깨짐 방지
sys.stdout.reconfigure(encoding="utf-8")
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# 입출력 경로
BASE = "c:/Users/USER/Documents/GitHub/기말공모전/NJT-PJT"
SRC = f"{BASE}/data/화재출동/화재출동_2021_2024.csv"
OUT = f"{BASE}/data/boxplot_injury_property.png"

# 화재 출동 원천 데이터 로드
df = pd.read_csv(SRC, encoding="utf-8-sig", low_memory=False)

# 인명피해 ≥ 1 + 대상 3개 카테고리만 필터
TARGET_CATS = ["주거", "판매/업무시설", "생활서비스"]
# 양끝 공백 제거 — 매칭 누락 방지
df["발화장소_대분류"] = df["발화장소_대분류"].str.strip()
df_inj = df[(df["인명피해계"] >= 1) & (df["발화장소_대분류"].isin(TARGET_CATS))].copy()
# 재산피해액 수치 변환 + NaN 행 제거
df_inj["재산피해액(천원)"] = pd.to_numeric(df_inj["재산피해액(천원)"], errors="coerce")
df_inj = df_inj.dropna(subset=["재산피해액(천원)"])

# 로그 스케일에서 0 이하 값 제거 (log(0) 방지)
df_inj = df_inj[df_inj["재산피해액(천원)"] > 0]
# 카테고리별 시리즈를 numpy 배열로 추출 (boxplot 입력)
groups = [
    df_inj[df_inj["발화장소_대분류"] == cat]["재산피해액(천원)"].values
    for cat in TARGET_CATS
]

# 그룹별 요약 통계 (콘솔 점검용)
print("=== 그룹별 재산피해액(천원) 요약 ===")
for cat, g in zip(TARGET_CATS, groups):
    print(f"\n  [{cat}]  N={len(g)}")
    print(
        f"    중앙값: {np.median(g):,.0f}  평균: {np.mean(g):,.0f}  최대: {np.max(g):,.0f}"
    )

# ── 박스플롯 스타일 ──────────────────────────────────────────────
COLORS = ["#4C72B0", "#DD8452", "#55A868"]
MEDIANPROPS = dict(color="#e74c3c", linewidth=2.5)
BOXPROPS = dict(linewidth=1.5)
WHISKERPROPS = dict(linewidth=1.4, linestyle="--")
CAPPROPS = dict(linewidth=1.4)
FLIERPROPS = dict(marker="o", markersize=3.5, alpha=0.45, linestyle="none")

fig, ax = plt.subplots(figsize=(10, 6))

# 박스플롯 그리기 (patch_artist=True 로 박스 채우기 가능)
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

# 박스마다 카테고리 색상 + 투명도 적용
for patch, color in zip(bp["boxes"], COLORS):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
# 이상치 마커도 색상 통일
for flier, color in zip(bp["fliers"], COLORS):
    flier.set_markerfacecolor(color)
    flier.set_markeredgecolor(color)

# 박스 위에 중앙값 라벨 표기 (가독성 ↑)
for i, (cat, g) in enumerate(zip(TARGET_CATS, groups), start=1):
    med = np.median(g)
    ax.text(
        i,
        med * 1.3,
        f"{med:,.0f}천원",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#e74c3c",
        fontweight="bold",
    )

# 재산피해액 폭이 매우 넓어 로그 스케일
ax.set_yscale("log")
ax.set_xticks([1, 2, 3])
ax.set_xticklabels(
    [f"{cat}\n(N={len(g)})" for cat, g in zip(TARGET_CATS, groups)],
    fontsize=11,
)
ax.set_ylabel("재산피해액 (천원, 로그 스케일)", fontsize=11)
ax.set_title(
    "인명피해 1명 이상 화재 — 발화장소 대분류별 재산피해액 분포\n(2021–2024, 서울)",
    fontsize=13,
    pad=14,
)
# y축 천 단위 콤마 + 가로 그리드
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax.grid(axis="y", alpha=0.3, linestyle="--")
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
fig.savefig(OUT, dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"\n저장: {OUT}")
