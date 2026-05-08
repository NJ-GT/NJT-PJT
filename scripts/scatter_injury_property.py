# -*- coding: utf-8 -*-
"""
인명피해 1명 이상 화재의 발화장소 대분류별 재산피해액 산점도.

데이터:
    NJT-PJT/data/화재출동/화재출동_2021_2024.csv
필터:
    인명피해계 ≥ 1
    발화장소_대분류 ∈ {주거, 판매/업무시설, 생활서비스}
시각화:
    - 카테고리별로 jitter된 산점도 (가독성 ↑)
    - 가로 막대로 중앙값 표시 + 옆에 수치 라벨
    - y축은 로그 스케일 (재산피해액의 이상치 영향 완화)
출력:
    NJT-PJT/data/scatter_injury_property.png
"""
import sys
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# Windows 콘솔 한글 깨짐 방지
sys.stdout.reconfigure(encoding="utf-8")
# 한글 폰트 + 음수 부호 깨짐 방지
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# 입출력 경로
BASE = "c:/Users/USER/Documents/GitHub/기말공모전/NJT-PJT"
SRC = f"{BASE}/data/화재출동/화재출동_2021_2024.csv"
OUT = f"{BASE}/data/scatter_injury_property.png"

# 원본 화재 출동 데이터
df = pd.read_csv(SRC, encoding="utf-8-sig", low_memory=False)
# 카테고리 양끝 공백 제거 — 매칭 누락 방지
df["발화장소_대분류"] = df["발화장소_대분류"].str.strip()

# 산점도에 표시할 카테고리와 색상
TARGET_CATS = ["주거", "판매/업무시설", "생활서비스"]
COLORS = ["#4C72B0", "#DD8452", "#55A868"]

# 인명피해 1명 이상 + 대상 카테고리만
df_inj = df[(df["인명피해계"] >= 1) & (df["발화장소_대분류"].isin(TARGET_CATS))].copy()
# 재산피해액 수치 변환 + NaN 행 제거
df_inj["재산피해액(천원)"] = pd.to_numeric(df_inj["재산피해액(천원)"], errors="coerce")
df_inj = df_inj.dropna(subset=["재산피해액(천원)"])

# 차트 객체
fig, ax = plt.subplots(figsize=(10, 6))

# jitter 재현성 — 같은 그림이 매번 동일하게 나오도록 시드 고정
rng = np.random.default_rng(42)

for i, (cat, color) in enumerate(zip(TARGET_CATS, COLORS)):
    # 해당 카테고리의 재산피해액 값들
    sub = df_inj[df_inj["발화장소_대분류"] == cat]["재산피해액(천원)"].values
    # x축에 약간의 가로 흔들림(jitter)을 줘 점이 겹치지 않게
    jitter = rng.uniform(-0.18, 0.18, size=len(sub))
    ax.scatter(
        np.full(len(sub), i + 1) + jitter,
        sub,
        color=color,
        alpha=0.55,
        s=30,
        linewidths=0.3,
        edgecolors="white",
        zorder=3,
        label=f"{cat} (N={len(sub)})",
    )
    # 카테고리 중앙값 — 가로 굵은 선으로 강조
    med = np.median(sub)
    ax.hlines(med, i + 0.78, i + 1.22, colors=color, linewidths=2.5, zorder=4)
    # 중앙값 옆에 수치 라벨
    ax.text(
        i + 1.25,
        med,
        f"중앙값\n{med:,.0f}천원",
        va="center",
        fontsize=8.5,
        color=color,
        fontweight="bold",
    )

# 재산피해액의 분포 폭이 매우 넓으므로 로그 스케일
ax.set_yscale("log")
# x 틱은 1,2,3 위치에 카테고리명 + N 표기
ax.set_xticks([1, 2, 3])
ax.set_xticklabels(
    [
        f"{cat}\n(N={df_inj[df_inj['발화장소_대분류'] == cat].shape[0]})"
        for cat in TARGET_CATS
    ],
    fontsize=11,
)
# 좌우 여백 + y축 라벨 + 제목
ax.set_xlim(0.5, 3.8)
ax.set_ylabel("재산피해액 (천원, 로그 스케일)", fontsize=11)
ax.set_title(
    "인명피해 1명 이상 화재 — 발화장소 대분류별 재산피해액 산점도\n(2021–2024, 서울 / 가로선 = 중앙값)",
    fontsize=13,
    pad=14,
)
# y축 숫자 포매팅 — 천 단위 콤마
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
# 가독성 보조 — 가로 그리드 + 상/우측 테두리 제거
ax.grid(axis="y", alpha=0.25, linestyle="--")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(loc="upper right", fontsize=9, framealpha=0.7)

plt.tight_layout()
fig.savefig(OUT, dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"저장: {OUT}")
