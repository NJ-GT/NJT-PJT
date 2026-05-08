# -*- coding: utf-8 -*-
"""
숙박업 인허가 시각화 듀얼 차트 — 마포구 심층(차트①) + 서울 10구 비교(차트②).

목적:
    - 차트①: 마포구를 4분면(누적 추이/파이/업종 비교/동별 히트맵)으로 심층 분석
    - 차트②: 동일 기간 서울 10구의 외국인민박/숙박/관광숙박 인허가 분포 비교

입력:
    - 원본데이터/서울시 외국인관광도시민박업 인허가 정보.csv
    - 원본데이터/서울시 관광숙박업 인허가 정보.csv
    - 원본데이터/서울시 숙박업 인허가 정보.csv

출력:
    - data/viz_mapo_license.png   (차트①)
    - data/viz_10gu_license.png   (차트②)
"""
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.gridspec import GridSpec  # 4분면/6분면 레이아웃

# 콘솔 한글 출력 깨짐 방지
sys.stdout.reconfigure(encoding="utf-8")

# 시스템 '맑은 고딕' 자동 등록
for font in fm.findSystemFonts():
    if "malgun" in font.lower():
        plt.rcParams["font.family"] = fm.FontProperties(fname=font).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

# 경로/대상 자치구/연도 설정
BASE = "c:/Users/USER/Documents/GitHub/기말공모전/NJT-PJT"
GU_10 = [
    "강남구",
    "강서구",
    "마포구",
    "서초구",
    "성동구",
    "송파구",
    "영등포구",
    "용산구",
    "종로구",
    "중구",
]
YEARS = list(range(2020, 2026))  # 2020~2025
# 색상 팔레트
BG = "#F8F9FC"
C_MAIN = "#2D5BE3"  # 메인 파랑(외국인민박/영업중)
C_LIVE = "#00C49F"  # 보조 초록(숙박업)
C_WARN = "#FF6B6B"  # 경고 빨강(폐업/관광숙박업)
PAL = [
    "#2D5BE3",
    "#00C49F",
    "#FF6B6B",
    "#FFD166",
    "#A855F7",
    "#F97316",
    "#14B8A6",
    "#EC4899",
    "#64748B",
    "#EF4444",
]


def load_gu(path):
    """
    인허가 CSV 로드 + 인허가연도 컬럼 추가 + 자치구 추출 + 10개구 필터링.

    인자:
        path (str): 원본 CSV 경로
    반환:
        DataFrame: '인허가연도' / '구' 컬럼이 추가된 GU_10 한정 사본
    """
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    df["인허가연도"] = pd.to_datetime(df["인허가일자"], errors="coerce").dt.year
    df["구"] = df["지번주소"].str.extract(r"서울특별시\s+(\S+구)")
    return df[df["구"].isin(GU_10)].copy()


# 3종 인허가 데이터 일괄 로드
f1 = load_gu(f"{BASE}/원본데이터/서울시 외국인관광도시민박업 인허가 정보.csv")
f2 = load_gu(f"{BASE}/원본데이터/서울시 관광숙박업 인허가 정보.csv")
f3 = load_gu(f"{BASE}/원본데이터/서울시 숙박업 인허가 정보.csv")

# 마포구 외국인민박 — 동 추출
mapo1 = f1[f1["구"] == "마포구"].copy()
mapo1["동"] = mapo1["지번주소"].str.extract(r"마포구\s+(\S+동|\S+가)")

# ════════════════════════════════════════════════════════════════════
# 차트 ① — 마포구 심층 분석 (4분면)
# ════════════════════════════════════════════════════════════════════
fig1 = plt.figure(figsize=(18, 10), facecolor=BG)
fig1.suptitle(
    "마포구 숙박업 인허가 심층 분석 (2020–2025)",
    fontsize=18,
    fontweight="bold",
    y=0.98,
    color="#1A1A2E",
)
# 2행 4열 — 상단행 4분할(2칸+1+1) + 하단행 1개(히트맵)
gs1 = GridSpec(
    2,
    4,
    figure=fig1,
    hspace=0.45,
    wspace=0.38,
    left=0.06,
    right=0.97,
    top=0.91,
    bottom=0.09,
)

# [A] 누적 건수 꺾은선 — 영업중 vs 폐업
ax = fig1.add_subplot(gs1[0, :2])
ax.set_facecolor("white")
누적_영업 = []
누적_폐업 = []
# 각 연도까지의 누적값
for yr in YEARS:
    sub = mapo1[mapo1["인허가연도"] <= yr]
    누적_영업.append((sub["영업상태명"] == "영업/정상").sum())
    누적_폐업.append((sub["영업상태명"] == "폐업").sum())
# 영역 채움 + 마커 라인 — 영업중(파랑) vs 폐업(빨강)
ax.fill_between(YEARS, 누적_영업, alpha=0.15, color=C_MAIN)
ax.fill_between(YEARS, 누적_폐업, alpha=0.15, color=C_WARN)
ax.plot(
    YEARS, 누적_영업, "o-", color=C_MAIN, lw=2.8, ms=8, label="영업중 누적", zorder=4
)
ax.plot(YEARS, 누적_폐업, "s-", color=C_WARN, lw=2.8, ms=8, label="폐업 누적", zorder=4)
# 라인 위/아래 숫자 라벨
for x, v in zip(YEARS, 누적_영업):
    ax.text(x, v + 8, str(v), ha="center", fontsize=9, fontweight="bold", color=C_MAIN)
for x, v in zip(YEARS, 누적_폐업):
    ax.text(x, v - 18, str(v), ha="center", fontsize=9, color=C_WARN)
ax.set_title(
    "외국인관광도시민박업 누적 현황 (영업중 vs 폐업)",
    fontsize=11,
    fontweight="bold",
    pad=8,
)
ax.set_ylabel("누적 업소 수", fontsize=9)
ax.set_xticks(YEARS)
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.25, linestyle="--")
ax.spines[["top", "right"]].set_visible(False)

# [B] 동별 최신연도 파이차트 + 기타 상세
ax2 = fig1.add_subplot(gs1[0, 2])
ax2.set_facecolor("white")
# 2020~2025 범위 내 가장 최신 연도 식별
latest_yr = mapo1[mapo1["인허가연도"].between(2020, 2025)]["인허가연도"].max()
dong_latest = mapo1[mapo1["인허가연도"] == latest_yr]["동"].value_counts()
# 상위 7개 + 나머지는 '기타'로 묶기
TOP_N = 7
top_d = dong_latest.head(TOP_N)
other_d = dong_latest.iloc[TOP_N:]
other_cnt = other_d.sum()
other_names = ", ".join(other_d.index.tolist())

labels_pie = list(top_d.index) + (["기타"] if other_cnt > 0 else [])
sizes_pie = list(top_d.values) + ([other_cnt] if other_cnt > 0 else [])
wedge_colors = PAL[: len(labels_pie)]
wedges, texts, autotexts = ax2.pie(
    sizes_pie,
    labels=labels_pie,
    autopct="%1.0f%%",
    colors=wedge_colors,
    startangle=140,
    pctdistance=0.75,
    textprops={"fontsize": 8},
)
# 비율 텍스트 — 굵게/작게
for at in autotexts:
    at.set_fontweight("bold")
    at.set_fontsize(8)
ax2.set_title(
    f"{latest_yr}년 신규 인허가\n동별 비중", fontsize=11, fontweight="bold", pad=8
)
# 기타 동 이름들을 차트 아래 박스에 풀어 표기
if other_cnt > 0:
    ax2.text(
        0,
        -1.45,
        f"기타({other_cnt}건): {other_names}",
        ha="center",
        va="top",
        fontsize=7,
        color="#555",
        wrap=True,
        transform=ax2.transData,
        bbox=dict(boxstyle="round,pad=0.3", fc="#F1F5F9", alpha=0.8),
    )

# [C] 업종별 신규 인허가 막대 — 외국인민박/숙박/관광숙박 3종 비교
ax3 = fig1.add_subplot(gs1[0, 3])
ax3.set_facecolor("white")
# 외국인민박 — 마포구 연도별 카운트
m1_yr = (
    mapo1[mapo1["인허가연도"].between(2020, 2025)]
    .groupby("인허가연도")
    .size()
    .reindex(YEARS, fill_value=0)
)
mapo2 = f2[f2["구"] == "마포구"]
mapo3 = f3[f3["구"] == "마포구"]
# 관광숙박업 / 일반숙박업 — 마포구 연도별 카운트
m2_yr = (
    mapo2[mapo2["인허가연도"].between(2020, 2025)]
    .groupby("인허가연도")
    .size()
    .reindex(YEARS, fill_value=0)
)
m3_yr = (
    mapo3[mapo3["인허가연도"].between(2020, 2025)]
    .groupby("인허가연도")
    .size()
    .reindex(YEARS, fill_value=0)
)
# 그룹 막대 — 너비 0.25 씩 좌/중/우로 배치
x = np.arange(len(YEARS))
w = 0.25
ax3.bar(x - w, m1_yr.values, width=w, color=C_MAIN, alpha=0.85, label="외국인민박")
ax3.bar(x, m3_yr.values, width=w, color=C_LIVE, alpha=0.85, label="숙박업")
ax3.bar(x + w, m2_yr.values, width=w, color=C_WARN, alpha=0.85, label="관광숙박업")
ax3.set_title("업종별 신규 인허가\n연도 비교", fontsize=11, fontweight="bold", pad=8)
ax3.set_xticks(x)
ax3.set_xticklabels(YEARS, fontsize=7, rotation=30)
ax3.legend(fontsize=7)
ax3.grid(axis="y", alpha=0.25, linestyle="--")
ax3.spines[["top", "right"]].set_visible(False)

# [D] 동별 히트맵 — 상위 10개 동 × 연도
ax4 = fig1.add_subplot(gs1[1, :])
ax4.set_facecolor("white")
# 동×연도 피벗
dong_piv = (
    mapo1[mapo1["인허가연도"].between(2020, 2025)]
    .groupby(["동", "인허가연도"])
    .size()
    .unstack(fill_value=0)
    .reindex(columns=YEARS, fill_value=0)
)
# 합계 기준 상위 10개 동만 추림
top10d = dong_piv.sum(axis=1).nlargest(10).index
dong_piv = dong_piv.loc[top10d]
im4 = ax4.imshow(dong_piv.values, aspect="auto", cmap="YlOrRd", vmin=0)
ax4.set_xticks(range(len(YEARS)))
ax4.set_xticklabels(YEARS, fontsize=10)
ax4.set_yticks(range(len(top10d)))
ax4.set_yticklabels(top10d, fontsize=10)
# 셀 값 텍스트 — 0이면 '·' 표기, 짙은 셀이면 흰 글자
for i in range(len(top10d)):
    for j in range(len(YEARS)):
        v = int(dong_piv.values[i, j])
        ax4.text(
            j,
            i,
            str(v) if v > 0 else "·",
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            color="white" if v > dong_piv.values.max() * 0.55 else "#333",
        )
ax4.set_title(
    "동별 외국인관광도시민박업 신규 인허가 히트맵 (상위 10개 동)",
    fontsize=11,
    fontweight="bold",
    pad=8,
)
plt.colorbar(im4, ax=ax4, orientation="vertical", shrink=0.9, label="건수", pad=0.01)
# 모든 테두리 제거 — 깔끔한 히트맵
ax4.spines[["top", "right", "left", "bottom"]].set_visible(False)

# 차트① 저장
plt.savefig(
    f"{BASE}/data/viz_mapo_license.png", dpi=150, bbox_inches="tight", facecolor=BG
)
print("저장: viz_mapo_license.png")
plt.close()

# ════════════════════════════════════════════════════════════════════
# 차트 ② — 서울 10개 구 비교 (4분면)
# ════════════════════════════════════════════════════════════════════
fig2 = plt.figure(figsize=(18, 12), facecolor=BG)
fig2.suptitle(
    "서울 10개 구 숙박업 인허가 비교 (2020–2025)",
    fontsize=18,
    fontweight="bold",
    y=0.98,
    color="#1A1A2E",
)
# 2행 3열 — 좌상(2칸) + 우상(1칸) + 좌하(2칸) + 우하(1칸)
gs2 = GridSpec(
    2,
    3,
    figure=fig2,
    hspace=0.42,
    wspace=0.35,
    left=0.07,
    right=0.96,
    top=0.91,
    bottom=0.08,
)

# 외국인민박 구×연도 피벗 — 모든 차트의 공통 입력
민박_pivot = (
    f1[f1["인허가연도"].between(2020, 2025)]
    .groupby(["구", "인허가연도"])
    .size()
    .unstack(fill_value=0)
    .reindex(index=GU_10, columns=YEARS, fill_value=0)
)

# [A] 구별 연도별 그룹 막대 — 6개 연도 × 10개 구
ax = fig2.add_subplot(gs2[0, :2])
ax.set_facecolor("white")
x = np.arange(len(GU_10))
w = 0.13
# 연도마다 막대 위치를 좌→우로 옮겨가며 그룹화
for i, yr in enumerate(YEARS):
    offset = (i - (len(YEARS) - 1) / 2) * w
    ax.bar(
        x + offset,
        민박_pivot[yr].values,
        width=w,
        color=PAL[i],
        alpha=0.85,
        label=str(yr),
        zorder=3,
    )
ax.set_title(
    "구별 외국인관광도시민박업 신규 인허가 (연도별)",
    fontsize=11,
    fontweight="bold",
    pad=8,
)
ax.set_xticks(x)
ax.set_xticklabels(GU_10, fontsize=9, rotation=15, ha="right")
ax.set_ylabel("신규 인허가 수", fontsize=9)
ax.legend(fontsize=8, ncol=6, loc="upper right")
ax.grid(axis="y", alpha=0.25, linestyle="--")
ax.spines[["top", "right"]].set_visible(False)

# [B] 구별 총합 수평 막대 — 마포구만 강조 색상
ax2 = fig2.add_subplot(gs2[0, 2])
ax2.set_facecolor("white")
총합 = 민박_pivot.sum(axis=1).sort_values(ascending=True)
# 마포구만 메인 색, 나머지는 회색
colors_h = [C_MAIN if g == "마포구" else "#94A3B8" for g in 총합.index]
ax2.barh(range(len(총합)), 총합.values, color=colors_h, alpha=0.9, height=0.6, zorder=3)
# 막대 끝 숫자 — 마포구만 진한 색
for i, (g, v) in enumerate(총합.items()):
    ax2.text(
        v + 5,
        i,
        str(int(v)),
        va="center",
        fontsize=9,
        fontweight="bold",
        color=C_MAIN if g == "마포구" else "#555",
    )
ax2.set_yticks(range(len(총합)))
ax2.set_yticklabels(총합.index, fontsize=9)
ax2.set_title(
    "구별 외국인민박\n2020~2025 총 신규 인허가", fontsize=11, fontweight="bold", pad=8
)
ax2.set_xlabel("합계 건수", fontsize=9)
ax2.grid(axis="x", alpha=0.25, linestyle="--")
ax2.spines[["top", "right"]].set_visible(False)

# [C] 구×연도 히트맵 — 외국인민박 신규 인허가
ax3 = fig2.add_subplot(gs2[1, :2])
ax3.set_facecolor("white")
im3 = ax3.imshow(민박_pivot.values, aspect="auto", cmap="Blues", vmin=0)
ax3.set_xticks(range(len(YEARS)))
ax3.set_xticklabels(YEARS, fontsize=10)
ax3.set_yticks(range(len(GU_10)))
ax3.set_yticklabels(GU_10, fontsize=10)
# 셀 값 텍스트 — 0이면 '·', 짙으면 흰 글자
for i in range(len(GU_10)):
    for j in range(len(YEARS)):
        v = int(민박_pivot.values[i, j])
        ax3.text(
            j,
            i,
            str(v) if v > 0 else "·",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="white" if v > 민박_pivot.values.max() * 0.55 else "#333",
        )
ax3.set_title(
    "구×연도 외국인관광도시민박업 신규 인허가 히트맵",
    fontsize=11,
    fontweight="bold",
    pad=8,
)
plt.colorbar(im3, ax=ax3, orientation="vertical", shrink=0.9, label="건수", pad=0.01)
ax3.spines[["top", "right", "left", "bottom"]].set_visible(False)

# [D] 업종 합산 스택 막대 — 외국인민박 + 숙박업 + 관광숙박업
ax4 = fig2.add_subplot(gs2[1, 2])
ax4.set_facecolor("white")
# 자치구별 3종 인허가 합계
숙박_구 = (
    f3[f3["인허가연도"].between(2020, 2025)]
    .groupby("구")
    .size()
    .reindex(GU_10, fill_value=0)
)
관광_구 = (
    f2[f2["인허가연도"].between(2020, 2025)]
    .groupby("구")
    .size()
    .reindex(GU_10, fill_value=0)
)
민박_구 = (
    f1[f1["인허가연도"].between(2020, 2025)]
    .groupby("구")
    .size()
    .reindex(GU_10, fill_value=0)
)
# 3종 합산을 기준으로 자치구 정렬 (가장 작은 구가 위에)
order = (숙박_구 + 관광_구 + 민박_구).sort_values(ascending=True)
gu_ord = order.index
# 누적 가로 막대 — 외국인민박(파) → 숙박(초록) → 관광숙박(빨강)
ax4.barh(
    range(len(gu_ord)),
    민박_구[gu_ord].values,
    color=C_MAIN,
    alpha=0.85,
    label="외국인민박",
    height=0.6,
)
ax4.barh(
    range(len(gu_ord)),
    숙박_구[gu_ord].values,
    left=민박_구[gu_ord].values,  # 외국인민박 막대 끝부터 시작
    color=C_LIVE,
    alpha=0.85,
    label="숙박업",
    height=0.6,
)
ax4.barh(
    range(len(gu_ord)),
    관광_구[gu_ord].values,
    left=(민박_구[gu_ord] + 숙박_구[gu_ord]).values,  # 두 막대 합 끝부터
    color=C_WARN,
    alpha=0.85,
    label="관광숙박업",
    height=0.6,
)
ax4.set_yticks(range(len(gu_ord)))
ax4.set_yticklabels(gu_ord, fontsize=9)
ax4.set_title(
    "구별 업종 합산\n신규 인허가 (2020~2025)", fontsize=11, fontweight="bold", pad=8
)
ax4.set_xlabel("합계 건수", fontsize=9)
ax4.legend(fontsize=8, loc="lower right")
ax4.grid(axis="x", alpha=0.25, linestyle="--")
ax4.spines[["top", "right"]].set_visible(False)

# 차트② 저장
plt.savefig(
    f"{BASE}/data/viz_10gu_license.png", dpi=150, bbox_inches="tight", facecolor=BG
)
print("저장: viz_10gu_license.png")
plt.close()
