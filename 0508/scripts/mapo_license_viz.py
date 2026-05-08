# -*- coding: utf-8 -*-
"""
마포구 한정 숙박업 인허가 트렌드 종합 시각화 스크립트.

목적:
    - 외국인민박/일반숙박/관광숙박 3종 인허가 데이터에서 마포구만 추출,
      6분면(연도별 막대+선, 동별 히트맵, 핵심3동 추이, 업태별 누적, 관광업태 누적, 2020 vs 2024 비교)
      종합 패널 PNG 1장으로 저장.

입력:
    - 원본데이터/서울시 외국인관광도시민박업 인허가 정보.csv
    - 원본데이터/서울시 관광숙박업 인허가 정보.csv
    - 원본데이터/서울시 숙박업 인허가 정보.csv

출력:
    - data/mapo_license_trend.png

처리 흐름:
    1) 폰트/경로 세팅 후 3종 CSV 마포구 필터링 로드
    2) 인허가연도(2020~2024) 기준 집계: 연도별/동별/업태별 피벗
    3) 6개 서브플롯 그리고 한 장 PNG 저장
"""
import pandas as pd  # CSV 로드와 집계
import numpy as np  # 막대 위치/누적용 숫자 연산
import sys  # 표준출력 인코딩
import matplotlib.pyplot as plt  # 시각화
import matplotlib.font_manager as fm  # 한글 폰트 탐색
from matplotlib.gridspec import GridSpec  # 6분면 레이아웃

# 콘솔 한글 깨짐 방지 — UTF-8 강제
sys.stdout.reconfigure(encoding="utf-8")

# 시스템에서 '맑은 고딕' 찾아서 matplotlib 기본 폰트로
for font in fm.findSystemFonts():
    if "malgun" in font.lower():
        plt.rcParams["font.family"] = fm.FontProperties(fname=font).get_name()
        break
# 한글 폰트 사용 시 음수 부호 깨짐 방지
plt.rcParams["axes.unicode_minus"] = False

# 프로젝트 루트
BASE = "c:/Users/USER/Documents/GitHub/기말공모전/NJT-PJT"


def load_mapo(path):
    """
    인허가 CSV 로드 후 마포구 행만 필터링.

    인자:
        path (str): 원본 CSV 경로
    반환:
        DataFrame: '인허가연도' 컬럼이 추가된, 마포구 데이터만의 DataFrame
    """
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    # 지번주소에 '마포구' 가 포함된 행만 — na 값 처리 옵션 켜둠
    df = df[df["지번주소"].str.contains("마포구", na=False)].copy()
    # 인허가일자 → 연도(int)
    df["인허가연도"] = pd.to_datetime(df["인허가일자"], errors="coerce").dt.year
    return df


# 3종 인허가 데이터 마포구만 로드
f1 = load_mapo(f"{BASE}/원본데이터/서울시 외국인관광도시민박업 인허가 정보.csv")
f2 = load_mapo(f"{BASE}/원본데이터/서울시 관광숙박업 인허가 정보.csv")
f3 = load_mapo(f"{BASE}/원본데이터/서울시 숙박업 인허가 정보.csv")

# 분석 대상 5년 (2020~2024)
YEARS = list(range(2020, 2025))

# 외국인민박 — 연도별 신규 인허가(전체)
민박_yr = (
    f1[f1["인허가연도"].between(2020, 2024)]
    .groupby("인허가연도")
    .size()
    .reindex(YEARS, fill_value=0)
)
# 외국인민박 — 연도별 신규 인허가 중 '영업/정상' 만
민박_영업 = (
    f1[f1["인허가연도"].between(2020, 2024) & (f1["영업상태명"] == "영업/정상")]
    .groupby("인허가연도")
    .size()
    .reindex(YEARS, fill_value=0)
)

# 마포구 내 동(또는 X가) 추출 — '마포구 OO동' 패턴
f1["동"] = f1["지번주소"].str.extract(r"마포구\s+(\S+동|\S+가)")
# 동×연도 피벗 — 결측은 0
동_pivot = (
    f1[f1["인허가연도"].between(2020, 2024)]
    .groupby(["동", "인허가연도"])
    .size()
    .unstack(fill_value=0)
    .reindex(columns=YEARS, fill_value=0)
)
# 합계 기준 상위 8개 동만 추림
top_dong = 동_pivot.sum(axis=1).nlargest(8).index
동_pivot = 동_pivot.loc[top_dong]

# 일반숙박업 — 위생업태명을 '업태' 로 별칭
f3["업태"] = f3["위생업태명"]
# 연도×업태 피벗
숙박_yr = (
    f3[f3["인허가연도"].between(2020, 2024)]
    .groupby(["인허가연도", "업태"])
    .size()
    .unstack(fill_value=0)
    .reindex(index=YEARS, fill_value=0)
)

# 관광숙박업 — '관광숙박업상세명' 을 '업태' 로
f2["업태"] = f2["관광숙박업상세명"]
관광_yr = (
    f2[f2["인허가연도"].between(2020, 2024)]
    .groupby(["인허가연도", "업태"])
    .size()
    .unstack(fill_value=0)
    .reindex(index=YEARS, fill_value=0)
)

# 컬러 팔레트
C_MAIN = "#2D5BE3"  # 메인 파랑
C_LIVE = "#00C49F"  # 영업중(초록)
PALETTE = [
    "#2D5BE3",
    "#00C49F",
    "#FF6B6B",
    "#FFD166",
    "#A855F7",
    "#F97316",
    "#14B8A6",
    "#EC4899",
]
BG = "#F8F9FC"  # 전체 배경

# figure 생성 — 가로 18 x 세로 12, 배경색 적용
fig = plt.figure(figsize=(18, 12), facecolor=BG)
fig.suptitle(
    "마포구 숙박업 신규 인허가 트렌드 (2020–2024)",
    fontsize=18,
    fontweight="bold",
    y=0.98,
    color="#1A1A2E",
)

# 2행 3열 그리드 (총 6개 서브플롯)
gs = GridSpec(
    2,
    3,
    figure=fig,
    hspace=0.42,
    wspace=0.35,
    left=0.07,
    right=0.96,
    top=0.92,
    bottom=0.08,
)

# [1] 외국인민박 연도별 막대(신규) + 선(영업중)
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor("white")
# 막대 — 신규인허가 전체
bars = ax1.bar(
    YEARS,
    민박_yr.values,
    color=C_MAIN,
    alpha=0.85,
    width=0.55,
    zorder=3,
    label="신규인허가",
)
# 라인 — 그 중 영업중인 비중
ax1.plot(
    YEARS, 민박_영업.values, "o-", color=C_LIVE, lw=2.5, ms=7, zorder=4, label="영업중"
)
# 막대 위 숫자 라벨
for bar, val in zip(bars, 민박_yr.values):
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        val + 5,
        str(int(val)),
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color=C_MAIN,
    )
ax1.set_title(
    "외국인관광 도시민박업\n연도별 신규 인허가", fontsize=11, fontweight="bold", pad=8
)
ax1.set_ylabel("신규 인허가 수", fontsize=9)
ax1.set_xticks(YEARS)
ax1.legend(fontsize=8, loc="upper left")
ax1.grid(axis="y", alpha=0.3, linestyle="--")
ax1.spines[["top", "right"]].set_visible(False)
# y축 상한을 최대값의 1.2배로 — 라벨 잘림 방지
ax1.set_ylim(0, max(민박_yr.values) * 1.2)

# [2] 동별 히트맵 — 상위 8개 동 × 연도
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_facecolor("white")
# 'Blues' 컬러맵 — 값이 클수록 짙은 파랑
im = ax2.imshow(동_pivot.values, aspect="auto", cmap="Blues", vmin=0)
ax2.set_xticks(range(len(YEARS)))
ax2.set_xticklabels(YEARS, fontsize=9)
ax2.set_yticks(range(len(top_dong)))
ax2.set_yticklabels(top_dong, fontsize=9)
# 셀 값 텍스트 — 0이면 빈칸, 짙은 셀이면 흰 글자
for i in range(len(top_dong)):
    for j in range(len(YEARS)):
        v = int(동_pivot.values[i, j])
        ax2.text(
            j,
            i,
            str(v) if v > 0 else "",
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="white" if v > 동_pivot.values.max() * 0.6 else "#333",
        )
ax2.set_title(
    "동별 외국인민박 신규 인허가\n(상위 8개 동)", fontsize=11, fontweight="bold", pad=8
)
plt.colorbar(im, ax=ax2, shrink=0.8, label="건수")
# 모든 테두리 제거 — 깔끔한 히트맵
ax2.spines[["top", "right", "left", "bottom"]].set_visible(False)

# [3] 핵심 3개 동 선그래프 — 연남동/서교동/동교동
ax3 = fig.add_subplot(gs[0, 2])
ax3.set_facecolor("white")
top3 = ["연남동", "서교동", "동교동"]  # 마포구 외국인민박 대표 동
for i, dong in enumerate(top3):
    if dong in 동_pivot.index:
        vals = 동_pivot.loc[dong].values
        # 라인 + 마커 + 라인 아래 옅은 영역 채움
        ax3.plot(
            YEARS, vals, "o-", color=PALETTE[i], lw=2.5, ms=7, label=dong, zorder=3
        )
        ax3.fill_between(YEARS, vals, alpha=0.08, color=PALETTE[i])
        # 마지막 점 옆에 동 이름 라벨 — 범례 의존도 낮춤
        ax3.text(
            YEARS[-1] + 0.05,
            vals[-1],
            dong,
            va="center",
            fontsize=9,
            color=PALETTE[i],
            fontweight="bold",
        )
ax3.set_title("핵심 3개 동 연도별 추이", fontsize=11, fontweight="bold", pad=8)
ax3.set_ylabel("신규 인허가 수", fontsize=9)
ax3.set_xticks(YEARS)
ax3.legend(fontsize=8)
ax3.grid(axis="y", alpha=0.3, linestyle="--")
ax3.spines[["top", "right"]].set_visible(False)

# [4] 일반숙박업 업태별 누적 막대 (스택바)
ax4 = fig.add_subplot(gs[1, 0])
ax4.set_facecolor("white")
if not 숙박_yr.empty and 숙박_yr.sum().sum() > 0:
    # 합계 큰 업태부터 누적 — 시각적 안정감
    cols = 숙박_yr.sum().sort_values(ascending=False).index
    bottom = np.zeros(len(YEARS))  # 누적 막대의 바닥값
    for i, col in enumerate(cols):
        vals = 숙박_yr[col].values if col in 숙박_yr.columns else np.zeros(len(YEARS))
        ax4.bar(
            YEARS,
            vals,
            bottom=bottom,
            color=PALETTE[i],
            alpha=0.85,
            width=0.55,
            label=col,
            zorder=3,
        )
        bottom += vals  # 다음 막대를 쌓을 새 바닥값
ax4.set_title(
    "숙박업 업태별 신규 인허가\n(위생업태 기준)", fontsize=11, fontweight="bold", pad=8
)
ax4.set_ylabel("신규 인허가 수", fontsize=9)
ax4.set_xticks(YEARS)
ax4.legend(fontsize=8, loc="upper right")
ax4.grid(axis="y", alpha=0.3, linestyle="--")
ax4.spines[["top", "right"]].set_visible(False)

# [5] 관광숙박업 업태별 누적 막대
ax5 = fig.add_subplot(gs[1, 1])
ax5.set_facecolor("white")
if not 관광_yr.empty and 관광_yr.sum().sum() > 0:
    cols2 = 관광_yr.sum().sort_values(ascending=False).index
    bottom2 = np.zeros(len(YEARS))
    for i, col in enumerate(cols2):
        vals = 관광_yr[col].values if col in 관광_yr.columns else np.zeros(len(YEARS))
        # 색은 [4]와 겹치지 않게 인덱스 +3
        ax5.bar(
            YEARS,
            vals,
            bottom=bottom2,
            color=PALETTE[i + 3],
            alpha=0.85,
            width=0.55,
            label=col,
            zorder=3,
        )
        bottom2 += vals
ax5.set_title("관광숙박업 업태별 신규 인허가", fontsize=11, fontweight="bold", pad=8)
ax5.set_ylabel("신규 인허가 수", fontsize=9)
ax5.set_xticks(YEARS)
ax5.legend(fontsize=8)
ax5.grid(axis="y", alpha=0.3, linestyle="--")
ax5.spines[["top", "right"]].set_visible(False)

# [6] 2020 vs 2024 업종별 비교 — 5년간 변화 한눈에
ax6 = fig.add_subplot(gs[1, 2])
ax6.set_facecolor("white")
# 3개 업종에 대해 (2020값, 2024값) 튜플로 요약
summary = {
    "외국인관광\n도시민박업": (int(민박_yr[2020]), int(민박_yr[2024])),
    "숙박업\n(합계)": (
        int(숙박_yr.sum(axis=1).get(2020, 0)),
        int(숙박_yr.sum(axis=1).get(2024, 0)),
    ),
    "관광숙박업\n(합계)": (
        int(관광_yr.sum(axis=1).get(2020, 0)),
        int(관광_yr.sum(axis=1).get(2024, 0)),
    ),
}
labels = list(summary.keys())
v2020 = [summary[k][0] for k in labels]
v2024 = [summary[k][1] for k in labels]
x = np.arange(len(labels))
w = 0.32
# 좌측 막대(2020) — 회색, 우측 막대(2024) — 메인 파랑
ax6.bar(
    x - w / 2, v2020, width=w, color="#94A3B8", alpha=0.85, label="2020년", zorder=3
)
ax6.bar(x + w / 2, v2024, width=w, color=C_MAIN, alpha=0.85, label="2024년", zorder=3)
# 막대 위 숫자 라벨 — 비교 가독성 향상
for i, (a, b) in enumerate(zip(v2020, v2024)):
    ax6.text(
        i - w / 2,
        a + 1,
        str(a),
        ha="center",
        va="bottom",
        fontsize=9,
        color="#64748B",
        fontweight="bold",
    )
    ax6.text(
        i + w / 2,
        b + 1,
        str(b),
        ha="center",
        va="bottom",
        fontsize=9,
        color=C_MAIN,
        fontweight="bold",
    )
ax6.set_title(
    "2020 vs 2024\n업종별 신규 인허가 비교", fontsize=11, fontweight="bold", pad=8
)
ax6.set_xticks(x)
ax6.set_xticklabels(labels, fontsize=9)
ax6.legend(fontsize=9)
ax6.grid(axis="y", alpha=0.3, linestyle="--")
ax6.spines[["top", "right"]].set_visible(False)

# 최종 PNG 저장 — 150 dpi 고해상도, 여백 자동, 배경색 유지
plt.savefig(
    f"{BASE}/data/mapo_license_trend.png", dpi=150, bbox_inches="tight", facecolor=BG
)
print("저장 완료: data/mapo_license_trend.png")
