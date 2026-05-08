# -*- coding: utf-8 -*-
"""
서울 10개 자치구별 숙박업(외국인관광도시민박업/관광숙박업/일반숙박업) 인허가 심층 분석 시각화 스크립트.

목적:
    - 자치구마다 4분면 패널(누적 추이/파이/업종 비교/동별 히트맵)을 그려 PNG로 저장.

입력:
    - 원본데이터/서울시 외국인관광도시민박업 인허가 정보.csv
    - 원본데이터/서울시 관광숙박업 인허가 정보.csv
    - 원본데이터/서울시 숙박업 인허가 정보.csv

출력:
    - data/viz_each_gu/{구}_인허가분석.png  (10개 파일)

처리 흐름:
    1) 폰트/색상/대상 자치구 등 전역 상수 설정
    2) 3종 인허가 CSV 로드 후 '서울특별시 OO구' 패턴으로 자치구 추출
    3) 각 구별로 figure 생성 → [A]누적 [B]파이 [C]업종비교 [D]동별 히트맵 그리기
    4) PNG 저장 후 figure 닫기
"""
import pandas as pd  # 데이터프레임 처리
import numpy as np  # 수치 연산(막대 위치 계산 등)
import sys  # 표준 출력 인코딩 재설정용
import os  # 출력 디렉터리 생성용
import matplotlib.pyplot as plt  # 시각화 메인
import matplotlib.font_manager as fm  # 한글 폰트 탐색
from matplotlib.gridspec import GridSpec  # 4분면 레이아웃

# Windows 콘솔에서 한글 깨짐 방지 — 표준 출력 인코딩을 UTF-8 로 강제
sys.stdout.reconfigure(encoding="utf-8")

# 시스템에 설치된 폰트들 중 'malgun'(맑은 고딕) 을 찾아 matplotlib 기본 폰트로 지정
for font in fm.findSystemFonts():
    if "malgun" in font.lower():
        plt.rcParams["font.family"] = fm.FontProperties(fname=font).get_name()
        break
# 한글 폰트 사용 시 마이너스 기호가 깨지는 문제 방지
plt.rcParams["axes.unicode_minus"] = False

# 프로젝트 루트 절대경로 (모든 입출력의 기준)
BASE = "c:/Users/USER/Documents/GitHub/기말공모전/NJT-PJT"
# 분석 대상 10개 자치구 (서울시 전체 25개 중 핵심 10개)
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
# 분석 연도 범위 (2020~2025)
YEARS = list(range(2020, 2026))
# 시각화 배경/주요 색상 팔레트
BG = "#F8F9FC"  # 전체 배경 (연한 회청색)
C_MAIN = "#2D5BE3"  # 메인 색(영업중/외국인민박)
C_LIVE = "#00C49F"  # 보조 색(숙박업)
C_WARN = "#FF6B6B"  # 경고 색(폐업/관광숙박업)
# 파이차트용 10색 팔레트 (TOP-N 동을 구분)
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

# PNG 출력 경로 — 없으면 생성
OUT_DIR = f"{BASE}/data/viz_each_gu"
os.makedirs(OUT_DIR, exist_ok=True)


def load_gu(path):
    """
    인허가 CSV 한 개를 로드하고 10개 자치구 데이터만 필터링해 반환.

    인자:
        path (str): 원본 CSV 경로

    반환:
        DataFrame: '인허가연도'(int) 와 '구'(str) 컬럼이 추가된, GU_10 에 속한 행만 남긴 사본
    """
    # 한글 BOM 포함 UTF-8 로 읽고, 컬럼별 dtype 추론을 메모리 절약 모드로
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    # '인허가일자' 문자열 → datetime → 연도(int)
    df["인허가연도"] = pd.to_datetime(df["인허가일자"], errors="coerce").dt.year
    # 지번주소에서 '서울특별시 OO구' 패턴으로 자치구만 추출
    df["구"] = df["지번주소"].str.extract(r"서울특별시\s+(\S+구)")
    # 10개 자치구 필터링 + 원본 보호용 사본 반환
    return df[df["구"].isin(GU_10)].copy()


# 3종 인허가 데이터 일괄 로드
f1 = load_gu(f"{BASE}/원본데이터/서울시 외국인관광도시민박업 인허가 정보.csv")
f2 = load_gu(f"{BASE}/원본데이터/서울시 관광숙박업 인허가 정보.csv")
f3 = load_gu(f"{BASE}/원본데이터/서울시 숙박업 인허가 정보.csv")

# 자치구 단위 반복 — 한 구에 대해 한 장의 종합 패널 PNG 생성
for GU in GU_10:
    # 외국인민박: 해당 구 + 동 정보 추출
    g1 = f1[f1["구"] == GU].copy()
    # 지번주소에서 'OO구 XX동' 또는 'OO구 X가' 패턴으로 동 단위 추출
    g1["동"] = g1["지번주소"].str.extract(r"%s\s+(\S+동|\S+가)" % GU)
    # 관광숙박업/일반숙박업 — 해당 구만
    g2 = f2[f2["구"] == GU]
    g3 = f3[f3["구"] == GU]

    # figure 생성 — 가로형 18x10, 배경색 적용
    fig = plt.figure(figsize=(18, 10), facecolor=BG)
    # 상단 메인 타이틀
    fig.suptitle(
        f"{GU} 숙박업 인허가 심층 분석 (2020–2025)",
        fontsize=18,
        fontweight="bold",
        y=0.98,
        color="#1A1A2E",
    )
    # 2행 4열 그리드 — 상단행은 4분할, 하단행은 1개(히트맵 전용)
    gs = GridSpec(
        2,
        4,
        figure=fig,
        hspace=0.45,
        wspace=0.38,
        left=0.06,
        right=0.97,
        top=0.91,
        bottom=0.09,
    )

    # [A] 누적 꺾은선 — 외국인민박 영업중 vs 폐업의 연도별 누적 추이
    ax = fig.add_subplot(gs[0, :2])  # 상단 좌측 절반(2칸 차지)
    ax.set_facecolor("white")
    누적_영업, 누적_폐업 = [], []
    # 각 연도 시점까지의 누적 영업/폐업 건수 계산
    for yr in YEARS:
        sub = g1[g1["인허가연도"] <= yr]
        누적_영업.append((sub["영업상태명"] == "영업/정상").sum())
        누적_폐업.append((sub["영업상태명"] == "폐업").sum())
    # 라인 아래 영역 채우기(시각적 강조)
    ax.fill_between(YEARS, 누적_영업, alpha=0.15, color=C_MAIN)
    ax.fill_between(YEARS, 누적_폐업, alpha=0.15, color=C_WARN)
    # 영업중 누적 라인 + 마커
    ax.plot(
        YEARS,
        누적_영업,
        "o-",
        color=C_MAIN,
        lw=2.8,
        ms=8,
        label="영업중 누적",
        zorder=4,
    )
    # 폐업 누적 라인 + 마커
    ax.plot(
        YEARS, 누적_폐업, "s-", color=C_WARN, lw=2.8, ms=8, label="폐업 누적", zorder=4
    )
    # 영업중 누적값 텍스트 (라인 위쪽)
    for x, v in zip(YEARS, 누적_영업):
        ax.text(
            x,
            v + max(누적_영업) * 0.02 + 0.5,
            str(v),
            ha="center",
            fontsize=9,
            fontweight="bold",
            color=C_MAIN,
        )
    # 폐업 누적값 텍스트 (라인 아래쪽 — offset 음수)
    for x, v in zip(YEARS, 누적_폐업):
        offset = -max(누적_영업) * 0.05 - 0.5
        ax.text(x, v + offset, str(v), ha="center", fontsize=8, color=C_WARN)
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
    # 위/오른쪽 테두리 제거(군더더기 정리)
    ax.spines[["top", "right"]].set_visible(False)

    # [B] 파이차트 — 가장 최신연도의 동별 신규 인허가 비중
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_facecolor("white")
    # 2020~2025 범위 필터링 후 최신 연도 식별
    yr_data = g1[g1["인허가연도"].between(2020, 2025)]
    latest_yr = int(yr_data["인허가연도"].max()) if len(yr_data) > 0 else 2025
    # 최신연도 동별 인허가 건수
    dong_cnt = g1[g1["인허가연도"] == latest_yr]["동"].value_counts()

    # 데이터 없을 때 안내 텍스트만 출력
    if len(dong_cnt) == 0:
        ax2.text(
            0,
            0,
            f"{latest_yr}년\n데이터 없음",
            ha="center",
            va="center",
            fontsize=12,
            color="#888",
        )
        ax2.set_title(
            f"{latest_yr}년 신규 인허가\n동별 비중",
            fontsize=11,
            fontweight="bold",
            pad=8,
        )
        ax2.axis("off")
    else:
        # 상위 7개 동 + 나머지는 '기타'로 묶기
        TOP_N = 7
        top_d = dong_cnt.head(TOP_N)
        other_d = dong_cnt.iloc[TOP_N:]
        other_cnt = other_d.sum()
        labels_pie = list(top_d.index) + (["기타"] if other_cnt > 0 else [])
        sizes_pie = list(top_d.values) + ([other_cnt] if other_cnt > 0 else [])
        # 파이차트 그리기 — 시작각 140°, 비율 텍스트 위치 0.75
        wedges, texts, autotexts = ax2.pie(
            sizes_pie,
            labels=labels_pie,
            autopct="%1.0f%%",
            colors=PAL[: len(labels_pie)],
            startangle=140,
            pctdistance=0.75,
            textprops={"fontsize": 8},
        )
        # 비율 텍스트 굵게/작게 강조
        for at in autotexts:
            at.set_fontweight("bold")
            at.set_fontsize(8)
        ax2.set_title(
            f"{latest_yr}년 신규 인허가\n동별 비중",
            fontsize=11,
            fontweight="bold",
            pad=8,
        )
        # '기타'로 묶인 동 이름들을 차트 아래 박스 안에 풀어서 표기
        if other_cnt > 0:
            other_names = ", ".join(other_d.index.tolist())
            ax2.text(
                0,
                -1.45,
                f"기타({int(other_cnt)}건): {other_names}",
                ha="center",
                va="top",
                fontsize=7,
                color="#555",
                bbox=dict(boxstyle="round,pad=0.3", fc="#F1F5F9", alpha=0.8),
            )

    # [C] 업종별 연도별 막대 — 외국인민박/일반숙박/관광숙박 비교
    ax3 = fig.add_subplot(gs[0, 3])
    ax3.set_facecolor("white")
    # 각 업종을 연도별 카운트로 집계 + 누락 연도는 0으로
    m1 = (
        g1[g1["인허가연도"].between(2020, 2025)]
        .groupby("인허가연도")
        .size()
        .reindex(YEARS, fill_value=0)
    )
    m2 = (
        g2[g2["인허가연도"].between(2020, 2025)]
        .groupby("인허가연도")
        .size()
        .reindex(YEARS, fill_value=0)
    )
    m3 = (
        g3[g3["인허가연도"].between(2020, 2025)]
        .groupby("인허가연도")
        .size()
        .reindex(YEARS, fill_value=0)
    )
    # 그룹 막대 — x 위치를 0,1,2,... 로 잡고 너비 0.25씩 옆으로 옮겨 3개 비교
    x = np.arange(len(YEARS))
    w = 0.25
    ax3.bar(x - w, m1.values, width=w, color=C_MAIN, alpha=0.85, label="외국인민박")
    ax3.bar(x, m3.values, width=w, color=C_LIVE, alpha=0.85, label="숙박업")
    ax3.bar(x + w, m2.values, width=w, color=C_WARN, alpha=0.85, label="관광숙박업")
    ax3.set_title(
        "업종별 신규 인허가\n연도 비교", fontsize=11, fontweight="bold", pad=8
    )
    ax3.set_xticks(x)
    ax3.set_xticklabels(YEARS, fontsize=7, rotation=30)
    ax3.legend(fontsize=7)
    ax3.grid(axis="y", alpha=0.25, linestyle="--")
    ax3.spines[["top", "right"]].set_visible(False)

    # [D] 동별 히트맵 — 외국인민박 동 × 연도 행렬 (상위 10개 동)
    ax4 = fig.add_subplot(gs[1, :])  # 하단 1행 4열 모두 차지
    ax4.set_facecolor("white")
    g1_filt = g1[g1["인허가연도"].between(2020, 2025)]
    if len(g1_filt) > 0 and g1_filt["동"].notna().any():
        # 동×연도 피벗 — 결측 연도/동은 0
        dong_piv = (
            g1_filt.groupby(["동", "인허가연도"])
            .size()
            .unstack(fill_value=0)
            .reindex(columns=YEARS, fill_value=0)
        )
        # 합계 기준 상위 10개 동만 추림
        top_n = min(10, len(dong_piv))
        top_d_idx = dong_piv.sum(axis=1).nlargest(top_n).index
        dong_piv = dong_piv.loc[top_d_idx]
        # 히트맵 — 색상 'YlOrRd' (밝은 노랑→짙은 빨강)
        im = ax4.imshow(dong_piv.values, aspect="auto", cmap="YlOrRd", vmin=0)
        ax4.set_xticks(range(len(YEARS)))
        ax4.set_xticklabels(YEARS, fontsize=10)
        ax4.set_yticks(range(len(top_d_idx)))
        ax4.set_yticklabels(top_d_idx, fontsize=10)
        # 각 셀에 숫자 라벨 — 값이 0이면 '·'(점)으로
        for i in range(len(top_d_idx)):
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
                    # 셀이 진할수록 흰 글자, 연하면 어두운 글자(가독성)
                    color="white" if v > dong_piv.values.max() * 0.55 else "#333",
                )
        # 컬러바 — 우측에 작게
        plt.colorbar(
            im, ax=ax4, orientation="vertical", shrink=0.9, label="건수", pad=0.01
        )
    else:
        # 동 데이터가 전혀 없을 때 안내 텍스트
        ax4.text(
            0.5,
            0.5,
            "동별 데이터 없음",
            ha="center",
            va="center",
            fontsize=13,
            color="#888",
            transform=ax4.transAxes,
        )
        ax4.axis("off")
    ax4.set_title(
        "동별 외국인관광도시민박업 신규 인허가 히트맵 (상위 10개 동)",
        fontsize=11,
        fontweight="bold",
        pad=8,
    )
    # 모든 테두리 제거 — 히트맵은 깔끔하게
    ax4.spines[["top", "right", "left", "bottom"]].set_visible(False)

    # PNG 저장 — 고해상도(150dpi), 여백 자동, 배경색 유지
    out = f"{OUT_DIR}/{GU}_인허가분석.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()  # 메모리 누수 방지 — figure 닫기
    print(f"저장: {GU}_인허가분석.png")

print("\n전체 완료!")
