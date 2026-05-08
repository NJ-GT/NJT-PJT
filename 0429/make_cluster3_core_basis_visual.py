# -*- coding: utf-8 -*-
"""
K=3 군집화 핵심 근거 시각화 스크립트.

목적:
    cluster3 파이프라인 산출물에서 클러스터별 변수 평균을 계산하고,
    "전체 평균 대비 차이(%)"를 색으로, 원자료 평균을 숫자로 표기하는
    히트맵을 생성하여 군집 해석 근거 자료로 활용한다.

산출물:
    NJT-PJT/0429/군집3개_핵심근거_프로파일히트맵_0429.png
    NJT-PJT/0429/군집3개_핵심근거_변수평균표_0429.csv
"""

# 경로 처리 유틸
from pathlib import Path

# 백엔드 지정을 위해 matplotlib 본체 먼저 import
import matplotlib

# GUI 없는 서버/배치 환경에서도 PNG 저장 가능하도록 Agg 백엔드 강제
matplotlib.use("Agg")
# 이 import는 반드시 use("Agg") 호출 이후에 와야 함
import matplotlib.pyplot as plt
# 수치 연산 (np.isclose 등)
import numpy as np
# 표 데이터 처리
import pandas as pd
# 히트맵/팔레트
import seaborn as sns


# 0429 폴더 기준 한 단계 위 (NJT-PJT/)
BASE = Path(__file__).resolve().parents[1]
# 입력 CSV가 들어 있는 디렉터리
DATA_DIR = BASE / "0424" / "data" / "cluster3_spatial_pipeline_fire_count_150m_0428"
# 산출물 저장 폴더 (PNG, CSV)
OUT_DIR = BASE / "0429"


# 히트맵 행/열로 사용할 변수 목록 (분석 시 핵심으로 보고 있는 7+1개)
FEATURES = [
    "구조노후도",
    "단속위험도",
    "도로폭위험도",
    "집중도",
    "주변건물수",
    "최근접_소화용수_거리등급",
    "소방위험도_점수",
    "최종_화재위험점수",
]

# 그래프에 표시될 라벨(한글 줄바꿈 포함) — 가독성을 위한 매핑
FEATURE_LABELS = {
    "구조노후도": "구조\n노후도",
    "단속위험도": "단속\n위험도",
    "도로폭위험도": "도로폭\n위험도",
    "집중도": "시설\n집중도",
    "주변건물수": "주변\n건물수",
    "최근접_소화용수_거리등급": "소화용수\n거리등급",
    "소방위험도_점수": "소방\n위험도",
    "최종_화재위험점수": "최종\n위험점수",
}

# 우측 패널에 표기할 군집별 정성적 해석 메모
CLUSTER_MEMO = {
    0: "밀집도는 높지만\n소화용수 접근성 양호",
    1: "저밀도이나\n소화용수 접근 취약",
    2: "밀집·도로폭·최종위험\n동시 고위험",
}


def set_korean_font() -> None:
    """matplotlib에서 한글이 깨지지 않도록 윈도우 기본 한글 폰트를 설정."""
    # 음수 부호가 깨지는 현상 방지 (마이너스 부호를 ASCII로)
    plt.rcParams["axes.unicode_minus"] = False
    # 윈도우에 기본 설치된 한글 폰트
    plt.rcParams["font.family"] = "Malgun Gothic"


def read_data() -> pd.DataFrame:
    """DATA_DIR에서 가장 큰 CSV를 입력으로 가정하고 필요한 컬럼만 추출."""
    # 파일 크기 내림차순 정렬 -> 가장 완전한 변수 테이블 선택
    csv_files = sorted(
        DATA_DIR.glob("*.csv"), key=lambda p: p.stat().st_size, reverse=True
    )
    if not csv_files:
        # 입력이 없으면 명시적 오류 (조용한 빈 결과 방지)
        raise FileNotFoundError(f"No CSV files found in {DATA_DIR}")
    # UTF-8 BOM 포함 가능성 대비 utf-8-sig 로드
    df = pd.read_csv(csv_files[0], encoding="utf-8-sig")
    # cluster + 핵심 변수 모두 있어야 의미 있는 결과
    needed = ["cluster", *FEATURES]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")
    # 필요한 컬럼만 추리고 결측 행은 제거 (평균 계산 신뢰성 확보)
    return df[needed].dropna().copy()


def minmax_by_feature(cluster_means: pd.DataFrame) -> pd.DataFrame:
    """클러스터 평균 표를 '전체 평균 대비 차이(%)'로 변환 — 색 지표용.

    - 변수 전체 변동이 매우 작거나 평균이 0이면 0(중립)으로 처리해
      과도하게 강조되지 않도록 한다.
    - 차이는 ±80%로 클립 (극단값으로 색이 포화되는 것 방지).
    """
    scored = cluster_means.copy()
    # 컬럼(변수) 단위로 처리
    for col in scored.columns:
        # 전체 클러스터에 걸친 평균 (분모 역할)
        mean = scored[col].mean()
        # 변동 폭 (max - min)
        value_range = scored[col].max() - scored[col].min()
        # 평균이 0이거나, 변동 폭이 평균 대비 5% 미만이면 차이가 미미하다고 판단
        if np.isclose(mean, 0) or value_range / abs(mean) < 0.05:
            scored[col] = 0
        else:
            # 그 외엔 ((값 - 평균) / |평균|) * 100 으로 % 차이 계산
            pct_diff = (scored[col] - mean) / abs(mean) * 100
            # ±80% 범위로 클립
            scored[col] = pct_diff.clip(-80, 80)
    return scored


def main() -> None:
    """히트맵 + 메모 패널을 한 PNG로 합쳐 저장."""
    # 한글 폰트 설정
    set_korean_font()
    # 입력 데이터 로드
    df = read_data()

    # 군집별 표본 수 (라벨에 표기)
    counts = df.groupby("cluster").size()
    # 군집별 변수 평균 (정렬된 cluster index)
    means = df.groupby("cluster")[FEATURES].mean().sort_index()
    # 색용 % 차이 행렬 생성
    relative = minmax_by_feature(means)
    # 컬럼/인덱스를 한글 라벨로 교체 (시각화 가독성 ↑)
    relative.columns = [FEATURE_LABELS[c] for c in relative.columns]
    relative.index = [
        f"Cluster {idx}\n(n={counts.loc[idx]:,})" for idx in relative.index
    ]

    # 산출 파일 경로
    out_png = OUT_DIR / "군집3개_핵심근거_프로파일히트맵_0429.png"
    out_csv = OUT_DIR / "군집3개_핵심근거_변수평균표_0429.csv"
    # 군집별 변수 평균 표는 그대로 CSV로 보존 (분석 재현용)
    means.round(4).to_csv(out_csv, encoding="utf-8-sig")

    # Figure 구성: 좌측은 히트맵, 우측은 해석 메모 (4.8:1.7 가로 비율)
    fig = plt.figure(figsize=(14.5, 8.2), dpi=180)
    gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[4.8, 1.7], wspace=0.07)
    ax = fig.add_subplot(gs[0, 0])
    ax_note = fig.add_subplot(gs[0, 1])

    # 히트맵: 색 = 차이(%), 셀 텍스트 = 원자료 평균값
    sns.heatmap(
        relative,
        ax=ax,
        # 발산형(blue↔red) 팔레트 — 0 중심
        cmap=sns.diverging_palette(220, 20, as_cmap=True),
        center=0,
        # 색 스케일을 ±80%로 통일
        vmin=-80,
        vmax=80,
        # 셀 표시값은 원자료 평균(소수 둘째자리)
        annot=means.rename(columns=FEATURE_LABELS).round(2),
        fmt=".2f",
        # 셀 사이 흰 선으로 가독성 ↑
        linewidths=1.2,
        linecolor="white",
        # 컬러바 라벨
        cbar_kws={"label": "전체 평균 대비 차이(%) · 차이 작으면 0"},
        # 셀 숫자 스타일
        annot_kws={"fontsize": 10, "weight": "bold"},
    )

    # 히트맵 상단 제목
    ax.set_title(
        "K=3 군집화 핵심 근거: 변수 평균 프로파일", fontsize=19, weight="bold", pad=18
    )
    # x/y 축 라벨 비표시 (틱 자체가 라벨 역할)
    ax.set_xlabel("")
    ax.set_ylabel("")
    # 틱 레이블 스타일
    ax.tick_params(axis="x", labelsize=10, rotation=0)
    ax.tick_params(axis="y", labelsize=11, rotation=0)

    # 우측 메모 패널은 축 라인을 끄고 텍스트만 배치
    ax_note.axis("off")
    # 메모 패널 제목
    ax_note.text(
        0,
        0.98,
        "군집 해석 요약",
        fontsize=15,
        weight="bold",
        va="top",
        transform=ax_note.transAxes,
    )

    # 메모 항목 시작 y 좌표 (axes 좌표계 기준 — 0~1)
    y = 0.82
    # 군집별 색상 (히트맵 외 별도 식별 색)
    colors = {0: "#66c2a5", 1: "#fc8d62", 2: "#8da0cb"}
    # 군집 번호 정렬해서 위→아래 순서로 메모 출력
    for cluster in sorted(CLUSTER_MEMO):
        # 색상 사각형(작은 칩) 추가
        ax_note.add_patch(
            plt.Rectangle(
                (0, y - 0.035),
                0.055,
                0.055,
                color=colors[cluster],
                transform=ax_note.transAxes,
                clip_on=False,
            )
        )
        # 군집 라벨
        ax_note.text(
            0.075,
            y,
            f"Cluster {cluster}",
            fontsize=12,
            weight="bold",
            va="center",
            transform=ax_note.transAxes,
        )
        # 군집 해석 메모(여러 줄 가능)
        ax_note.text(
            0.075,
            y - 0.08,
            CLUSTER_MEMO[cluster],
            fontsize=11,
            color="#283747",
            va="top",
            linespacing=1.35,
            transform=ax_note.transAxes,
        )
        # 다음 항목 위치로 한 칸 내림
        y -= 0.27

    # 패널 하단의 보조 설명 (시각화 해석 가이드)
    ax_note.text(
        0,
        0.04,
        "숫자: 각 군집의 원자료 평균\n색: 전체 평균 대비 차이\n구조노후도처럼 차이가 미미하면 중립색 처리",
        fontsize=9.5,
        color="#5f6b7a",
        va="bottom",
        linespacing=1.35,
        transform=ax_note.transAxes,
    )

    # 배경을 흰색으로 강제 — 인쇄/캡쳐 시 투명 배경 회피
    fig.patch.set_facecolor("white")
    # PNG로 저장 (여백 자동 조정)
    fig.savefig(out_png, bbox_inches="tight", facecolor="white")
    # 메모리 정리 (배치 처리 시 필수)
    plt.close(fig)
    # 산출 경로 출력 (CLI 확인용)
    print(out_png)
    print(out_csv)


# 직접 실행 시에만 main 호출
if __name__ == "__main__":
    main()
