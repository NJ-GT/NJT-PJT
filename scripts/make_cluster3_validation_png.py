# -*- coding: utf-8 -*-
"""
K=3 군집 타당성 평가 결과를 발표용 카드 PNG 한 장으로 정리한다.

목적:
    - cluster3_validation_indices.csv 의 risk 9변수 + MinMax 기준 행만 사용
    - Calinski-Harabasz / Silhouette / Davies-Bouldin 세 지수 카드 + 군집 구성 + 해석 스트립
    - 헤더 띠, 카드 그림자, 라운드 박스 등 정적 발표용 디자인

입력:
    - 0424/data/cluster3_spatial_pipeline_fire_count_150m_0428/cluster3_validation_indices.csv

출력:
    - 같은 폴더의 cluster3_validation_indices_presentation.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
# 라운드 박스 / 사각형 패치
from matplotlib.patches import FancyBboxPatch, Rectangle


# 경로 상수
BASE = Path(__file__).resolve().parents[1]
DIR = BASE / "0424" / "data" / "cluster3_spatial_pipeline_fire_count_150m_0428"
SRC = DIR / "cluster3_validation_indices.csv"
OUT = DIR / "cluster3_validation_indices_presentation.png"


def add_card(ax, x, y, w, h, title, value, hint, color):
    """발표용 지수 카드 1장(그림자 + 라운드 박스 + 색 칩 + 값 + 부제) 그리기.

    파라미터:
        x, y, w, h : axes 좌표계 비율(0~1)
        title      : 카드 상단 제목 (예: "Silhouette Score")
        value      : 큰 숫자 (예: "0.341")
        hint       : 부제 (해석 도움말)
        color      : 강조색 (값/색칩 컬러)
    """
    # 그림자(반투명 다크)
    shadow = FancyBboxPatch(
        (x + 0.006, y - 0.008),
        w,
        h,
        boxstyle="round,pad=0.015,rounding_size=0.025",
        transform=ax.transAxes,
        facecolor="#0F172A",
        edgecolor="none",
        alpha=0.08,
        zorder=2,
    )
    # 본 카드
    card = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.015,rounding_size=0.025",
        transform=ax.transAxes,
        facecolor="#FFFFFF",
        edgecolor="#E2E8F0",
        linewidth=1.2,
        zorder=3,
    )
    ax.add_patch(shadow)
    ax.add_patch(card)
    # 좌상단 색 칩 (강조)
    ax.add_patch(
        FancyBboxPatch(
            (x + 0.018, y + h - 0.075),
            0.055,
            0.045,
            boxstyle="round,pad=0.008,rounding_size=0.012",
            transform=ax.transAxes,
            facecolor=color,
            edgecolor="none",
            zorder=4,
        )
    )
    # 카드 제목
    ax.text(
        x + 0.088,
        y + h - 0.048,
        title,
        transform=ax.transAxes,
        fontsize=16,
        color="#0F172A",
        weight="bold",
        va="center",
        zorder=5,
    )
    # 큰 숫자 (콤마 포함이면 살짝 작게)
    ax.text(
        x + 0.03,
        y + 0.165,
        value,
        transform=ax.transAxes,
        fontsize=35 if "," in value else 42,
        color=color,
        weight="bold",
        va="center",
        zorder=5,
    )
    # 해석 부제
    ax.text(
        x + 0.032,
        y + 0.075,
        hint,
        transform=ax.transAxes,
        fontsize=12.5,
        color="#64748B",
        va="center",
        zorder=5,
    )


def main() -> None:
    """검증 csv 로딩 → 발표용 카드 PNG 생성."""
    # risk 9변수 행만 선택
    res = pd.read_csv(SRC, encoding="utf-8-sig")
    row = res[res["basis"].str.contains("risk 9", na=False)].iloc[0]

    # 카드별 표시 값 / 해석 / 색
    values = {
        "Calinski-Harabasz\nIndex": f"{float(row['calinski_harabasz_index']):,.3f}",
        "Silhouette\nScore": f"{float(row['silhouette_score']):.3f}",
        "Davies-Bouldin\nIndex": f"{float(row['davies_bouldin_index']):.3f}",
    }
    hints = {
        "Calinski-Harabasz\nIndex": "높을수록 군집 간 분리가 좋음",
        "Silhouette\nScore": "-1~1 범위, 높을수록 좋음",
        "Davies-Bouldin\nIndex": "낮을수록 군집 중첩이 적음",
    }
    colors = {
        "Calinski-Harabasz\nIndex": "#2563EB",
        "Silhouette\nScore": "#0F766E",
        "Davies-Bouldin\nIndex": "#F97316",
    }
    # 군집 구성 (사전 계산값)
    counts = {0: 1458, 1: 1238, 2: 1550}

    # 한글 폰트
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    # 캔버스 (단일 ax 에 모든 요소 직접 배치)
    fig = plt.figure(figsize=(15.5, 8.7), dpi=180)
    fig.patch.set_facecolor("#EEF3F8")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    # 상단 짙은 헤더 띠 + 하단 하늘색 띠
    ax.add_patch(
        Rectangle((0, 0.78), 1, 0.22, transform=ax.transAxes, color="#0F172A", zorder=0)
    )
    ax.add_patch(
        Rectangle(
            (0, 0.765),
            1,
            0.015,
            transform=ax.transAxes,
            color="#38BDF8",
            zorder=0,
            alpha=0.8,
        )
    )
    # 헤더 텍스트
    ax.text(
        0.055,
        0.91,
        "K=3 군집 타당성 평가",
        transform=ax.transAxes,
        fontsize=31,
        color="white",
        weight="bold",
        va="center",
    )
    ax.text(
        0.057,
        0.845,
        "Calinski-Harabasz · Silhouette · Davies-Bouldin | 위험 변수 9개 + MinMaxScaler 기준",
        transform=ax.transAxes,
        fontsize=13.5,
        color="#CBD5E1",
        va="center",
    )

    # 본 패널 (큰 라운드 박스)
    panel = FancyBboxPatch(
        (0.045, 0.08),
        0.91,
        0.64,
        boxstyle="round,pad=0.014,rounding_size=0.026",
        transform=ax.transAxes,
        facecolor="white",
        edgecolor="#D8E2EF",
        linewidth=1.3,
        zorder=1,
    )
    ax.add_patch(panel)

    # 3개 카드 가로 배치
    xs = [0.075, 0.365, 0.655]
    for x, title in zip(xs, values):
        add_card(
            ax, x, 0.32, 0.255, 0.34, title, values[title], hints[title], colors[title]
        )

    # 군집 구성 헤더
    ax.text(
        0.075,
        0.245,
        "군집 구성",
        transform=ax.transAxes,
        fontsize=16,
        color="#0F172A",
        weight="bold",
    )
    ax.text(
        0.075,
        0.213,
        "총 4,246개 숙박시설 · 3개 군집",
        transform=ax.transAxes,
        fontsize=12.5,
        color="#64748B",
    )

    # 군집 비율 가로 막대 (구성비)
    bar_x, bar_y, bar_w, bar_h = 0.32, 0.205, 0.56, 0.045
    total = sum(counts.values())
    start = bar_x
    cluster_colors = ["#60A5FA", "#34D399", "#FBBF24"]
    for idx, (cid, cnt) in enumerate(counts.items()):
        width = bar_w * cnt / total
        # 색 채우기
        ax.add_patch(
            Rectangle(
                (start, bar_y),
                width,
                bar_h,
                transform=ax.transAxes,
                color=cluster_colors[idx],
                zorder=4,
            )
        )
        # 라벨 (Cn 1,234)
        ax.text(
            start + width / 2,
            bar_y + bar_h / 2,
            f"C{cid} {cnt:,}",
            transform=ax.transAxes,
            fontsize=11.5,
            color="#0F172A",
            weight="bold",
            ha="center",
            va="center",
            zorder=5,
        )
        start += width
    # 막대 외곽선
    ax.add_patch(
        Rectangle(
            (bar_x, bar_y),
            bar_w,
            bar_h,
            transform=ax.transAxes,
            fill=False,
            edgecolor="#CBD5E1",
            linewidth=1.0,
            zorder=6,
        )
    )

    # 하단 해석 스트립
    strip = FancyBboxPatch(
        (0.075, 0.115),
        0.83,
        0.055,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        transform=ax.transAxes,
        facecolor="#F8FAFC",
        edgecolor="#E2E8F0",
        linewidth=1.0,
        zorder=3,
    )
    ax.add_patch(strip)
    ax.text(
        0.095,
        0.143,
        "해석",
        transform=ax.transAxes,
        fontsize=12.5,
        color="#0F172A",
        weight="bold",
        va="center",
    )
    ax.text(
        0.145,
        0.143,
        "CH는 군집 간 분리도, Silhouette은 응집·분리 균형, DB는 군집 간 중첩 정도를 보여준다.",
        transform=ax.transAxes,
        fontsize=12.2,
        color="#475569",
        va="center",
    )

    # 저장
    fig.savefig(OUT, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"saved={OUT}")


if __name__ == "__main__":
    main()
