# -*- coding: utf-8 -*-
"""
구별 최근접 소화용수 거리등급 구성을 인터랙티브 HTML(Plotly)로 시각화한다.

목적:
    - 분석변수_최종테이블0428.csv 의 거리등급(0/1/2)을 구별로 집계
    - 누적 막대(좌) + 평균등급 막대(우) 의 1×2 패널 HTML 생성, hover 로 상세 정보 노출

입력:
    - NJT-PJT/0424/data/분석변수_최종테이블0428.csv

출력:
    - NJT-PJT/0424/data/구별_소화용수_거리등급_시각화.html

처리 흐름:
    1) CSV 로딩 + 거리등급 정수화
    2) 구×등급 시설수 / 비율 / 평균등급 / 20m 이내 / 40m 이내 비율 집계
    3) Plotly subplots(1×2): 누적 가로 막대 + 평균등급 막대 + hover 템플릿
    4) 레이아웃/주석 정리 후 HTML 저장 (CDN plotly.js)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
# Plotly 인터랙티브 차트
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ── 경로 상수 ─────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parents[1]
SRC = BASE / "0424" / "data" / "분석변수_최종테이블0428.csv"
OUT = BASE / "0424" / "data" / "구별_소화용수_거리등급_시각화.html"

# 거리등급 0~2 한글 라벨
GRADE_LABELS = {
    0: "20m 이내",
    1: "20~40m",
    2: "40m 초과",
}

# 등급별 색
COLORS = {
    0: "#16A34A",
    1: "#F59E0B",
    2: "#DC2626",
}


def main() -> None:
    """집계 → Plotly 1×2 패널 → HTML 저장."""
    # CSV 로딩 + 거리등급 정수형 변환
    df = pd.read_csv(SRC, encoding="utf-8-sig")
    df["최근접_소화용수_거리등급"] = pd.to_numeric(
        df["최근접_소화용수_거리등급"], errors="coerce"
    )
    df = df.dropna(subset=["구", "최근접_소화용수_거리등급"]).copy()
    df["최근접_소화용수_거리등급"] = df["최근접_소화용수_거리등급"].astype(int)

    # 구×등급별 시설수, 전체대비 비율
    counts = (
        df.groupby(["구", "최근접_소화용수_거리등급"])
        .size()
        .rename("시설수")
        .reset_index()
    )
    total = df.groupby("구").size().rename("전체시설수")
    counts = counts.merge(total, on="구")
    counts["비율"] = counts["시설수"] / counts["전체시설수"] * 100

    # 비율 피벗 (구×등급)
    pivot_pct = (
        counts.pivot(index="구", columns="최근접_소화용수_거리등급", values="비율")
        .fillna(0)
        .reindex(columns=[0, 1, 2], fill_value=0)
    )
    # hover 표시용 시설수 피벗
    pivot_count = (
        counts.pivot(index="구", columns="최근접_소화용수_거리등급", values="시설수")
        .fillna(0)
        .reindex(columns=[0, 1, 2], fill_value=0)
        .astype(int)
    )
    # 정렬 기준: 등급0 → 등급1 비율
    gu_order = pivot_pct.sort_values([0, 1], ascending=[True, True]).index.tolist()

    # 구별 평균등급 / 시설수 / 20m·40m 이내 비율 요약
    summary = (
        df.groupby("구")
        .agg(
            평균등급=("최근접_소화용수_거리등급", "mean"),
            시설수=("숙소명", "size"),
            이십미터이내비율=(
                "최근접_소화용수_거리등급",
                lambda s: s.eq(0).mean() * 100,
            ),
            사십미터이내비율=(
                "최근접_소화용수_거리등급",
                lambda s: s.le(1).mean() * 100,
            ),
        )
        .reindex(gu_order)
        .reset_index()
    )

    # ── Plotly subplots: 좌(누적 막대) + 우(평균등급) ─────────────────
    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.68, 0.32],
        horizontal_spacing=0.08,
        subplot_titles=("구별 최근접 소화용수 거리등급 구성", "평균 거리등급"),
    )

    # 등급 2 → 1 → 0 순서로 추가 (스택 시 빨강이 가장 안쪽으로 가도록)
    for grade in [2, 1, 0]:
        # hover 에 띄울 [등급별 시설수, 전체 시설수, 비율]
        custom = [
            [
                int(pivot_count.loc[gu, grade]),
                int(total.loc[gu]),
                pivot_pct.loc[gu, grade],
            ]
            for gu in gu_order
        ]
        fig.add_trace(
            go.Bar(
                y=gu_order,
                x=pivot_pct.loc[gu_order, grade],
                name=GRADE_LABELS[grade],
                orientation="h",
                marker=dict(color=COLORS[grade], line=dict(color="white", width=0.8)),
                customdata=custom,
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    + GRADE_LABELS[grade]
                    + "<br>시설수: %{customdata[0]:,} / %{customdata[1]:,}<br>"
                    + "비율: %{customdata[2]:.1f}%<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )

    # ── 우패널: 평균등급 (값에 따른 컬러스케일) ───────────────────────
    fig.add_trace(
        go.Bar(
            y=summary["구"],
            x=summary["평균등급"],
            orientation="h",
            marker=dict(
                color=summary["평균등급"],
                colorscale=[
                    [0.0, "#16A34A"],
                    [0.5, "#F59E0B"],
                    [1.0, "#DC2626"],
                ],
                cmin=0,
                cmax=2,
                line=dict(color="white", width=0.8),
            ),
            text=summary["평균등급"].map(lambda v: f"{v:.2f}"),
            textposition="outside",
            customdata=summary[["시설수", "이십미터이내비율", "사십미터이내비율"]],
            hovertemplate=(
                "<b>%{y}</b><br>"
                "평균등급: %{x:.2f}<br>"
                "시설수: %{customdata[0]:,}<br>"
                "20m 이내: %{customdata[1]:.1f}%<br>"
                "40m 이내: %{customdata[2]:.1f}%<extra></extra>"
            ),
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    # ── 레이아웃 ──────────────────────────────────────────────────────
    fig.update_layout(
        title=dict(
            text="<b>구별 소화용수 접근성 등급</b><br><sup>0=20m 이내, 1=20~40m, 2=40m 초과 | 낮을수록 가까움</sup>",
            x=0.02,
            xanchor="left",
            font=dict(size=24, color="#0F172A"),
        ),
        barmode="stack",  # 좌패널 누적 막대
        width=1280,
        height=760,
        template="plotly_white",
        paper_bgcolor="#F8FAFC",
        plot_bgcolor="#FFFFFF",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.04,
            xanchor="left",
            x=0,
            font=dict(size=13),
        ),
        margin=dict(l=80, r=50, t=120, b=70),
        font=dict(
            family="Malgun Gothic, Apple SD Gothic Neo, Arial", size=13, color="#1E293B"
        ),
    )
    # 좌패널 x: 0~100% / 우패널 x: 0~2.25
    fig.update_xaxes(
        title_text="구성비 (%)", range=[0, 100], ticksuffix="%", row=1, col=1
    )
    fig.update_xaxes(title_text="평균 등급", range=[0, 2.25], row=1, col=2)
    # y축은 위→아래 정렬 유지 (autorange="reversed")
    fig.update_yaxes(title_text=None, autorange="reversed", row=1, col=1)
    fig.update_yaxes(
        title_text=None, autorange="reversed", showticklabels=False, row=1, col=2
    )
    # 하단 캡션
    fig.add_annotation(
        text="초록 비중이 클수록 소화용수 접근성이 좋고, 빨강 비중이 클수록 최근접 소화용수가 40m를 초과합니다.",
        x=0,
        y=-0.11,
        xref="paper",
        yref="paper",
        showarrow=False,
        align="left",
        font=dict(size=13, color="#475569"),
    )

    # CDN plotly.js 사용한 자체 완결 HTML 저장
    fig.write_html(OUT, include_plotlyjs="cdn", full_html=True)
    print(f"saved={OUT}")
    # 평균등급 정렬 출력 (디버그용)
    print(summary.sort_values("평균등급").to_string(index=False))


if __name__ == "__main__":
    main()
