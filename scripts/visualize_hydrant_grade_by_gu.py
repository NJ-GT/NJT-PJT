# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


BASE = Path(__file__).resolve().parents[1]
SRC = BASE / "0424" / "data" / "분석변수_최종테이블0428.csv"
OUT = BASE / "0424" / "data" / "구별_소화용수_거리등급_시각화.html"

GRADE_LABELS = {
    0: "20m 이내",
    1: "20~40m",
    2: "40m 초과",
}

COLORS = {
    0: "#16A34A",
    1: "#F59E0B",
    2: "#DC2626",
}


def main() -> None:
    df = pd.read_csv(SRC, encoding="utf-8-sig")
    df["최근접_소화용수_거리등급"] = pd.to_numeric(df["최근접_소화용수_거리등급"], errors="coerce")
    df = df.dropna(subset=["구", "최근접_소화용수_거리등급"]).copy()
    df["최근접_소화용수_거리등급"] = df["최근접_소화용수_거리등급"].astype(int)

    counts = (
        df.groupby(["구", "최근접_소화용수_거리등급"])
        .size()
        .rename("시설수")
        .reset_index()
    )
    total = df.groupby("구").size().rename("전체시설수")
    counts = counts.merge(total, on="구")
    counts["비율"] = counts["시설수"] / counts["전체시설수"] * 100

    pivot_pct = (
        counts.pivot(index="구", columns="최근접_소화용수_거리등급", values="비율")
        .fillna(0)
        .reindex(columns=[0, 1, 2], fill_value=0)
    )
    pivot_count = (
        counts.pivot(index="구", columns="최근접_소화용수_거리등급", values="시설수")
        .fillna(0)
        .reindex(columns=[0, 1, 2], fill_value=0)
        .astype(int)
    )
    gu_order = pivot_pct.sort_values([0, 1], ascending=[True, True]).index.tolist()

    summary = (
        df.groupby("구")
        .agg(
            평균등급=("최근접_소화용수_거리등급", "mean"),
            시설수=("숙소명", "size"),
            이십미터이내비율=("최근접_소화용수_거리등급", lambda s: (s.eq(0).mean() * 100)),
            사십미터이내비율=("최근접_소화용수_거리등급", lambda s: (s.le(1).mean() * 100)),
        )
        .reindex(gu_order)
        .reset_index()
    )

    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.68, 0.32],
        horizontal_spacing=0.08,
        subplot_titles=("구별 최근접 소화용수 거리등급 구성", "평균 거리등급"),
    )

    for grade in [2, 1, 0]:
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

    fig.update_layout(
        title=dict(
            text="<b>구별 소화용수 접근성 등급</b><br><sup>0=20m 이내, 1=20~40m, 2=40m 초과 | 낮을수록 가까움</sup>",
            x=0.02,
            xanchor="left",
            font=dict(size=24, color="#0F172A"),
        ),
        barmode="stack",
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
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, Arial", size=13, color="#1E293B"),
    )
    fig.update_xaxes(title_text="구성비 (%)", range=[0, 100], ticksuffix="%", row=1, col=1)
    fig.update_xaxes(title_text="평균 등급", range=[0, 2.25], row=1, col=2)
    fig.update_yaxes(title_text=None, autorange="reversed", row=1, col=1)
    fig.update_yaxes(title_text=None, autorange="reversed", showticklabels=False, row=1, col=2)
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

    fig.write_html(OUT, include_plotlyjs="cdn", full_html=True)
    print(f"saved={OUT}")
    print(summary.sort_values("평균등급").to_string(index=False))


if __name__ == "__main__":
    main()
