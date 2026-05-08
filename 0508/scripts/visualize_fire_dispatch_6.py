# -*- coding: utf-8 -*-
"""
화재출동 CSV 로부터 인터랙티브 6종 시각화(Plotly + Folium HTML) 생성 + 인덱스 페이지.

PNG 정적 버전(visualize_fire_dispatch_6_png.py)의 인터랙티브 사촌:
    - 01: Plotly 라인 — 월별 추세 (rangeslider 포함)
    - 02: Plotly 히트맵 — 요일×시간대
    - 03: Folium 지도 — 구별 발생 + 화재 밀도 히트맵
    - 04: Plotly 가로 막대 — 발화요인 TOP10
    - 05: Plotly 산점도 — 출동시간 vs 현장거리 + 추세선
    - 06: Folium MarkerCluster — 재산피해 TOP100 + 우상단 TOP20 표

부가:
    - summary.json — 행수/연도 범위/구별 분포/원인 TOP10
    - index.html   — 6개 HTML 링크를 카드 그리드로 표시

사용:
    python scripts/visualize_fire_dispatch_6.py [--csv PATH] [--out DIR] [--start-year N] [--end-year N]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from branca.colormap import linear  # 컬러맵 (구별 색)
from folium.plugins import HeatMap, MarkerCluster


# 기본 경로
BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CSV = BASE_DIR / "data" / "화재출동" / "화재출동_2021_2024.csv"
DEFAULT_OUT = BASE_DIR / "reports" / "fire_visualizations"

# 요일 정렬 순서
WEEKDAY_ORDER = ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"]
# Plotly 공통 옵션 — 로고 숨김 + 반응형
PLOTLY_CONFIG = {"displaylogo": False, "responsive": True}
# 일관된 색상 팔레트
COLOR_SEQUENCE = ["#2563eb", "#dc2626", "#059669", "#f59e0b", "#7c3aed", "#0891b2"]


def read_csv_robust(path: Path) -> pd.DataFrame:
    """다중 인코딩 시도 — utf-8-sig → cp949 → euc-kr → utf-8."""
    for encoding in ("utf-8-sig", "cp949", "euc-kr", "utf-8"):
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, low_memory=False)


def clean_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """컬럼명/문자열 컬럼의 공백 정리 + 빈 문자열 NaN 통일."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype("string").str.strip()
        df[col] = df[col].replace("", pd.NA)
    return df


def prepare_data(
    csv_path: Path, start_year: int | None, end_year: int | None
) -> pd.DataFrame:
    """원본 로드 + 숫자형 변환 + 날짜/파생 컬럼 + 연도 필터."""
    df = clean_text_columns(read_csv_robust(csv_path))

    numeric_cols = [
        "발생연도",
        "발생월",
        "발생일",
        "발생시",
        "발생분",
        "사망자수",
        "부상자수",
        "인명피해계",
        "재산피해액(천원)",
        "출동소요시간",
        "진압소요시간",
        "경도",
        "위도",
        "현장거리(km)",
        "안전센터_현장거리(km)",
        "출동대_현장거리(km)",
        "기온(℃)",
        "강수량(mm)",
        "풍속(m/s)",
        "습도(%)",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 발생일자 → datetime + 월 단위 + 분 단위 + 백만원 단위
    date_text = (
        df["발생일자"]
        .astype("string")
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(8)
    )
    df["발생일자_dt"] = pd.to_datetime(date_text, format="%Y%m%d", errors="coerce")
    df["발생연월"] = df["발생일자_dt"].dt.to_period("M").dt.to_timestamp()
    df["출동소요시간_분"] = df["출동소요시간"] / 60
    df["진압소요시간_분"] = df["진압소요시간"] / 60
    df["재산피해액_백만원"] = df["재산피해액(천원)"] / 1000

    # 연도 범위 옵션 필터
    if start_year is not None:
        df = df[df["발생연도"] >= start_year]
    if end_year is not None:
        df = df[df["발생연도"] <= end_year]

    return df.reset_index(drop=True)


def write_plotly(fig: go.Figure, out_path: Path) -> Path:
    """Plotly figure 공통 레이아웃 적용 후 단독 HTML 저장 (plotly.js 포함)."""
    fig.update_layout(
        template="plotly_white",
        font=dict(
            family="Malgun Gothic, Apple SD Gothic Neo, Arial, sans-serif", size=13
        ),
        title=dict(x=0.02, xanchor="left"),
        margin=dict(l=48, r=28, t=72, b=48),
        colorway=COLOR_SEQUENCE,
    )
    # plotly.js 를 HTML 안에 임베드 — 오프라인 열림
    pio.write_html(
        fig, out_path, include_plotlyjs=True, full_html=True, config=PLOTLY_CONFIG
    )
    return out_path


def viz_01_monthly_trend(df: pd.DataFrame, out_dir: Path) -> Path:
    """01: Plotly 라인 — 월별 화재 추세 + rangeslider."""
    monthly = (
        df.dropna(subset=["발생연월"])
        .groupby("발생연월")
        .size()
        .reset_index(name="화재건수")
        .sort_values("발생연월")
    )
    fig = px.line(
        monthly,
        x="발생연월",
        y="화재건수",
        markers=True,
        title="연도-월별 화재 발생 추세",
        labels={"발생연월": "발생 연월", "화재건수": "화재 건수"},
    )
    fig.update_traces(
        line=dict(width=3, color="#2563eb"), marker=dict(size=7, color="#dc2626")
    )
    # 하단 미니 슬라이더로 기간 줌인 가능
    fig.update_xaxes(rangeslider_visible=True)
    return write_plotly(fig, out_dir / "01_year_month_fire_trend.html")


def viz_02_weekday_hour_heatmap(df: pd.DataFrame, out_dir: Path) -> Path:
    """02: Plotly 히트맵 — 요일×시간대 화재 발생."""
    heat = df.dropna(subset=["발생요일", "발생시"]).copy()
    heat = heat[heat["발생요일"].isin(WEEKDAY_ORDER)]
    heat["발생시"] = heat["발생시"].astype(int)
    pivot = (
        heat.groupby(["발생요일", "발생시"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=WEEKDAY_ORDER, columns=range(24), fill_value=0)
    )
    fig = px.imshow(
        pivot,
        aspect="auto",
        color_continuous_scale="YlOrRd",
        title="요일 × 시간대 화재 발생 히트맵",
        labels=dict(x="발생 시간", y="발생 요일", color="화재 건수"),
    )
    fig.update_xaxes(dtick=1)  # 1시간 간격 눈금
    # 사용자가 셀에 마우스 올렸을 때 보이는 텍스트 (한글 라벨 + 천단위 콤마)
    fig.update_traces(
        hovertemplate="요일=%{y}<br>시간=%{x}시<br>화재 건수=%{z:,}건<extra></extra>"
    )
    return write_plotly(fig, out_dir / "02_weekday_hour_fire_heatmap.html")


def valid_geo_rows(df: pd.DataFrame) -> pd.DataFrame:
    """서울 범위 내 위경도만 — bounding box 필터."""
    geo = df.dropna(subset=["위도", "경도"]).copy()
    return geo[(geo["위도"].between(37.4, 37.8)) & (geo["경도"].between(126.7, 127.3))]


def viz_03_district_map(df: pd.DataFrame, out_dir: Path) -> Path:
    """03: Folium 지도 — 구별 발생 통계 + 전체 화재 히트맵."""
    geo = valid_geo_rows(df)
    # 구별 통계 — 화재수/대표좌표/평균 출동/총 피해
    gu_stats = (
        geo.dropna(subset=["발생시군구"])
        .groupby("발생시군구")
        .agg(
            화재건수=("화재번호", "count"),
            위도=("위도", "median"),
            경도=("경도", "median"),
            평균출동분=("출동소요시간_분", "mean"),
            총재산피해백만원=("재산피해액_백만원", "sum"),
        )
        .reset_index()
        .sort_values("화재건수", ascending=False)
    )

    # 지도 중심 — 점들의 중앙값
    center = [geo["위도"].median(), geo["경도"].median()]
    m = folium.Map(location=center, zoom_start=11, tiles="CartoDB positron")

    # 화재 밀도 히트맵 레이어 (옅은 배경)
    heat_data = geo[["위도", "경도"]].dropna().values.tolist()
    HeatMap(
        heat_data, name="전체 화재 밀도", radius=14, blur=18, min_opacity=0.25
    ).add_to(m)

    # 구별 화재수에 비례한 컬러맵 (YlOrRd 9단계)
    colormap = linear.YlOrRd_09.scale(
        gu_stats["화재건수"].min(), gu_stats["화재건수"].max()
    )
    colormap.caption = "구별 화재 건수"
    colormap.add_to(m)

    # 구별 버블 마커 + 팝업
    district_layer = folium.FeatureGroup(name="구별 발생 건수", show=True)
    max_count = gu_stats["화재건수"].max()
    for _, row in gu_stats.iterrows():
        # 반지름 — sqrt 비례 (시각적으로 균형)
        radius = 8 + 26 * math.sqrt(row["화재건수"] / max_count)
        popup = (
            f"<b>{row['발생시군구']}</b><br>"
            f"화재 건수: {int(row['화재건수']):,}건<br>"
            f"평균 출동: {row['평균출동분']:.1f}분<br>"
            f"재산피해 합계: {row['총재산피해백만원']:,.1f}백만원"
        )
        folium.CircleMarker(
            location=[row["위도"], row["경도"]],
            radius=radius,
            color="#7f1d1d",
            fill=True,
            fill_color=colormap(row["화재건수"]),
            fill_opacity=0.72,
            weight=1,
            tooltip=f"{row['발생시군구']}: {int(row['화재건수']):,}건",
            popup=folium.Popup(popup, max_width=300),
        ).add_to(district_layer)
    district_layer.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    out_path = out_dir / "03_district_fire_map.html"
    m.save(out_path)
    # 구별 요약 CSV 도 저장
    gu_stats.to_csv(
        out_dir / "03_district_fire_summary.csv", index=False, encoding="utf-8-sig"
    )
    return out_path


def viz_04_cause_top10(df: pd.DataFrame, out_dir: Path) -> Path:
    """04: Plotly 가로 막대 — 발화요인 TOP 10."""
    cause = (
        df.dropna(subset=["발화요인_대분류"])
        .groupby("발화요인_대분류")
        .size()
        .reset_index(name="화재건수")
        .sort_values("화재건수", ascending=False)
        .head(10)
        .sort_values("화재건수")  # 가로 막대용 오름차순
    )
    fig = px.bar(
        cause,
        x="화재건수",
        y="발화요인_대분류",
        orientation="h",
        color="화재건수",
        color_continuous_scale="Tealrose",
        title="발화요인 대분류 TOP 10",
        labels={"화재건수": "화재 건수", "발화요인_대분류": "발화요인"},
        text="화재건수",
    )
    # 막대 끝 숫자 표기 + 잘림 방지
    fig.update_traces(
        texttemplate="%{text:,}건", textposition="outside", cliponaxis=False
    )
    fig.update_layout(coloraxis_showscale=False)
    return write_plotly(fig, out_dir / "04_top10_fire_causes.html")


def viz_05_response_vs_distance(df: pd.DataFrame, out_dir: Path) -> Path:
    """05: Plotly 산점도 — 출동시간 vs 현장거리 + 추세선 + 상관계수."""
    scatter = df.dropna(subset=["현장거리(km)", "출동소요시간_분"]).copy()
    scatter = scatter[(scatter["현장거리(km)"] >= 0) & (scatter["출동소요시간_분"] > 0)]
    # 상위 1% 이상치 제거 — 분포 안정화
    distance_cap = scatter["현장거리(km)"].quantile(0.99)
    response_cap = scatter["출동소요시간_분"].quantile(0.99)
    plot_df = scatter[
        (scatter["현장거리(km)"] <= distance_cap)
        & (scatter["출동소요시간_분"] <= response_cap)
    ].copy()
    # 점 너무 많으면 9000개로 다운샘플
    if len(plot_df) > 9000:
        plot_df = plot_df.sample(9000, random_state=42)

    corr = plot_df["현장거리(km)"].corr(plot_df["출동소요시간_분"])
    fig = px.scatter(
        plot_df,
        x="현장거리(km)",
        y="출동소요시간_분",
        color="발생시군구",
        opacity=0.62,
        title="출동소요시간 vs 현장거리",
        labels={"현장거리(km)": "현장거리(km)", "출동소요시간_분": "출동소요시간(분)"},
        hover_data=["발생일자", "화재유형", "발화장소_대분류", "발화요인_대분류"],
    )

    # 선형 추세선 — np.polyfit
    line_df = plot_df[["현장거리(km)", "출동소요시간_분"]].dropna()
    if len(line_df) >= 2:
        slope, intercept = np.polyfit(
            line_df["현장거리(km)"], line_df["출동소요시간_분"], 1
        )
        x_line = np.linspace(
            line_df["현장거리(km)"].min(), line_df["현장거리(km)"].max(), 100
        )
        y_line = slope * x_line + intercept
        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                name="선형 추세",
                line=dict(color="#111827", width=3),
            )
        )
    # 우상단 주석 박스 — 상관계수 r 표기
    fig.add_annotation(
        x=0.98,
        y=0.98,
        xref="paper",
        yref="paper",
        text=f"상관계수 r = {corr:.3f}<br>상위 1% 이상치 제외",
        showarrow=False,
        align="right",
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="#d1d5db",
        borderwidth=1,
    )
    return write_plotly(fig, out_dir / "05_response_time_vs_distance.html")


def viz_06_high_damage_map(df: pd.DataFrame, out_dir: Path) -> Path:
    """06: Folium MarkerCluster — 재산피해 TOP100 지도 + 우상단 TOP20 표."""
    geo = valid_geo_rows(df)
    # 피해액 > 0 인 화재만 추려 상위 100개
    top = (
        geo.dropna(subset=["재산피해액(천원)"])
        .query("`재산피해액(천원)` > 0")
        .sort_values("재산피해액(천원)", ascending=False)
        .head(100)
        .copy()
    )
    top["재산피해액_백만원"] = top["재산피해액(천원)"] / 1000

    # 지도 중심 — TOP100 의 중앙값 (없으면 시청 좌표)
    center = (
        [top["위도"].median(), top["경도"].median()]
        if len(top)
        else [37.5665, 126.9780]
    )
    m = folium.Map(location=center, zoom_start=11, tiles="CartoDB positron")
    cluster = MarkerCluster(name="재산피해 TOP 100").add_to(m)

    # 마커 반지름 — log 스케일 (피해액 매우 우편향)
    max_damage = top["재산피해액(천원)"].max() if len(top) else 1
    for rank, (_, row) in enumerate(top.iterrows(), start=1):
        damage = row["재산피해액(천원)"]
        radius = 5 + 18 * math.log10(damage + 1) / math.log10(max_damage + 1)
        popup = (
            f"<b>#{rank} 재산피해 {row['재산피해액_백만원']:,.1f}백만원</b><br>"
            f"발생일자: {row.get('발생일자', '')}<br>"
            f"지역: {row.get('발생시군구', '')}<br>"
            f"화재유형: {row.get('화재유형', '')}<br>"
            f"발화요인: {row.get('발화요인_대분류', '')} / {row.get('발화요인_소분류', '')}<br>"
            f"장소: {row.get('발화장소_대분류', '')} / {row.get('발화장소_중분류', '')}<br>"
            f"사상자: {int(row.get('인명피해계', 0) or 0)}명"
        )
        folium.CircleMarker(
            location=[row["위도"], row["경도"]],
            radius=radius,
            color="#7f1d1d",
            fill=True,
            fill_color="#dc2626",
            fill_opacity=0.72,
            weight=1,
            tooltip=f"#{rank} {row.get('발생시군구', '')}: {row['재산피해액_백만원']:,.1f}백만원",
            popup=folium.Popup(popup, max_width=360),
        ).add_to(cluster)

    # 우상단 고정 패널 — TOP 20 표
    table_rows = []
    for rank, (_, row) in enumerate(top.head(20).iterrows(), start=1):
        table_rows.append(
            "<tr>"
            f"<td>{rank}</td>"
            f"<td>{row.get('발생시군구', '')}</td>"
            f"<td>{row.get('발생일자', '')}</td>"
            f"<td>{row['재산피해액_백만원']:,.1f}</td>"
            "</tr>"
        )
    table_html = "\n".join(table_rows)
    panel = f"""
    <div style="
        position: fixed; top: 16px; right: 16px; z-index: 9999; width: 360px;
        background: rgba(255,255,255,0.94); border: 1px solid #d1d5db;
        border-radius: 8px; box-shadow: 0 8px 24px rgba(0,0,0,0.18);
        font-family: Malgun Gothic, Apple SD Gothic Neo, Arial, sans-serif;
        font-size: 12px; max-height: 56vh; overflow: auto;">
      <div style="padding: 10px 12px; font-weight: 700; font-size: 14px; color: #111827;">
        재산피해액 TOP 20
      </div>
      <table style="width: 100%; border-collapse: collapse;">
        <thead>
          <tr style="background: #f3f4f6;">
            <th style="padding: 6px; text-align: right;">순위</th>
            <th style="padding: 6px; text-align: left;">구</th>
            <th style="padding: 6px; text-align: left;">일자</th>
            <th style="padding: 6px; text-align: right;">백만원</th>
          </tr>
        </thead>
        <tbody>{table_html}</tbody>
      </table>
    </div>
    """
    m.get_root().html.add_child(folium.Element(panel))
    folium.LayerControl(collapsed=False).add_to(m)

    # TOP100 CSV 저장
    ranking_cols = [
        "화재번호",
        "발생일자",
        "발생시군구",
        "화재유형",
        "발화요인_대분류",
        "발화요인_소분류",
        "발화장소_대분류",
        "발화장소_중분류",
        "재산피해액(천원)",
        "재산피해액_백만원",
        "사망자수",
        "부상자수",
        "인명피해계",
        "위도",
        "경도",
    ]
    top[[c for c in ranking_cols if c in top.columns]].to_csv(
        out_dir / "06_high_damage_ranking_top100.csv", index=False, encoding="utf-8-sig"
    )

    out_path = out_dir / "06_high_damage_fire_map_and_ranking.html"
    m.save(out_path)
    return out_path


def write_summary(
    df: pd.DataFrame, csv_path: Path, out_dir: Path, outputs: list[Path]
) -> Path:
    """전체 요약 JSON — 행수/연도 범위/구별 분포/원인 TOP10."""
    summary = {
        "source_csv": str(csv_path),
        "row_count": int(len(df)),
        "year_min": int(df["발생연도"].min()) if df["발생연도"].notna().any() else None,
        "year_max": int(df["발생연도"].max()) if df["발생연도"].notna().any() else None,
        "fire_count_by_year": {
            str(int(k)): int(v)
            for k, v in df.groupby("발생연도").size().sort_index().items()
            if pd.notna(k)
        },
        "top_districts": df.groupby("발생시군구")
        .size()
        .sort_values(ascending=False)
        .head(10)
        .to_dict(),
        "top_causes": df.groupby("발화요인_대분류")
        .size()
        .sort_values(ascending=False)
        .head(10)
        .to_dict(),
        "outputs": [p.name for p in outputs],
    }
    out_path = out_dir / "summary.json"
    out_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return out_path


def write_index(out_dir: Path, outputs: list[Path], summary_path: Path) -> Path:
    """6개 시각화로 가는 카드형 그리드 인덱스 페이지 (index.html) 작성."""
    # 파일명 → 한글 라벨 매핑 (UI 가독성)
    labels = {
        "01_year_month_fire_trend.html": "1. 연도-월별 화재 발생 추세",
        "02_weekday_hour_fire_heatmap.html": "2. 요일 × 시간대 화재 히트맵",
        "03_district_fire_map.html": "3. 구별 화재 발생 지도",
        "04_top10_fire_causes.html": "4. 발화요인 TOP 10",
        "05_response_time_vs_distance.html": "5. 출동소요시간 vs 현장거리",
        "06_high_damage_fire_map_and_ranking.html": "6. 재산피해액 TOP 화재 지도/랭킹",
    }
    items = "\n".join(
        f'<li><a href="{path.name}">{labels.get(path.name, path.name)}</a></li>'
        for path in outputs
    )
    # f-string 안에서 CSS 의 { } 는 { { } } 로 이스케이프
    html = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>화재출동 6개 시각화</title>
  <style>
    body {{
      margin: 0;
      font-family: "Malgun Gothic", "Apple SD Gothic Neo", Arial, sans-serif;
      background: #f8fafc;
      color: #111827;
    }}
    main {{
      max-width: 880px;
      margin: 0 auto;
      padding: 40px 20px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 28px;
    }}
    p {{
      margin: 0 0 24px;
      color: #4b5563;
      line-height: 1.6;
    }}
    ul {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 12px;
      list-style: none;
      padding: 0;
      margin: 0;
    }}
    li {{
      background: #ffffff;
      border: 1px solid #e5e7eb;
      border-radius: 8px;
    }}
    a {{
      display: block;
      padding: 16px;
      color: #1d4ed8;
      font-weight: 700;
      text-decoration: none;
    }}
    a:hover {{
      background: #eff6ff;
    }}
    .meta {{
      margin-top: 18px;
      font-size: 13px;
    }}
  </style>
</head>
<body>
  <main>
    <h1>화재출동 6개 시각화</h1>
    <p>각 링크를 열면 개별 인터랙티브 시각화를 볼 수 있습니다.</p>
    <ul>{items}</ul>
    <p class="meta">요약 파일: <a href="{summary_path.name}">{summary_path.name}</a></p>
  </main>
</body>
</html>
"""
    out_path = out_dir / "index.html"
    out_path.write_text(html, encoding="utf-8")
    return out_path


def parse_args() -> argparse.Namespace:
    """CLI 인자 — 입력 CSV/출력 폴더/연도 범위 옵션."""
    parser = argparse.ArgumentParser(
        description="Build six fire-dispatch visualizations."
    )
    parser.add_argument(
        "--csv", type=Path, default=DEFAULT_CSV, help="Input fire dispatch CSV path"
    )
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUT, help="Output directory"
    )
    parser.add_argument(
        "--start-year", type=int, default=None, help="Optional inclusive start year"
    )
    parser.add_argument(
        "--end-year", type=int, default=None, help="Optional inclusive end year"
    )
    return parser.parse_args()


def main() -> None:
    """메인 — 데이터 준비 → 6개 시각화 → 요약/인덱스 생성."""
    args = parse_args()
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    df = prepare_data(args.csv, args.start_year, args.end_year)
    # 6개 HTML 순차 생성
    outputs = [
        viz_01_monthly_trend(df, out_dir),
        viz_02_weekday_hour_heatmap(df, out_dir),
        viz_03_district_map(df, out_dir),
        viz_04_cause_top10(df, out_dir),
        viz_05_response_vs_distance(df, out_dir),
        viz_06_high_damage_map(df, out_dir),
    ]
    summary_path = write_summary(df, args.csv, out_dir, outputs)
    index_path = write_index(out_dir, outputs, summary_path)

    # 콘솔 요약
    print(f"rows={len(df):,}")
    print(f"years={int(df['발생연도'].min())}-{int(df['발생연도'].max())}")
    print(f"output_dir={out_dir}")
    print(f"index={index_path}")
    for path in outputs:
        print(f"created={path}")
    print(f"created={summary_path}")


if __name__ == "__main__":
    main()
