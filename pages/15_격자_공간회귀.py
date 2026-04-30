# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(page_title="격자 공간회귀", page_icon="▦", layout="wide")

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "data" / "grid_spatial_dashboard"


@st.cache_data
def load_tables() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    moran = pd.read_csv(OUT / "grid_moran_summary.csv", encoding="utf-8-sig")
    models = pd.read_csv(OUT / "grid_model_comparison.csv", encoding="utf-8-sig")
    metadata = json.loads((OUT / "metadata.json").read_text(encoding="utf-8"))
    return moran, models, metadata


@st.cache_data
def load_grid(cell_size: int) -> tuple[pd.DataFrame, dict]:
    path = OUT / f"seoul_grid_{cell_size}m.geojson"
    grid = gpd.read_file(path).to_crs("EPSG:4326")
    geojson = json.loads(grid.to_json())
    df = pd.DataFrame(grid.drop(columns="geometry"))
    return df, geojson


required = [
    OUT / "grid_moran_summary.csv",
    OUT / "grid_model_comparison.csv",
    OUT / "seoul_grid_250m.geojson",
    OUT / "seoul_grid_500m.geojson",
]
if not all(path.exists() for path in required):
    st.error("격자 공간회귀 산출물이 없습니다. `python scripts/build_grid_spatial_dashboard_data.py`를 먼저 실행하세요.")
    st.stop()

moran, models, metadata = load_tables()

st.title("격자 기반 공간회귀 시각화")
st.caption("250m grid 생성 → spatial join 변수 집계 → Moran's I → Queen/DistanceBand 가중치 → OLS·SLM·SEM 비교 → 500m robustness check")

steps = pd.DataFrame(
    [
        ["1", "250m grid 생성", "서울 shp 경계(EPSG:5179) 기준 정방 격자 생성 후 경계로 clip"],
        ["2", "spatial join 변수 집계", "숙박시설 포인트를 격자에 결합하고 화재수·위험변수 평균/합계 산출"],
        ["3", "Moran's I 확인", "격자 단위 log1p 화재수의 전역 공간자기상관 검정"],
        ["4", "W 설정", "추천 1순위 Queen 인접, 보조 검증으로 DistanceBand(500m) 비교"],
        ["5", "OLS → SLM/SEM 비교", "OLS 잔차 Moran's I와 공간모형 잔차를 비교"],
        ["6", "500m robustness check", "격자를 500m로 키워 결론 방향이 유지되는지 확인"],
    ],
    columns=["단계", "작업", "구현"],
)

with st.expander("분석 체인", expanded=True):
    st.dataframe(steps, hide_index=True, use_container_width=True)

cell_size = st.sidebar.radio("격자 크기", [250, 500], horizontal=True)
color_col = st.sidebar.selectbox(
    "지도 색상",
    [
        "facility_count",
        "facility_density_km2",
        "fire_sum",
        "log1p_fire_sum",
        "mean_fire_risk",
        "mean_building_age",
        "mean_nearby_buildings",
        "mean_density",
        "mean_enforcement",
        "mean_road_risk",
        "mean_structure_age",
    ],
    index=0,
)

grid_df, grid_geojson = load_grid(cell_size)
queen_moran = moran[(moran["grid_size_m"] == cell_size) & (moran["weights"] == "Queen")].iloc[0]
queen_models = models[(models["grid_size_m"] == cell_size) & (models["weights"] == "Queen")].copy()

c1, c2, c3, c4 = st.columns(4)
c1.metric("격자 수", f"{int(queen_moran['n_cells']):,}")
c2.metric("시설 포함 격자", f"{int(queen_moran['n_active_cells']):,}")
c3.metric("Moran's I", f"{queen_moran['moran_I']:.3f}", f"p={queen_moran['p_value']:.3f}")
c4.metric("Queen 평균 이웃", f"{queen_moran['mean_neighbors']:.1f}")

tab_map, tab_model, tab_weight, tab_data = st.tabs(["격자 지도", "OLS·SLM·SEM", "가중치·강건성", "집계 데이터"])

with tab_map:
    plot_df = grid_df.copy()
    plot_df[color_col] = pd.to_numeric(plot_df[color_col], errors="coerce").fillna(0)
    fig = px.choropleth_mapbox(
        plot_df,
        geojson=grid_geojson,
        locations="grid_id",
        featureidkey="properties.grid_id",
        color=color_col,
        hover_data={
            "grid_id": True,
            "facility_count": ":.0f",
            "fire_sum": ":.0f",
            "mean_fire_risk": ":.3f",
            "mean_density": ":.1f",
            "mean_road_risk": ":.3f",
        },
        color_continuous_scale="YlOrRd",
        mapbox_style="carto-positron",
        center={"lat": 37.56, "lon": 126.98},
        zoom=9.3 if cell_size == 250 else 9.0,
        opacity=0.68,
        height=680,
    )
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig, use_container_width=True)

with tab_model:
    left, right = st.columns([1.05, 1])
    with left:
        fig_r2 = px.bar(
            queen_models,
            x="model",
            y="r2",
            color="model",
            text=queen_models["r2"].map(lambda v: f"{v:.3f}"),
            color_discrete_map={"OLS": "#64748B", "SLM": "#DC2626", "SEM": "#2563EB"},
            title=f"{cell_size}m Queen 기준 모델 설명력",
        )
        fig_r2.update_traces(textposition="outside")
        fig_r2.update_layout(showlegend=False, yaxis_title="R² / pseudo R²", xaxis_title=None)
        st.plotly_chart(fig_r2, use_container_width=True)
    with right:
        fig_mi = px.bar(
            queen_models,
            x="model",
            y="residual_moran_I",
            color="model",
            text=queen_models["residual_moran_I"].map(lambda v: f"{v:.3f}"),
            color_discrete_map={"OLS": "#64748B", "SLM": "#DC2626", "SEM": "#2563EB"},
            title="잔차 Moran's I",
        )
        fig_mi.add_hline(y=0, line_dash="dash", line_color="#334155")
        fig_mi.update_traces(textposition="outside")
        fig_mi.update_layout(showlegend=False, yaxis_title="Residual Moran's I", xaxis_title=None)
        st.plotly_chart(fig_mi, use_container_width=True)

    show = queen_models[["model", "r2", "aic", "spatial_param", "residual_moran_I", "residual_moran_p"]].copy()
    show.columns = ["모델", "R²/pseudo R²", "AIC", "공간계수(ρ/λ)", "잔차 Moran's I", "잔차 p-value"]
    st.dataframe(show, hide_index=True, use_container_width=True)

with tab_weight:
    w250 = moran[moran["grid_size_m"].eq(250)].copy()
    fig_w = px.bar(
        w250,
        x="weights",
        y="moran_I",
        color="weights",
        text=w250["moran_I"].map(lambda v: f"{v:.3f}"),
        title="250m 격자: Queen vs DistanceBand(500m)",
    )
    fig_w.update_traces(textposition="outside")
    fig_w.update_layout(showlegend=False, xaxis_title=None, yaxis_title="Moran's I")
    st.plotly_chart(fig_w, use_container_width=True)

    robust = models[models["weights"].eq("Queen")].copy()
    fig_rb = px.line(
        robust,
        x="grid_size_m",
        y="r2",
        color="model",
        markers=True,
        title="Queen 기준 250m vs 500m robustness check",
    )
    fig_rb.update_layout(xaxis_title="Grid size (m)", yaxis_title="R² / pseudo R²")
    st.plotly_chart(fig_rb, use_container_width=True)

    compare = models[["grid_size_m", "weights", "model", "r2", "spatial_param", "residual_moran_I", "residual_moran_p"]]
    st.dataframe(compare, hide_index=True, use_container_width=True)

with tab_data:
    st.write("사용 변수")
    st.json(metadata, expanded=False)
    cols = [
        "grid_id",
        "cell_size_m",
        "facility_count",
        "facility_density_km2",
        "fire_sum",
        "log1p_fire_sum",
        "mean_fire_risk",
        "mean_building_age",
        "mean_nearby_buildings",
        "mean_density",
        "mean_enforcement",
        "mean_road_risk",
        "mean_structure_age",
    ]
    st.dataframe(grid_df[cols].sort_values("facility_count", ascending=False), hide_index=True, use_container_width=True)
