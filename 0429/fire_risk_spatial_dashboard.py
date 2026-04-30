# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from shapely.validation import make_valid


st.set_page_config(
    page_title="숙박시설 화재위험 공간분석 대시보드",
    page_icon="",
    layout="wide",
)

BASE = Path(__file__).resolve().parents[1]
K2_DIR = BASE / "0429" / "cluster2_spatial_pipeline_fire_count_150m_0429"
LISA_DIR = K2_DIR / "lisa_fire_count_150m"
NEIGHBORHOOD_GEO_PATH = BASE / "data" / "seoul_legal_dong_age_buckets_joined_0415.geojson"
STATION_PATH = BASE / "data" / "firestation_data.json"

MAP_STYLE = "carto-positron"
MAP_HEIGHT = 780
GU_CODE_MAP = {
    "11110": "종로구",
    "11140": "중구",
    "11170": "용산구",
    "11200": "성동구",
    "11440": "마포구",
    "11500": "강서구",
    "11560": "영등포구",
    "11650": "서초구",
    "11680": "강남구",
    "11710": "송파구",
}

LISA_COLORS = {
    "High-High": "#d73027",
    "Low-Low": "#4575b4",
    "High-Low": "#fc8d59",
    "Low-High": "#91bfdb",
    "Not significant": "#c7ced6",
}


st.markdown(
    """
<style>
    .stApp { background: #F6FAFF; color: #102A43; }
    .block-container { padding-top: 1.1rem; max-width: 1480px; }
    h1, h2, h3, h4, p, span, label { letter-spacing: 0; }
    .hero {
        background: linear-gradient(135deg, #E8F7FF 0%, #F1FFF8 48%, #FFF8E8 100%);
        border: 1px solid #D6EAF8;
        box-shadow: 0 10px 24px rgba(15, 39, 66, 0.08);
        border-radius: 8px;
        padding: 24px 30px;
        margin-bottom: 18px;
    }
    .hero h1 { color: #082F49; margin: 0; font-size: 2.08rem; font-weight: 850; }
    .hero p { color: #3C5870; margin: 8px 0 0 0; font-size: 1rem; }
    .method-card {
        background: #FFFFFF;
        border: 1px solid #DCECF7;
        border-radius: 8px;
        padding: 14px 16px;
        min-height: 118px;
        box-shadow: 0 6px 16px rgba(46, 87, 118, 0.06);
    }
    .method-card .step { color: #159A9C; font-weight: 850; font-size: 0.78rem; }
    .method-card .title { color: #0F172A; font-weight: 850; font-size: 1rem; margin-top: 6px; }
    .method-card .desc { color: #52677A; font-size: 0.86rem; margin-top: 6px; line-height: 1.38; }
    [data-testid="metric-container"] {
        background: #FFFFFF;
        border: 1px solid #DCECF7;
        border-radius: 8px;
        padding: 12px 14px;
        box-shadow: 0 5px 14px rgba(46, 87, 118, 0.05);
    }
    section[data-testid="stSidebar"] {
        background: #EEF8FF;
        border-right: 1px solid #D6EAF8;
    }
    section[data-testid="stSidebar"] * { color: #12324A !important; }
    section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
        background: #FFFFFF !important;
        border: 1px solid #BFDDF0 !important;
        border-radius: 8px !important;
    }
    .note {
        background: #FFFFFF;
        border: 1px solid #DCECF7;
        border-left: 5px solid #5BC0BE;
        border-radius: 8px;
        padding: 12px 14px;
        color: #334155;
        font-size: 0.92rem;
        line-height: 1.45;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background: #EAF5FC;
        border-radius: 8px;
        padding: 6px;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 7px;
        color: #3A5168;
    }
    .stTabs [aria-selected="true"] {
        background: #FFFFFF;
        color: #0F2742;
        box-shadow: 0 2px 8px rgba(46, 87, 118, 0.08);
    }
    div[data-testid="stPlotlyChart"],
    div[data-testid="stPlotlyChart"] > div,
    div[data-testid="stPlotlyChart"] .js-plotly-plot,
    div[data-testid="stPlotlyChart"] .plot-container,
    div[data-testid="stPlotlyChart"] .svg-container {
        background: #FFFFFF !important;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data
def load_analysis_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_path = max(K2_DIR.glob("*cluster_k2.csv"), key=lambda p: p.stat().st_size)
    df = pd.read_csv(data_path, encoding="utf-8-sig")
    lisa = pd.read_csv(LISA_DIR / "lisa_fire_count_150m_results.csv", encoding="utf-8-sig")
    model = pd.read_csv(K2_DIR / "spatial_model_summary_by_cluster_k2.csv", encoding="utf-8-sig")
    tune = pd.read_csv(K2_DIR / "k2_cluster_feature_set_tuning.csv", encoding="utf-8-sig")
    gwr = pd.read_csv(K2_DIR / "gwr_local_diagnostics_by_cluster_k2.csv", encoding="utf-8-sig")

    lisa_cols = ["lisa_type", "lisa_I", "lisa_p", "lisa_significant"]
    if len(df) == len(lisa):
        for col in lisa_cols:
            df[col] = lisa[col].to_numpy()
    else:
        keys = ["숙소명", "x_5181", "y_5181"]
        df = df.merge(lisa.drop_duplicates(keys)[keys + lisa_cols], on=keys, how="left")

    geo = df[["구", "동", "숙소명", "경도", "위도", "x_5181", "y_5181", "cluster_k2"]].copy()
    geo["x_round"] = geo["x_5181"].round(6)
    geo["y_round"] = geo["y_5181"].round(6)
    gwr = gwr.copy()
    gwr["x_round"] = gwr["x_5181"].round(6)
    gwr["y_round"] = gwr["y_5181"].round(6)
    gwr = gwr.merge(
        geo[["x_round", "y_round", "구", "동", "숙소명", "경도", "위도", "cluster_k2"]],
        on=["x_round", "y_round"],
        how="left",
    )
    return df, lisa, model, tune, gwr


def repair_geometry(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    fixed = gdf.copy()
    fixed["geometry"] = fixed.geometry.apply(lambda geom: make_valid(geom) if geom is not None else geom)
    fixed["geometry"] = fixed.geometry.buffer(0)
    return fixed[fixed.geometry.notna() & ~fixed.geometry.is_empty].copy()


@st.cache_data
def load_dong_geojson() -> dict:
    gdf = gpd.read_file(NEIGHBORHOOD_GEO_PATH)
    gdf["구"] = gdf["EMD_CD"].astype(str).str[:5].map(GU_CODE_MAP)
    gdf = repair_geometry(gdf.dropna(subset=["구"]))
    gdf["동"] = gdf["EMD_KOR_NM"].astype(str)
    gdf["지역키"] = gdf["구"] + " " + gdf["동"]
    return json.loads(gdf[["지역키", "구", "동", "geometry"]].to_json())


@st.cache_data
def load_gu_geojson() -> dict:
    gdf = gpd.read_file(NEIGHBORHOOD_GEO_PATH)
    gdf["구"] = gdf["EMD_CD"].astype(str).str[:5].map(GU_CODE_MAP)
    gdf = repair_geometry(gdf.dropna(subset=["구"]))
    gdf = repair_geometry(gdf.dissolve(by="구", as_index=False))
    return json.loads(gdf[["구", "geometry"]].to_json())


@st.cache_data
def load_stations() -> pd.DataFrame:
    with STATION_PATH.open(encoding="utf-8") as f:
        raw = json.load(f)
    stations = pd.DataFrame(raw).rename(columns={"lng": "경도", "lat": "위도", "name": "센터명"})
    stations["시설유형"] = np.where(stations.index < 26, "소방서", "119안전센터")
    stations["센터명"] = stations["센터명"].astype(str).str.strip()
    stations.loc[stations["센터명"].eq("") | stations["센터명"].str.contains(r"\?", regex=True), "센터명"] = (
        stations["시설유형"] + " " + (stations.index + 1).astype(str)
    )
    return stations[["센터명", "시설유형", "경도", "위도", "count"]].dropna(subset=["경도", "위도"])


def haversine_km(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    radius = 6371.0088
    p1 = np.radians(lat1)
    p2 = np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * radius * np.arcsin(np.sqrt(a))


def apply_filters(df: pd.DataFrame, gu: str, cluster: str, lisa_type: str) -> pd.DataFrame:
    out = df.copy()
    if gu != "전체":
        out = out[out["구"].eq(gu)]
    if cluster != "전체":
        out = out[out["cluster_k2"].eq(int(cluster.split()[-1]))]
    if lisa_type != "전체":
        out = out[out["lisa_type"].eq(lisa_type)]
    return out


def summarize_dong(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.groupby(["구", "동"])
        .agg(
            시설수=("숙소명", "count"),
            평균위험=("최종_화재위험점수", "mean"),
            평균화재=("fire_count_150m", "mean"),
            HighHigh수=("lisa_type", lambda s: int((s == "High-High").sum())),
            고위험군비율=("cluster_k2", lambda s: float((s == 1).mean())),
        )
        .reset_index()
    )
    out["지역키"] = out["구"] + " " + out["동"]
    return out


def summarize_gu(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("구")
        .agg(
            시설수=("숙소명", "count"),
            평균위험=("최종_화재위험점수", "mean"),
            평균화재=("fire_count_150m", "mean"),
            HighHigh수=("lisa_type", lambda s: int((s == "High-High").sum())),
            고위험군비율=("cluster_k2", lambda s: float((s == 1).mean())),
        )
        .reset_index()
    )


def choropleth_map(summary: pd.DataFrame, geojson: dict, location: str, metric: str, title: str) -> go.Figure:
    fig = px.choropleth_mapbox(
        summary,
        geojson=geojson,
        locations=location,
        featureidkey=f"properties.{location}",
        color=metric,
        color_continuous_scale="YlOrRd",
        hover_name=location,
        hover_data={
            "시설수": ":,",
            "평균위험": ":.1f",
            "평균화재": ":.1f",
            "HighHigh수": ":,",
            "고위험군비율": ":.1%",
        },
        mapbox_style=MAP_STYLE,
        center={"lat": 37.545, "lon": 126.99},
        zoom=9.75,
        opacity=0.9,
        height=MAP_HEIGHT,
    )
    fig.update_traces(marker_line_width=1.15, marker_line_color="white")
    fig.update_layout(
        title=dict(text=title, x=0.01, y=0.98, font=dict(size=19, color="#102A43")),
        margin=dict(l=0, r=0, t=44, b=0),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(color="#102A43"),
        coloraxis_colorbar=dict(
            title=metric,
            bgcolor="#FFFFFF",
            bordercolor="#DCECF7",
            borderwidth=1,
            tickfont=dict(color="#102A43"),
            titlefont=dict(color="#102A43"),
        ),
    )
    return fig


def grid_summary(df: pd.DataFrame, grid_m: int) -> pd.DataFrame:
    work = df.copy()
    work["gx"] = (work["x_5181"] // grid_m).astype(int)
    work["gy"] = (work["y_5181"] // grid_m).astype(int)
    return (
        work.groupby(["gx", "gy"])
        .agg(
            경도=("경도", "mean"),
            위도=("위도", "mean"),
            시설수=("숙소명", "count"),
            평균위험=("최종_화재위험점수", "mean"),
            평균화재=("fire_count_150m", "mean"),
            HighHigh수=("lisa_type", lambda s: int((s == "High-High").sum())),
            고위험군비율=("cluster_k2", lambda s: float((s == 1).mean())),
        )
        .reset_index()
    )


def grid_map(summary: pd.DataFrame, metric: str) -> go.Figure:
    fig = px.scatter_mapbox(
        summary,
        lat="위도",
        lon="경도",
        size="시설수",
        color=metric,
        color_continuous_scale="YlOrRd",
        hover_data={"시설수": True, "평균위험": ":.1f", "평균화재": ":.1f", "HighHigh수": True, "고위험군비율": ":.1%"},
        mapbox_style=MAP_STYLE,
        center={"lat": 37.545, "lon": 126.99},
        zoom=10.1,
        height=MAP_HEIGHT,
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=20, b=0),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(color="#102A43"),
        coloraxis_colorbar=dict(
            title=metric,
            bgcolor="#FFFFFF",
            bordercolor="#DCECF7",
            borderwidth=1,
            tickfont=dict(color="#102A43"),
            titlefont=dict(color="#102A43"),
        ),
    )
    return fig


def facility_map(df: pd.DataFrame, color_by: str) -> go.Figure:
    if color_by == "LISA":
        color = "lisa_type"
        color_map = LISA_COLORS
    elif color_by == "군집":
        color = "cluster_k2"
        color_map = {0: "#4C78A8", 1: "#E45756"}
    else:
        color = "최종_화재위험점수"
        color_map = None
    fig = px.scatter_mapbox(
        df,
        lat="위도",
        lon="경도",
        color=color,
        color_discrete_map=color_map,
        color_continuous_scale="YlOrRd",
        size=np.clip(df["fire_count_150m"].fillna(0) + 1, 1, 18),
        hover_name="숙소명",
        hover_data={"구": True, "동": True, "cluster_k2": True, "lisa_type": True, "fire_count_150m": ":.1f", "최종_화재위험점수": ":.1f"},
        mapbox_style=MAP_STYLE,
        center={"lat": 37.545, "lon": 126.99},
        zoom=10.1,
        height=MAP_HEIGHT,
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=20, b=0),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(color="#102A43"),
        legend_title_text=color_by,
        legend=dict(bgcolor="#FFFFFF", bordercolor="#DCECF7", borderwidth=1, font=dict(color="#102A43")),
        coloraxis_colorbar=dict(
            bgcolor="#FFFFFF",
            bordercolor="#DCECF7",
            borderwidth=1,
            tickfont=dict(color="#102A43"),
            titlefont=dict(color="#102A43"),
        ),
    )
    return fig


df, lisa, model, tune, gwr = load_analysis_data()
dong_geojson = load_dong_geojson()
gu_geojson = load_gu_geojson()
stations = load_stations()

st.markdown(
    """
<div class="hero">
  <h1>숙박시설 화재위험 공간분석 대시보드</h1>
  <p>K=2 군집화, LISA High-High, SLM 공간시차효과, GWR/MGWR 지역 차이를 한 흐름으로 확인합니다.</p>
</div>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("필터")
    gu_filter = st.selectbox("자치구", ["전체"] + sorted(df["구"].dropna().unique().tolist()))
    cluster_filter = st.selectbox("군집", ["전체", "Cluster 0", "Cluster 1"])
    lisa_filter = st.selectbox("LISA 유형", ["전체", "High-High", "Low-Low", "High-Low", "Low-High", "Not significant"])
    st.divider()
    map_mode = st.radio("지도 보기", ["동 영역", "자치구 영역", "격자 요약", "시설 점"])
    metric = st.radio("색상 기준", ["HighHigh수", "평균위험", "평균화재", "고위험군비율", "시설수"])
    grid_m = st.slider("격자 크기", 400, 1200, 700, 100)

view = apply_filters(df, gu_filter, cluster_filter, lisa_filter)

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("분석 시설", f"{len(view):,}개")
m2.metric("High-High", f"{(view['lisa_type'] == 'High-High').sum():,}개")
m3.metric("고위험군 비율", f"{(view['cluster_k2'] == 1).mean() * 100:.1f}%")
m4.metric("평균 150m 화재", f"{view['fire_count_150m'].mean():.2f}건")
m5.metric("평균 위험점수", f"{view['최종_화재위험점수'].mean():.1f}")

steps = [
    ("STEP 1", "변수 조합 후보 비교", "전체 변수, 정책형 변수, 위험점수 포함 변수 조합 비교"),
    ("STEP 2", "군집화 변수 선택", "실루엣 점수와 Calinski-Harabasz 지수로 최종 조합 선택"),
    ("STEP 3", "회귀 변수 검토", "반복적 소거, VIF, 해석 가능성으로 안정성 확인"),
    ("STEP 4", "공간통계", "Moran's I와 LISA로 공간 군집성과 핫스팟 확인"),
    ("STEP 5", "공간회귀", "SLM 주력, GWR/MGWR은 지역별 차이 보조 설명"),
]
cards = st.columns(5)
for col, (step, title, desc) in zip(cards, steps):
    col.markdown(
        f'<div class="method-card"><div class="step">{step}</div><div class="title">{title}</div><div class="desc">{desc}</div></div>',
        unsafe_allow_html=True,
    )

tab1, tab2, tab3, tab4 = st.tabs(["위험도 요약", "공간 지도", "모델 근거", "안전센터 시뮬레이션"])

with tab1:
    left, right = st.columns([1.05, 1])
    dong_sum = summarize_dong(view).sort_values(["HighHigh수", "평균화재", "평균위험"], ascending=False)
    with left:
        st.subheader("동별 위험 우선순위")
        st.dataframe(
            dong_sum.head(25).style.format({"평균위험": "{:.1f}", "평균화재": "{:.1f}", "고위험군비율": "{:.1%}"}),
            hide_index=True,
            use_container_width=True,
        )
    with right:
        st.subheader("High-High 상위 동")
        plot = dong_sum.head(12).sort_values("HighHigh수")
        fig = px.bar(plot, x="HighHigh수", y="지역키", orientation="h", color="평균화재", color_continuous_scale="Reds", text="HighHigh수")
        fig.update_layout(height=520, margin=dict(l=0, r=0, t=10, b=0), yaxis_title="", xaxis_title="High-High 시설 수")
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("공간 위험 지도")
    if map_mode == "동 영역":
        st.plotly_chart(choropleth_map(summarize_dong(view), dong_geojson, "지역키", metric, "동별 화재위험 공간 분포"), use_container_width=True)
    elif map_mode == "자치구 영역":
        st.plotly_chart(choropleth_map(summarize_gu(view), gu_geojson, "구", metric, "자치구별 화재위험 공간 분포"), use_container_width=True)
    elif map_mode == "격자 요약":
        st.plotly_chart(grid_map(grid_summary(view, grid_m), metric), use_container_width=True)
    else:
        st.plotly_chart(facility_map(view, "LISA"), use_container_width=True)
    st.markdown('<div class="note">동 영역 지도는 점을 숨기고 행정구역 자체를 색으로 칠합니다. High-High수는 LISA 기준으로 주변까지 함께 화재건수가 높은 시설이 얼마나 모였는지를 뜻합니다.</div>', unsafe_allow_html=True)

    with st.expander("GWR local R² 표본 보기"):
        gview = gwr.dropna(subset=["경도", "위도"]).copy()
        if gu_filter != "전체":
            gview = gview[gview["구"].eq(gu_filter)]
        gview["local_R2_raw"] = gview["local_R2"]
        gview["local_R2"] = gview["local_R2"].clip(lower=0)
        fig = px.scatter_mapbox(
            gview,
            lat="위도",
            lon="경도",
            color="local_R2",
            color_continuous_scale="Viridis",
            hover_name="숙소명",
            hover_data={"구": True, "동": True, "cluster": True, "local_R2": ":.3f", "local_R2_raw": ":.3f"},
            mapbox_style=MAP_STYLE,
            center={"lat": 37.545, "lon": 126.99},
            zoom=10.2,
            height=680,
        )
        fig.update_layout(margin=dict(l=0, r=0, t=20, b=0))
        fig.update_layout(
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            font=dict(color="#102A43"),
            coloraxis_colorbar=dict(
                bgcolor="#FFFFFF",
                bordercolor="#DCECF7",
                borderwidth=1,
                tickfont=dict(color="#102A43"),
                titlefont=dict(color="#102A43"),
            ),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("local R² 원값이 0보다 작은 지점은 지도 색상에서 0으로 보정했습니다.")

with tab3:
    st.subheader("모델 선택 근거")
    left, right = st.columns(2)
    with left:
        st.markdown("**군집화 변수 조합 비교**")
        st.dataframe(
            tune[["feature_set", "n_features", "silhouette", "calinski_harabasz", "cluster0_n", "cluster1_n"]]
            .style.format({"silhouette": "{:.3f}", "calinski_harabasz": "{:.1f}"}),
            hide_index=True,
            use_container_width=True,
        )
    with right:
        st.markdown("**공간회귀 모델 점수**")
        st.dataframe(
            model[["cluster", "model", "knn_k", "n", "fit", "adj_fit", "aic", "rho_or_lambda", "resid_moran_I", "resid_moran_p", "bandwidth"]]
            .style.format({"fit": "{:.3f}", "adj_fit": "{:.3f}", "aic": "{:.1f}", "rho_or_lambda": "{:.3f}", "resid_moran_I": "{:.3f}", "resid_moran_p": "{:.3f}", "bandwidth": "{:.1f}"}),
            hide_index=True,
            use_container_width=True,
        )
    c1, c2, c3 = st.columns(3)
    model_plot = model.copy()
    model_plot["cluster"] = "Cluster " + model_plot["cluster"].astype(str)
    for col, y, title in [(c1, "fit", "설명력"), (c2, "aic", "AIC"), (c3, "resid_moran_I", "잔차 Moran's I")]:
        fig = px.bar(model_plot, x="model", y=y, color="cluster", barmode="group", color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(height=340, title=title, margin=dict(l=0, r=0, t=45, b=0), xaxis_title="", yaxis_title="")
        col.plotly_chart(fig, use_container_width=True)
    st.markdown('<div class="note">최종 해석 모델은 K=2 + fire_count_150m + SLM입니다. LISA는 공간 핫스팟 근거, GWR/MGWR은 지역별 설명력 차이를 확인하는 보조 분석으로 사용합니다.</div>', unsafe_allow_html=True)

with tab4:
    st.subheader("가장 가까운 119안전센터 출동 시뮬레이션")
    sim = view.copy()
    only_hh = st.checkbox("High-High 시설만 보기", value=True)
    if only_hh:
        sim = sim[sim["lisa_type"].eq("High-High")]
    sim["선택명"] = sim["구"] + " " + sim["동"] + " | " + sim["숙소명"].astype(str)
    selected = st.selectbox("화재 발생 숙박시설", sim.sort_values(["fire_count_150m", "최종_화재위험점수"], ascending=False)["선택명"].head(500))
    speed = st.slider("가정 평균 출동속도", 15, 50, 28, 1)
    fac = sim[sim["선택명"].eq(selected)].iloc[0]
    safe = stations[stations["시설유형"].eq("119안전센터")].copy()
    safe["거리_km"] = haversine_km(fac["위도"], fac["경도"], safe["위도"].to_numpy(), safe["경도"].to_numpy())
    nearest = safe.sort_values("거리_km").iloc[0]
    minutes = nearest["거리_km"] / speed * 60

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("선택 시설", str(fac["숙소명"])[:18])
    s2.metric("최근접 안전센터", str(nearest["센터명"])[:18])
    s3.metric("직선거리", f"{nearest['거리_km']:.2f} km")
    s4.metric("예상 도착", f"{minutes:.1f}분")

    route = pd.DataFrame({"경도": [nearest["경도"], fac["경도"]], "위도": [nearest["위도"], fac["위도"]], "구분": ["119안전센터", "화재 발생 숙박시설"]})
    fig = go.Figure()
    fig.add_trace(
        go.Scattermapbox(
            lat=route["위도"],
            lon=route["경도"],
            mode="lines+markers+text",
            text=route["구분"],
            textposition="top right",
            marker=dict(size=[14, 17], color=["#2563EB", "#D73027"]),
            line=dict(width=4, color="#111827"),
            hovertext=[nearest["센터명"], fac["숙소명"]],
            hoverinfo="text",
        )
    )
    fig.update_layout(
        mapbox=dict(style=MAP_STYLE, center=dict(lat=float(route["위도"].mean()), lon=float(route["경도"].mean())), zoom=12),
        height=720,
        margin=dict(l=0, r=0, t=20, b=0),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(color="#102A43"),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('<div class="note">이 시뮬레이션은 실제 도로 경로·신호·교통상황을 반영하지 않은 직선거리 기반입니다. 최근접 안전센터 접근성의 상대 비교 용도로 해석하세요.</div>', unsafe_allow_html=True)
