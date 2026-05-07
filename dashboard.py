# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import math
import heapq
import ast
import re
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from shapely.geometry import Point, shape
from shapely.ops import unary_union


st.set_page_config(
    page_title="서울시 관광 지역 내 숙박 시설 화재 위험도 분석",
    page_icon="📊",
    layout="wide",
)

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "0430"
LICENSE_VIZ_DIR = BASE / "data" / "viz_each_gu"
RAW_DIR = BASE / "원본데이터"
FOREIGN_LICENSE = RAW_DIR / "서울시 외국인관광도시민박업 인허가 정보.csv"
TOUR_LICENSE = RAW_DIR / "서울시 관광숙박업 인허가 정보.csv"
LODGING_LICENSE = RAW_DIR / "서울시 숙박업 인허가 정보.csv"
FINAL_TABLE = DATA_DIR / "최종테이블0429.csv"
SCORE_TABLE = DATA_DIR / "시설별_화재위험_성적표_0429_v3.csv"
MODEL_SUMMARY = DATA_DIR / "가중치군집_대표모형_최종결과표.csv"
MODEL_COMPARE = DATA_DIR / "가중치군집_공간모형_비교표.csv"
GWR_LOCAL_SOURCE = BASE / "data" / "gwr_results.csv"
MGWR_LOCAL_PARAM_SOURCE = BASE / "data" / "mgwr_local_params_low_high.csv"
MGWR_CLUSTER_LOCAL_SOURCE = (
    BASE
    / "0424"
    / "data"
    / "cluster3_spatial_pipeline_fire_count_150m_0428"
    / "gwr_local_diagnostics_by_cluster.csv"
)
MGWR_CLUSTER_INPUT_SOURCE = (
    BASE
    / "0424"
    / "data"
    / "cluster3_spatial_pipeline_fire_count_150m_0428"
    / "최최최종0428변수테이블.csv"
)
DONG_BOUNDARY_SOURCE = (
    BASE / "data" / "seoul_legal_dong_age_buckets_joined_0415.geojson"
)
GRID_250_SOURCE = BASE / "data" / "grid_spatial_dashboard" / "seoul_grid_250m.geojson"
GRID_500_SOURCE = BASE / "data" / "grid_spatial_dashboard" / "seoul_grid_500m.geojson"

SEOUL_SIGUNGU = {
    "11110": "종로구",
    "11140": "중구",
    "11170": "용산구",
    "11200": "성동구",
    "11215": "광진구",
    "11230": "동대문구",
    "11260": "중랑구",
    "11290": "성북구",
    "11305": "강북구",
    "11320": "도봉구",
    "11350": "노원구",
    "11380": "은평구",
    "11410": "서대문구",
    "11440": "마포구",
    "11470": "양천구",
    "11500": "강서구",
    "11530": "구로구",
    "11545": "금천구",
    "11560": "영등포구",
    "11590": "동작구",
    "11620": "관악구",
    "11650": "서초구",
    "11680": "강남구",
    "11710": "송파구",
    "11740": "강동구",
}
MODEL_IMAGE = DATA_DIR / "공간 모델 선택.png"
FEATURE_IMAGE = DATA_DIR / "군집별 피처 평균값-중앙값 비교.png"
ROUTE_SOURCE = BASE / "data" / "서울10구_숙소_소방거리_유클리드.csv"
FIRE_FACILITY_SOURCE = BASE / "data" / "소방서_안전센터_구조대_위치정보_2025_wgs84.csv"
ROAD_WIDTH_SOURCE = (
    BASE / "road_width_10gu" / "data" / "seoul_road_width_10gu_official_only_roads.csv"
)
ROAD_WIDTH_LINE_SOURCE = (
    BASE
    / "road_width_10gu"
    / "data"
    / "seoul_road_width_10gu_official_complete_lines.geojson"
)
ROAD_ROUTE_LINE_SOURCE = (
    BASE / "road_width_10gu" / "data" / "seoul_road_width_10gu_osm_lines.geojson"
)
ROAD_ROUTE_UNMATCHED_SOURCE = (
    BASE / "road_width_10gu" / "data" / "seoul_road_width_10gu_osm_line_unmatched.csv"
)

COLORS = {
    "blue": "#3267e8",
    "mint": "#10c5a5",
    "coral": "#ff7d7d",
    "amber": "#f4bd45",
    "ink": "#1f2b3d",
    "muted": "#6f7c91",
    "line": "#e8eef7",
}

CLUSTER_COLORS = {
    "저위험군": "#67c8a2",
    "중위험군": "#f3bd4f",
    "고위험군": "#ef7777",
}


st.markdown(
    """
    <style>
    .stApp { background: #f3f6fb; color: #1f2b3d; }
    .block-container { max-width: 1240px; padding: 2rem 1.4rem 3rem 1.4rem; }
    [data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e8eef7; }
    h1, h2, h3 { color: #1f2b3d; letter-spacing: 0; }
    .sidebar-title {
        color: #1f2b3d;
        font-size: 1.14rem;
        line-height: 1.35;
        font-weight: 850;
        margin: 0 0 1.1rem 0;
    }
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e8eef7;
        border-radius: 14px;
        padding: 14px 16px;
        box-shadow: 0 10px 26px rgba(38, 52, 82, .06);
    }
    div[data-testid="stMetricLabel"] { color: #6f7c91; }
    div[role="radiogroup"] {
        gap: 10px;
        margin-bottom: 10px;
    }
    .section-title {
        margin-top: 18px;
        margin-bottom: 8px;
        font-size: 1.08rem;
        font-weight: 800;
        color: #1f2b3d;
    }
    .soft-note { color: #6f7c91; font-size: .9rem; margin-top: -4px; margin-bottom: 8px; }
    [data-testid="stDataFrame"] {
        border: 1px solid #e8eef7;
        border-radius: 14px;
        overflow: hidden;
        box-shadow: 0 10px 26px rgba(38, 52, 82, .05);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def read_csv(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "cp949", "euc-kr", "utf-8"):
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, low_memory=False)


@st.cache_data
def load_0430() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    final_df = read_csv(FINAL_TABLE)
    score_df = read_csv(SCORE_TABLE)
    model_df = read_csv(MODEL_SUMMARY)
    compare_df = read_csv(MODEL_COMPARE)

    for df in (final_df, score_df, model_df, compare_df):
        df.columns = [str(col).strip() for col in df.columns]
        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].astype("string").str.strip()

    numeric_cols = [
        "승인연도",
        "소방위험도_점수",
        "주변건물수",
        "집중도",
        "단속위험도",
        "구조노후도",
        "도로폭위험도",
        "위도",
        "경도",
        "최근접_소화용수_거리등급",
        "총층수",
        "연면적",
        "cluster",
        "최종위험점수_new",
    ]
    for col in numeric_cols:
        if col in final_df.columns:
            final_df[col] = pd.to_numeric(final_df[col], errors="coerce")
    final_df["승인연도"] = final_df["승인연도"].astype("Int64")

    if "cluster_label" not in final_df.columns:
        final_df["cluster_label"] = (
            final_df["cluster"]
            .map({0: "중위험군", 1: "저위험군", 2: "고위험군"})
            .fillna("미분류")
        )

    return final_df, score_df, model_df, compare_df


@st.cache_data
def load_gwr_local(final_df: pd.DataFrame) -> pd.DataFrame:
    if not GWR_LOCAL_SOURCE.exists():
        return pd.DataFrame()

    gwr = read_csv(GWR_LOCAL_SOURCE)
    gwr.columns = [str(col).strip() for col in gwr.columns]
    for col in gwr.columns:
        if (
            col.startswith("coef_")
            or col.startswith("tval_")
            or col in ["위도", "경도", "local_R2", "bandwidth"]
        ):
            gwr[col] = pd.to_numeric(gwr[col], errors="coerce")

    keys = ["위도", "경도"]
    if not set(keys).issubset(gwr.columns) or not set(keys).issubset(final_df.columns):
        return pd.DataFrame()

    left = gwr.copy()
    left["_lat_key"] = left["위도"].round(6)
    left["_lon_key"] = left["경도"].round(6)

    meta_cols = [
        "구",
        "동",
        "숙소명",
        "cluster_label",
        "최종위험점수_new",
        "위도",
        "경도",
    ]
    meta_cols = [col for col in meta_cols if col in final_df.columns]
    right = final_df[meta_cols].dropna(subset=["위도", "경도"]).copy()
    right["_lat_key"] = right["위도"].round(6)
    right["_lon_key"] = right["경도"].round(6)
    right = right.drop_duplicates(["_lat_key", "_lon_key"])

    merged = left.merge(
        right.drop(columns=["위도", "경도"], errors="ignore"),
        on=["_lat_key", "_lon_key"],
        how="left",
    )

    coef_cols = [col for col in merged.columns if col.startswith("coef_")]
    if coef_cols:
        abs_coef = merged[coef_cols].abs()
        merged["주요설명변수"] = abs_coef.idxmax(axis=1).str.replace(
            "coef_", "", regex=False
        )
        merged["주요설명계수"] = abs_coef.max(axis=1)
    return merged.drop(columns=["_lat_key", "_lon_key"], errors="ignore")


@st.cache_data
def load_mgwr_cluster_local() -> pd.DataFrame:
    if MGWR_LOCAL_PARAM_SOURCE.exists():
        mgwr = read_csv(MGWR_LOCAL_PARAM_SOURCE)
        mgwr.columns = [str(col).strip() for col in mgwr.columns]
        for col in mgwr.columns:
            if col.startswith(("coef_", "tval_", "bw_", "z_", "contrib_")) or col in [
                "cluster",
                "x_5181",
                "y_5181",
                "위도",
                "경도",
                "local_R2",
                "residual",
            ]:
                mgwr[col] = pd.to_numeric(mgwr[col], errors="coerce")
        return mgwr

    if not MGWR_CLUSTER_LOCAL_SOURCE.exists() or not MGWR_CLUSTER_INPUT_SOURCE.exists():
        return pd.DataFrame()

    local = read_csv(MGWR_CLUSTER_LOCAL_SOURCE)
    source = read_csv(MGWR_CLUSTER_INPUT_SOURCE)
    local.columns = [str(col).strip() for col in local.columns]
    source.columns = [str(col).strip() for col in source.columns]

    for df in (local, source):
        for col in df.columns:
            if col.startswith("coef_") or col in [
                "cluster",
                "x_5181",
                "y_5181",
                "위도",
                "경도",
                "local_R2",
                "residual",
            ]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

    required_local = {"cluster", "x_5181", "y_5181"}
    required_source = {"cluster", "x_5181", "y_5181", "구", "동", "위도", "경도"}
    if not required_local.issubset(local.columns) or not required_source.issubset(
        source.columns
    ):
        return pd.DataFrame()

    left = local.copy()
    right = source.copy()
    for df in (left, right):
        df["_x_key"] = df["x_5181"].round(3)
        df["_y_key"] = df["y_5181"].round(3)

    meta_cols = [
        "cluster",
        "_x_key",
        "_y_key",
        "구",
        "동",
        "숙소명",
        "위도",
        "경도",
        "최종_화재위험점수",
        "fire_count_150m",
    ]
    meta_cols = [col for col in meta_cols if col in right.columns]
    right = right[meta_cols].drop_duplicates(["cluster", "_x_key", "_y_key"])

    merged = left.merge(right, on=["cluster", "_x_key", "_y_key"], how="left")
    merged["cluster_label"] = (
        merged["cluster"]
        .map({0: "저위험군", 1: "중위험군", 2: "고위험군"})
        .fillna("미분류")
    )

    coef_cols = [col for col in merged.columns if col.startswith("coef_")]
    if coef_cols:
        abs_coef = merged[coef_cols].abs()
        merged["주요설명변수"] = abs_coef.idxmax(axis=1).str.replace(
            "coef_", "", regex=False
        )
        merged["주요설명계수"] = abs_coef.max(axis=1)

    return merged.drop(columns=["_x_key", "_y_key"], errors="ignore")


@st.cache_data
def load_dong_boundaries() -> dict:
    if not DONG_BOUNDARY_SOURCE.exists():
        return {"type": "FeatureCollection", "features": []}
    try:
        with DONG_BOUNDARY_SOURCE.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {"type": "FeatureCollection", "features": []}


@st.cache_data
def load_seoul_grid(cell_size_m: int = 500) -> dict:
    source = GRID_250_SOURCE if cell_size_m == 250 else GRID_500_SOURCE
    if not source.exists():
        return {"type": "FeatureCollection", "features": []}
    try:
        with source.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {"type": "FeatureCollection", "features": []}


def parse_bw_table(
    model_df: pd.DataFrame, risk_group: str | None = None
) -> pd.DataFrame:
    if (
        model_df.empty
        or "BW" not in model_df.columns
        or "대표모형" not in model_df.columns
    ):
        return pd.DataFrame()

    mgwr_rows = model_df[model_df["대표모형"].astype("string").str.upper() == "MGWR"]
    if risk_group and "위험군" in mgwr_rows.columns:
        mgwr_rows = mgwr_rows[mgwr_rows["위험군"].astype("string") == risk_group]
    if mgwr_rows.empty:
        return pd.DataFrame()

    try:
        bw_values = ast.literal_eval(str(mgwr_rows.iloc[0]["BW"]))
    except (ValueError, SyntaxError):
        return pd.DataFrame()
    if not isinstance(bw_values, dict):
        return pd.DataFrame()

    rows = []
    for variable, bw in bw_values.items():
        if variable == "Intercept":
            continue
        bw_num = pd.to_numeric(pd.Series([bw]), errors="coerce").iloc[0]
        if pd.isna(bw_num):
            continue
        if bw_num <= 250:
            scale = "국소"
            note = "동/골목 단위 영향"
        elif bw_num <= 700:
            scale = "중간"
            note = "인접 동까지 확산"
        else:
            scale = "광역"
            note = "구 이상 범위의 완만한 영향"
        rows.append(
            {"변수": variable, "BW": float(bw_num), "공간범위": scale, "해석": note}
        )

    return pd.DataFrame(rows).sort_values("BW", ascending=True).reset_index(drop=True)


def extract_gu_dong(address: pd.Series) -> tuple[pd.Series, pd.Series]:
    text = (
        address.fillna("")
        .astype("string")
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    gu = text.str.extract(r"서울특별시\s+(\S+구)", expand=False)
    dong_values = []
    for addr, gu_name in zip(text, gu):
        if pd.isna(gu_name) or not addr:
            dong_values.append(pd.NA)
            continue
        parts = str(addr).split()
        try:
            gu_idx = parts.index(str(gu_name))
            dong_values.append(parts[gu_idx + 1] if gu_idx + 1 < len(parts) else pd.NA)
        except ValueError:
            dong_values.append(pd.NA)
    return gu, pd.Series(dong_values, index=address.index, dtype="string")


@st.cache_data
def load_license_sources() -> pd.DataFrame:
    sources = [
        ("외국인민박", FOREIGN_LICENSE),
        ("관광숙박업", TOUR_LICENSE),
        ("숙박업", LODGING_LICENSE),
    ]
    frames = []
    for category, path in sources:
        df = read_csv(path)
        df.columns = [str(col).strip() for col in df.columns]
        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].astype("string").str.strip()
        df["인허가연도"] = pd.to_datetime(df["인허가일자"], errors="coerce").dt.year
        df["구"], df["동"] = extract_gu_dong(df["지번주소"])
        df["분류"] = category
        frames.append(df[["인허가연도", "구", "동", "영업상태명", "분류"]])
    return pd.concat(frames, ignore_index=True)


def flatten_road_coordinates(geometry: dict) -> list[list[tuple[float, float]]]:
    if not geometry:
        return []
    coords = geometry.get("coordinates", [])
    if geometry.get("type") == "LineString":
        return [[(float(lat), float(lon)) for lon, lat in coords]]
    if geometry.get("type") == "MultiLineString":
        return [[(float(lat), float(lon)) for lon, lat in line] for line in coords]
    return []


def synthetic_road_line(
    lat: float, lon: float, length_m: float = 120, bearing_degrees: float = 32
) -> list[tuple[float, float]]:
    half = length_m / 2
    bearing = math.radians(bearing_degrees)
    dlat = math.cos(bearing) * half / 110_540
    dlon = math.sin(bearing) * half / (111_320 * math.cos(math.radians(lat)))
    return [(lat - dlat, lon - dlon), (lat + dlat, lon + dlon)]


@st.cache_data
def load_unmatched_road_points() -> pd.DataFrame:
    if not ROAD_ROUTE_UNMATCHED_SOURCE.exists():
        return pd.DataFrame()
    unmatched = read_csv(ROAD_ROUTE_UNMATCHED_SOURCE)
    unmatched.columns = [str(col).strip() for col in unmatched.columns]
    for col in unmatched.select_dtypes(include="object").columns:
        unmatched[col] = unmatched[col].astype("string").str.strip()
    for col in ["위도", "경도"]:
        if col in unmatched.columns:
            unmatched[col] = pd.to_numeric(unmatched[col], errors="coerce")
    required = {"구", "도로명", "도로폭", "위도", "경도"}
    if not required.issubset(unmatched.columns):
        return pd.DataFrame()
    return unmatched.dropna(subset=["구", "도로명", "위도", "경도"]).copy()


@st.cache_data
def load_road_width_lines() -> pd.DataFrame:
    data = json.loads(ROAD_WIDTH_LINE_SOURCE.read_text(encoding="utf-8"))
    rows = []
    for feature in data.get("features", []):
        props = feature.get("properties", {})
        lines = flatten_road_coordinates(feature.get("geometry", {}))
        points = [point for line in lines for point in line]
        if not points:
            continue
        rows.append(
            {
                "구": props.get("구"),
                "도로명": props.get("도로명"),
                "도로폭표시": props.get("표시도로폭") or props.get("API도로폭"),
                "공식도로폭m": props.get("공식도로폭평균m"),
                "공식도로폭최소m": props.get("공식도로폭최소m"),
                "공식도로폭최대m": props.get("공식도로폭최대m"),
                "도로폭공식구간수": props.get("공식구간수"),
                "공식선형길이m": props.get("공식선형길이m"),
                "폭출처": props.get("폭출처"),
                "road_lines": lines,
                "대표위도": sum(point[0] for point in points) / len(points),
                "대표경도": sum(point[1] for point in points) / len(points),
            }
        )
    road_line_df = pd.DataFrame(rows)
    for col in [
        "공식도로폭m",
        "공식도로폭최소m",
        "공식도로폭최대m",
        "도로폭공식구간수",
        "공식선형길이m",
        "대표위도",
        "대표경도",
    ]:
        road_line_df[col] = pd.to_numeric(road_line_df[col], errors="coerce")
    unmatched = load_unmatched_road_points()
    if not unmatched.empty:
        supplemental = []
        for _, row in unmatched.iterrows():
            line = synthetic_road_line(row["위도"], row["경도"])
            supplemental.append(
                {
                    "구": row["구"],
                    "도로명": row["도로명"],
                    "도로폭표시": row.get("도로폭", "-"),
                    "공식도로폭m": pd.NA,
                    "공식도로폭최소m": pd.NA,
                    "공식도로폭최대m": pd.NA,
                    "도로폭공식구간수": 1,
                    "공식선형길이m": 120,
                    "폭출처": "OSM 미매칭 도로 대표좌표 보강",
                    "road_lines": [line],
                    "대표위도": row["위도"],
                    "대표경도": row["경도"],
                }
            )
        road_line_df = pd.concat(
            [road_line_df, pd.DataFrame(supplemental)], ignore_index=True
        )
    return road_line_df


@st.cache_data
def load_road_route_lines() -> pd.DataFrame:
    data = json.loads(ROAD_ROUTE_LINE_SOURCE.read_text(encoding="utf-8"))
    rows = []
    for feature in data.get("features", []):
        props = feature.get("properties", {})
        lines = flatten_road_coordinates(feature.get("geometry", {}))
        points = [point for line in lines for point in line]
        if not points:
            continue
        rows.append(
            {
                "구": props.get("구"),
                "도로명": props.get("도로명"),
                "도로폭표시": props.get("도로폭"),
                "road_lines": lines,
                "대표위도": sum(point[0] for point in points) / len(points),
                "대표경도": sum(point[1] for point in points) / len(points),
            }
        )
    road_route_df = pd.DataFrame(rows)
    for col in ["대표위도", "대표경도"]:
        road_route_df[col] = pd.to_numeric(road_route_df[col], errors="coerce")
    unmatched = load_unmatched_road_points()
    if not unmatched.empty:
        supplemental = []
        for _, row in unmatched.iterrows():
            line = synthetic_road_line(row["위도"], row["경도"])
            supplemental.append(
                {
                    "구": row["구"],
                    "도로명": row["도로명"],
                    "도로폭표시": row.get("도로폭", "-"),
                    "road_lines": [line],
                    "대표위도": row["위도"],
                    "대표경도": row["경도"],
                }
            )
        road_route_df = pd.concat(
            [road_route_df, pd.DataFrame(supplemental)], ignore_index=True
        )
    return road_route_df


@st.cache_data
def load_route_sources() -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    route_df = read_csv(ROUTE_SOURCE)
    fire_df = read_csv(FIRE_FACILITY_SOURCE)
    road_width_df = read_csv(ROAD_WIDTH_SOURCE)
    road_line_df = load_road_width_lines()
    road_route_df = load_road_route_lines()

    for df in (route_df, fire_df, road_width_df):
        df.columns = [str(col).strip() for col in df.columns]
        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].astype("string").str.strip()

    route_numeric_cols = [
        "위도",
        "경도",
        "안전센터_유클리드m",
        "최근접_거리m",
        "이동시간초",
        "예상도착초",
        "군집",
    ]
    for col in route_numeric_cols:
        if col in route_df.columns:
            route_df[col] = pd.to_numeric(route_df[col], errors="coerce")

    for col in ["위도", "경도", "X좌표", "Y좌표"]:
        if col in fire_df.columns:
            fire_df[col] = pd.to_numeric(fire_df[col], errors="coerce")

    for col in ["공식도로폭평균m", "공식구간수"]:
        if col in road_width_df.columns:
            road_width_df[col] = pd.to_numeric(road_width_df[col], errors="coerce")

    return route_df, fire_df, road_width_df, road_line_df, road_route_df


def apply_chart_style(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=8, r=8, t=58, b=10),
        font=dict(color=COLORS["ink"], family="Arial"),
        title_font=dict(size=17),
    )
    return fig


def build_yearly_top3(df: pd.DataFrame, gu: str) -> pd.DataFrame:
    scoped = df[(df["구"] == gu) & (df["승인연도"].between(2020, 2025))].copy()
    top_dongs = scoped["동"].value_counts().head(3).index.tolist()
    yearly = (
        scoped[scoped["동"].isin(top_dongs)]
        .groupby(["승인연도", "동"])
        .size()
        .rename("신규 인허가 수")
        .reset_index()
    )
    years = pd.DataFrame({"승인연도": range(2020, 2026)})
    parts = [
        years.assign(동=dong).merge(yearly, on=["승인연도", "동"], how="left")
        for dong in top_dongs
    ]
    if not parts:
        return pd.DataFrame(columns=["승인연도", "동", "신규 인허가 수"])
    return pd.concat(parts, ignore_index=True).fillna({"신규 인허가 수": 0})


def trend_chart(df: pd.DataFrame, gu: str) -> go.Figure:
    fig = px.line(
        build_yearly_top3(df, gu),
        x="승인연도",
        y="신규 인허가 수",
        color="동",
        markers=True,
        color_discrete_sequence=[COLORS["blue"], COLORS["mint"], COLORS["coral"]],
    )
    fig.update_traces(line=dict(width=3), marker=dict(size=8))
    fig.update_layout(
        title=f"{gu} 상위 3개 동 연도별 신규 인허가 추이",
        hovermode="x unified",
        legend_title_text="",
        xaxis=dict(dtick=1, gridcolor=COLORS["line"]),
        yaxis=dict(title="신규 인허가 수", gridcolor=COLORS["line"]),
    )
    return apply_chart_style(fig)


def new_license_pie(df: pd.DataFrame, gu: str) -> go.Figure:
    pie = (
        df[(df["구"] == gu) & (df["승인연도"] == 2025)]["동"]
        .value_counts()
        .reset_index()
    )
    pie.columns = ["동", "신규 인허가 수"]
    if pie.empty:
        pie = pd.DataFrame({"동": ["2025년 데이터 없음"], "신규 인허가 수": [1]})
    fig = px.pie(
        pie,
        names="동",
        values="신규 인허가 수",
        color_discrete_sequence=[
            COLORS["blue"],
            COLORS["mint"],
            COLORS["coral"],
            COLORS["amber"],
            "#9aa8ff",
            "#8bd3ff",
        ],
        hole=0.36,
    )
    fig.update_traces(
        textposition="outside", textinfo="label+percent", pull=[0.03] * len(pie)
    )
    fig.update_layout(title=f"{gu} 2025년 신규 인허가 동별 비중", showlegend=False)
    return apply_chart_style(fig)


def license_cumulative_chart(license_df: pd.DataFrame, gu: str) -> go.Figure:
    years = list(range(2020, 2026))
    foreign = license_df[
        (license_df["구"] == gu) & (license_df["분류"] == "외국인민박")
    ].copy()
    rows = []
    for year in years:
        upto = foreign[foreign["인허가연도"] <= year]
        rows.append(
            {
                "연도": year,
                "상태": "영업중 누적",
                "건수": int((upto["영업상태명"] == "영업/정상").sum()),
            }
        )
        rows.append(
            {
                "연도": year,
                "상태": "폐업 누적",
                "건수": int((upto["영업상태명"] == "폐업").sum()),
            }
        )
    plot_df = pd.DataFrame(rows)
    fig = px.line(
        plot_df,
        x="연도",
        y="건수",
        color="상태",
        markers=True,
        color_discrete_map={
            "영업중 누적": COLORS["blue"],
            "폐업 누적": COLORS["coral"],
        },
    )
    fig.update_traces(line=dict(width=3), marker=dict(size=8))
    fig.update_layout(
        title=f"{gu} 외국인관광도시민박업 누적 현황",
        hovermode="x unified",
        xaxis=dict(dtick=1),
        yaxis_title="누적 업소 수",
    )
    return apply_chart_style(fig)


def license_dong_share_chart(license_df: pd.DataFrame, gu: str) -> go.Figure:
    foreign = license_df[
        (license_df["구"] == gu)
        & (license_df["분류"] == "외국인민박")
        & (license_df["인허가연도"].between(2020, 2025))
    ].copy()
    latest = (
        int(foreign["인허가연도"].max())
        if foreign["인허가연도"].notna().any()
        else 2025
    )
    counts = foreign[foreign["인허가연도"] == latest]["동"].value_counts()
    top = counts.head(7)
    other = counts.iloc[7:].sum()
    pie = top.reset_index()
    pie.columns = ["동", "건수"]
    if other > 0:
        pie = pd.concat(
            [pie, pd.DataFrame([{"동": "기타", "건수": int(other)}])], ignore_index=True
        )
    if pie.empty:
        pie = pd.DataFrame([{"동": "데이터 없음", "건수": 1}])
    fig = px.pie(
        pie,
        names="동",
        values="건수",
        color_discrete_sequence=[
            COLORS["blue"],
            COLORS["mint"],
            COLORS["coral"],
            COLORS["amber"],
            "#a855f7",
            "#f97316",
            "#14b8a6",
            "#64748b",
        ],
        hole=0.46,
    )
    fig.update_traces(
        textinfo="percent",
        textposition="inside",
        insidetextorientation="radial",
        hovertemplate="<b>%{label}</b><br>건수: %{value:,}건<br>비중: %{percent}<extra></extra>",
    )
    fig.update_layout(
        title=f"{latest}년 신규 인허가 동별 비중",
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.02,
            font=dict(size=11),
        ),
        margin=dict(l=4, r=112, t=58, b=8),
        height=390,
    )
    return fig


def license_category_year_chart(license_df: pd.DataFrame, gu: str) -> go.Figure:
    years = list(range(2020, 2026))
    scoped = license_df[
        (license_df["구"] == gu) & (license_df["인허가연도"].between(2020, 2025))
    ]
    plot_df = scoped.groupby(["인허가연도", "분류"]).size().rename("건수").reset_index()
    full = pd.MultiIndex.from_product(
        [years, ["외국인민박", "숙박업", "관광숙박업"]], names=["인허가연도", "분류"]
    ).to_frame(index=False)
    plot_df = full.merge(plot_df, on=["인허가연도", "분류"], how="left").fillna(
        {"건수": 0}
    )
    fig = px.bar(
        plot_df,
        x="인허가연도",
        y="건수",
        color="분류",
        barmode="group",
        color_discrete_map={
            "외국인민박": COLORS["blue"],
            "숙박업": COLORS["mint"],
            "관광숙박업": COLORS["coral"],
        },
    )
    fig.update_layout(
        title="업종별 신규 인허가 연도 비교",
        xaxis=dict(dtick=1),
        yaxis_title="건수",
        legend_title_text="",
    )
    return apply_chart_style(fig)


def license_heatmap_chart(license_df: pd.DataFrame, gu: str) -> go.Figure:
    years = list(range(2020, 2026))
    foreign = license_df[
        (license_df["구"] == gu)
        & (license_df["분류"] == "외국인민박")
        & (license_df["인허가연도"].between(2020, 2025))
    ]
    pivot = (
        foreign.groupby(["동", "인허가연도"])
        .size()
        .rename("건수")
        .reset_index()
        .pivot_table(index="동", columns="인허가연도", values="건수", fill_value=0)
        .reindex(columns=years, fill_value=0)
    )
    if pivot.empty:
        pivot = pd.DataFrame([[0] * len(years)], index=["데이터 없음"], columns=years)
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).head(10).index]
    fig = px.imshow(
        pivot,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="YlOrRd",
        labels=dict(x="연도", y="동", color="건수"),
    )
    fig.update_layout(title="동별 외국인관광도시민박업 신규 인허가 히트맵")
    return apply_chart_style(fig)


def risk_map(df: pd.DataFrame, gu: str, clusters: list[str]) -> go.Figure:
    scoped = (
        df[(df["구"] == gu) & (df["cluster_label"].isin(clusters))]
        .dropna(subset=["위도", "경도"])
        .copy()
    )
    if scoped.empty:
        return go.Figure()

    scoped["위험도"] = scoped["최종위험점수_new"].round(2)
    scoped["소화용수등급"] = scoped["최근접_소화용수_거리등급"].fillna(-1).astype(int)
    scoped["마커크기"] = (
        scoped["최종위험점수_new"]
        .fillna(scoped["최종위험점수_new"].median())
        .clip(8, 70)
    )

    fig = go.Figure()
    for label, color in CLUSTER_COLORS.items():
        part = scoped[scoped["cluster_label"] == label]
        if part.empty:
            continue
        fig.add_trace(
            go.Scattermapbox(
                lat=part["위도"],
                lon=part["경도"],
                mode="markers",
                name=label,
                marker=dict(
                    size=(part["마커크기"] / 2.8).clip(7, 18), color=color, opacity=0.82
                ),
                customdata=part[
                    ["숙소명", "동", "위험도", "업종", "소화용수등급", "도로폭위험도"]
                ],
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "동: %{customdata[1]}<br>"
                    "업종: %{customdata[3]}<br>"
                    "위험도: %{customdata[2]}점<br>"
                    "소화용수 거리등급: %{customdata[4]}<br>"
                    "도로폭위험도: %{customdata[5]:.2f}"
                    "<extra></extra>"
                ),
            )
        )

    zoom = (
        12
        if max(
            scoped["위도"].max() - scoped["위도"].min(),
            scoped["경도"].max() - scoped["경도"].min(),
        )
        < 0.05
        else 11
    )
    fig.update_layout(
        title=f"{gu} 숙박시설 위험도 지도",
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=scoped["위도"].mean(), lon=scoped["경도"].mean()),
            zoom=zoom,
        ),
        margin=dict(l=0, r=0, t=54, b=0),
        height=610,
        legend=dict(orientation="h", yanchor="bottom", y=0.01, x=0.01),
        paper_bgcolor="white",
        font=dict(color=COLORS["ink"]),
    )
    return fig


def convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    unique = sorted(set(points))
    if len(unique) <= 2:
        return unique

    def cross(
        o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]
    ) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)

    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)

    return lower[:-1] + upper[:-1]


def dong_focus_map(df: pd.DataFrame, gu: str, clusters: list[str]) -> go.Figure:
    scoped = (
        df[(df["구"] == gu) & (df["cluster_label"].isin(clusters))]
        .dropna(subset=["위도", "경도"])
        .copy()
    )
    if scoped.empty:
        return go.Figure()
    if not clusters:
        return go.Figure()

    cluster_order = [
        label for label in ["저위험군", "중위험군", "고위험군"] if label in clusters
    ]
    elegant_cluster_colors = {
        "저위험군": "#58C7A5",
        "중위험군": "#F2B84B",
        "고위험군": "#E96B6C",
    }
    elegant_cluster_fill = {
        "저위험군": "rgba(88,199,165,0.62)",
        "중위험군": "rgba(242,184,75,0.64)",
        "고위험군": "rgba(233,107,108,0.66)",
    }

    dong_summary = (
        scoped.groupby("동", as_index=False)
        .agg(
            숙박시설수=("숙소명", "count"),
            평균위험도=("최종위험점수_new", "mean"),
            최고위험도=("최종위험점수_new", "max"),
            평균소화용수등급=("최근접_소화용수_거리등급", "mean"),
        )
        .sort_values(["숙박시설수", "평균위험도"], ascending=False)
    )
    cluster_counts = (
        scoped.pivot_table(
            index="동",
            columns="cluster_label",
            values="숙소명",
            aggfunc="count",
            fill_value=0,
        )
        .reindex(columns=["저위험군", "중위험군", "고위험군"], fill_value=0)
        .reset_index()
    )
    dong_summary = dong_summary.merge(cluster_counts, on="동", how="left")
    for label in ["저위험군", "중위험군", "고위험군"]:
        if label not in dong_summary.columns:
            dong_summary[label] = 0
    selected_cluster_order = [
        label for label in ["저위험군", "중위험군", "고위험군"] if label in clusters
    ]
    dong_summary["대표위험군"] = dong_summary[selected_cluster_order].idxmax(axis=1)
    dong_summary["대표위험군수"] = (
        dong_summary[selected_cluster_order].max(axis=1).astype(int)
    )
    dong_summary["위험군구성"] = dong_summary.apply(
        lambda row: " / ".join(
            f"{label.replace('위험군', '')} {int(row[label])}개"
            for label in selected_cluster_order
        ),
        axis=1,
    )

    summary_by_dong = {str(row["동"]): row for _, row in dong_summary.iterrows()}
    fig = go.Figure()

    boundary_features = []
    for feature in dong_boundary_geo.get("features", []):
        props = feature.get("properties", {})
        feature_gu = infer_gu_name(props)
        dong_name = props.get("법정동명") or props.get("EMD_KOR_NM")
        if feature_gu == gu and dong_name in summary_by_dong:
            boundary_features.append((feature, str(dong_name)))

    label_rows = []
    for feature, dong_name in boundary_features:
        row = summary_by_dong[dong_name]
        xs, ys = polygon_line_coords(feature.get("geometry", {}))
        if not xs:
            continue
        risk_group = row["대표위험군"]
        hover_text = (
            f"<b>{dong_name}</b><br>"
            f"대표 위험군: {risk_group}<br>"
            f"숙박시설: {int(row['숙박시설수']):,}개<br>"
            f"위험군 구성: {row['위험군구성']}<br>"
            f"평균 위험도: {float(row['평균위험도']):.2f}점<br>"
            f"최고 위험도: {float(row['최고위험도']):.2f}점<br>"
            f"평균 소화용수 거리등급: {float(row['평균소화용수등급']):.2f}"
        )
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                fill="toself",
                fillcolor=elegant_cluster_fill.get(
                    risk_group, "rgba(226,232,240,0.56)"
                ),
                line=dict(color="rgba(61,78,96,0.58)", width=0.75),
                text=[hover_text] * len(xs),
                hovertemplate="%{text}<extra></extra>",
                name=dong_name,
                showlegend=False,
            )
        )
        try:
            centroid = shape(feature.get("geometry", {})).centroid
            label_rows.append(
                {
                    "동": dong_name,
                    "경도": centroid.x,
                    "위도": centroid.y,
                    "숙박시설수": row["숙박시설수"],
                }
            )
        except Exception:
            pass

    if not boundary_features:
        fig.add_trace(
            go.Scatter(
                x=scoped["경도"],
                y=scoped["위도"],
                mode="markers",
                marker=dict(
                    size=7,
                    color=scoped["cluster_label"].map(elegant_cluster_colors),
                    opacity=0.78,
                    line=dict(color="white", width=0.7),
                ),
                customdata=scoped[
                    ["숙소명", "동", "업종", "cluster_label", "최종위험점수_new"]
                ],
                hovertemplate="<b>%{customdata[0]}</b><br>법정동: %{customdata[1]}<br>업종: %{customdata[2]}<br>위험군: %{customdata[3]}<br>위험도: %{customdata[4]:.2f}점<extra></extra>",
                name="숙박시설 위치",
            )
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=scoped["경도"],
                y=scoped["위도"],
                mode="markers",
                marker=dict(
                    size=4.2,
                    color="rgba(30,41,59,0.34)",
                    line=dict(color="rgba(255,255,255,0.62)", width=0.4),
                    opacity=0.78,
                ),
                customdata=scoped[
                    ["숙소명", "동", "업종", "cluster_label", "최종위험점수_new"]
                ],
                hovertemplate="<b>%{customdata[0]}</b><br>법정동: %{customdata[1]}<br>업종: %{customdata[2]}<br>위험군: %{customdata[3]}<br>위험도: %{customdata[4]:.2f}점<extra></extra>",
                name="숙박시설 위치",
            )
        )

    if label_rows:
        label_df = (
            pd.DataFrame(label_rows).sort_values("숙박시설수", ascending=False).head(14)
        )
        fig.add_trace(
            go.Scatter(
                x=label_df["경도"],
                y=label_df["위도"],
                mode="text",
                text=label_df["동"],
                textfont=dict(size=11, color="#223044"),
                textposition="middle center",
                hoverinfo="skip",
                name="법정동명",
                showlegend=False,
            )
        )

    for label in cluster_order:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(
                    size=13, color=elegant_cluster_colors[label], symbol="square"
                ),
                name=f"{label} 우세 동",
                hoverinfo="skip",
            )
        )

    lon_values = []
    lat_values = []
    for trace in fig.data:
        xs = [v for v in getattr(trace, "x", []) if v is not None]
        ys = [v for v in getattr(trace, "y", []) if v is not None]
        lon_values.extend(xs)
        lat_values.extend(ys)
    if not lon_values or not lat_values:
        lon_values = scoped["경도"].tolist()
        lat_values = scoped["위도"].tolist()
    lon_range = max(lon_values) - min(lon_values)
    lat_range = max(lat_values) - min(lat_values)
    pad_lon = max(lon_range * 0.08, 0.003)
    pad_lat = max(lat_range * 0.08, 0.003)

    fig.update_layout(
        title=f"{gu} 법정동별 위험군 구역도",
        xaxis=dict(
            title="",
            range=[min(lon_values) - pad_lon, max(lon_values) + pad_lon],
            showgrid=False,
            showticklabels=False,
            zeroline=False,
        ),
        yaxis=dict(
            title="",
            range=[min(lat_values) - pad_lat, max(lat_values) + pad_lat],
            showgrid=False,
            showticklabels=False,
            zeroline=False,
            scaleanchor="x",
            scaleratio=1,
        ),
        height=620,
        plot_bgcolor="#f8fafc",
        paper_bgcolor="white",
        margin=dict(l=8, r=8, t=58, b=8),
        font=dict(color=COLORS["ink"]),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=0.01,
            x=0.01,
            bgcolor="rgba(255,255,255,0.86)",
            bordercolor="#e5edf5",
            borderwidth=1,
        ),
    )
    return fig


GWR_VARIABLE_COLORS = {
    "구조노후도": "#66c2a5",
    "단속위험도": "#fc8d62",
    "도로폭위험도": "#8da0cb",
    "집중도": "#e78ac3",
    "주변건물수": "#a6d854",
    "소방위험도_점수": "#ffd92f",
}


def polygon_line_coords(geometry: dict) -> tuple[list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    if not geometry:
        return xs, ys

    polygons = []
    if geometry.get("type") == "Polygon":
        polygons = [geometry.get("coordinates", [])]
    elif geometry.get("type") == "MultiPolygon":
        polygons = geometry.get("coordinates", [])

    for polygon in polygons:
        for ring in polygon[:1]:
            for lon, lat, *_ in ring:
                xs.append(lon)
                ys.append(lat)
            xs.append(None)
            ys.append(None)
    return xs, ys


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    value = hex_color.lstrip("#")
    if len(value) != 6:
        return f"rgba(179,179,179,{alpha})"
    red = int(value[0:2], 16)
    green = int(value[2:4], 16)
    blue = int(value[4:6], 16)
    return f"rgba({red},{green},{blue},{alpha})"


def sample_diverging_color(
    value: float, zmin: float, zmax: float, alpha: float = 0.62
) -> str:
    if pd.isna(value) or zmax <= zmin:
        return f"rgba(226,232,240,{alpha})"
    norm = (value - zmin) / (zmax - zmin)
    norm = min(1.0, max(0.0, float(norm)))
    rgb = px.colors.sample_colorscale("RdBu_r", [norm])[0]
    nums = [int(float(part)) for part in re.findall(r"[\d.]+", rgb)[:3]]
    return f"rgba({nums[0]},{nums[1]},{nums[2]},{alpha})"


def full_boundary_coords(boundaries: dict) -> tuple[list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    for feature in boundaries.get("features", []):
        bx, by = polygon_line_coords(feature.get("geometry", {}))
        xs.extend(bx)
        ys.extend(by)
    return xs, ys


def infer_gu_name(props: dict) -> str | None:
    gu_name = props.get("구")
    if gu_name:
        return str(gu_name)
    code = str(
        props.get("EMD_CD") or props.get("법정동코드") or props.get("join_code") or ""
    )
    return SEOUL_SIGUNGU.get(code[:5])


def build_gu_boundaries(dong_boundaries: dict) -> dict:
    grouped: dict[str, list] = {}
    for feature in dong_boundaries.get("features", []):
        gu_name = infer_gu_name(feature.get("properties", {}))
        if not gu_name:
            continue
        try:
            geom = shape(feature.get("geometry", {}))
        except Exception:
            continue
        if geom.is_empty:
            continue
        if not geom.is_valid:
            geom = geom.buffer(0)
        grouped.setdefault(gu_name, []).append(geom)

    features = []
    for gu_name, geometries in grouped.items():
        try:
            merged = unary_union(geometries)
        except Exception:
            merged = unary_union([geom.buffer(0) for geom in geometries])
        features.append(
            {
                "type": "Feature",
                "properties": {"구": gu_name},
                "geometry": merged.__geo_interface__,
            }
        )
    return {"type": "FeatureCollection", "features": features}


def add_gu_boundary_trace(
    fig: go.Figure,
    gu_boundaries: dict,
    *,
    width: float = 1.25,
    show_labels: bool = True,
) -> None:
    xs: list[float] = []
    ys: list[float] = []
    label_rows = []
    for feature in gu_boundaries.get("features", []):
        bx, by = polygon_line_coords(feature.get("geometry", {}))
        xs.extend(bx)
        ys.extend(by)
        if show_labels:
            try:
                centroid = shape(feature.get("geometry", {})).centroid
                label_rows.append(
                    {
                        "구": feature.get("properties", {}).get("구"),
                        "경도": centroid.x,
                        "위도": centroid.y,
                    }
                )
            except Exception:
                pass

    if xs:
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line=dict(color="rgba(15,23,42,0.70)", width=width),
                name="서울시 구 경계",
                hoverinfo="skip",
            )
        )
    if label_rows:
        labels = pd.DataFrame(label_rows)
        fig.add_trace(
            go.Scatter(
                x=labels["경도"],
                y=labels["위도"],
                mode="text",
                text=labels["구"],
                textfont=dict(size=11, color="#0f172a"),
                hoverinfo="skip",
                showlegend=False,
            )
        )


def grid_effect_rows(
    scoped: pd.DataFrame, grid_geo: dict, value_col: str
) -> list[dict]:
    points = [
        (float(row["경도"]), float(row["위도"]), float(row[value_col]))
        for _, row in scoped.dropna(subset=["경도", "위도", value_col]).iterrows()
    ]
    if not points:
        return []

    rows = []
    for feature in grid_geo.get("features", []):
        geometry = feature.get("geometry", {})
        if not geometry:
            continue
        try:
            polygon = shape(geometry)
        except Exception:
            continue
        minx, miny, maxx, maxy = polygon.bounds
        values = [
            value
            for lon, lat, value in points
            if minx <= lon <= maxx
            and miny <= lat <= maxy
            and polygon.covers(Point(lon, lat))
        ]
        if not values:
            continue
        props = feature.get("properties", {})
        rows.append(
            {
                "feature": feature,
                "grid_id": props.get("grid_id", "-"),
                "value": float(pd.Series(values).mean()),
                "abs_value": float(pd.Series(values).abs().mean()),
                "sample_count": len(values),
            }
        )
    return rows


def mgwr_grid_seoul_map(
    gwr_df: pd.DataFrame,
    variable: str,
    dong_boundaries: dict,
    grid_geo: dict,
    gu_boundaries: dict,
    bw_info: dict | None = None,
    risk_group: str | None = None,
    metric: str = "contribution",
    color_scale_mode: str = "relative",
) -> go.Figure:
    coef_col = f"coef_{variable}"
    contribution_col = f"contrib_{variable}"
    value_col = (
        contribution_col
        if metric == "contribution" and contribution_col in gwr_df.columns
        else coef_col
    )
    metric_label = "지역기여도" if value_col == contribution_col else "MGWR 계수"
    if value_col not in gwr_df.columns:
        return go.Figure()

    scoped = gwr_df.dropna(subset=["위도", "경도", value_col]).copy()
    if scoped.empty:
        return go.Figure()
    scoped[value_col] = pd.to_numeric(scoped[value_col], errors="coerce")
    scoped = scoped.dropna(subset=[value_col])
    if scoped.empty:
        return go.Figure()

    grid_rows = grid_effect_rows(scoped, grid_geo, value_col)
    if not grid_rows:
        return go.Figure()

    values = pd.Series([row["value"] for row in grid_rows])
    if color_scale_mode == "relative":
        center_value = float(values.median())
        color_values = values - center_value
        color_label = "평균 대비"
    else:
        center_value = 0.0
        color_values = values
        color_label = "0 기준"

    q = color_values.abs().quantile(0.95)
    z_abs = float(q) if pd.notna(q) and q > 0 else float(color_values.abs().max())
    if not z_abs or pd.isna(z_abs):
        z_abs = 1.0
    zmin, zmax = -z_abs, z_abs
    for row, color_value in zip(grid_rows, color_values):
        row["color_value"] = float(color_value)

    fig = go.Figure()

    boundary_x, boundary_y = full_boundary_coords(dong_boundaries)
    if boundary_x:
        fig.add_trace(
            go.Scatter(
                x=boundary_x,
                y=boundary_y,
                mode="lines",
                line=dict(color="rgba(51,65,85,0.28)", width=0.55),
                name="서울시 법정동 경계",
                hoverinfo="skip",
            )
        )

    add_gu_boundary_trace(fig, gu_boundaries, width=1.35, show_labels=True)

    for row in grid_rows:
        xs, ys = polygon_line_coords(row["feature"].get("geometry", {}))
        if not xs:
            continue
        fill = sample_diverging_color(row["color_value"], zmin, zmax, alpha=0.62)
        border = sample_diverging_color(row["color_value"], zmin, zmax, alpha=0.84)
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                fill="toself",
                fillcolor=fill,
                line=dict(color=border, width=0.45),
                customdata=[
                    [
                        row["grid_id"],
                        row["value"],
                        row["abs_value"],
                        row["sample_count"],
                        row["color_value"],
                    ]
                ]
                * len(xs),
                hovertemplate=(
                    "<b>격자 %{customdata[0]}</b><br>"
                    f"{variable} {metric_label}: " + "%{customdata[1]:.4f}<br>"
                    f"색상값({color_label}): " + "%{customdata[4]:.4f}<br>"
                    f"절대 {metric_label}: " + "%{customdata[2]:.4f}<br>"
                    "MGWR 표본수: %{customdata[3]:,}"
                    "<extra></extra>"
                ),
                showlegend=False,
                name="MGWR 격자",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=scoped["경도"],
            y=scoped["위도"],
            mode="markers",
            marker=dict(
                size=4,
                color="rgba(15,23,42,0.28)",
                line=dict(color="white", width=0.25),
            ),
            customdata=scoped[["구", "동", "숙소명", value_col]],
            hovertemplate=(
                "<b>%{customdata[2]}</b><br>"
                "%{customdata[0]} %{customdata[1]}<br>"
                f"{variable} {metric_label}: " + "%{customdata[3]:.4f}"
                "<extra></extra>"
            ),
            name="MGWR 표본 위치",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(
                colorscale="RdBu_r",
                cmin=zmin,
                cmax=zmax,
                color=[zmin, zmax],
                showscale=True,
                colorbar=dict(title=f"{variable}<br>{color_label}"),
            ),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    xs = []
    ys = []
    for feature in grid_geo.get("features", []):
        gx, gy = polygon_line_coords(feature.get("geometry", {}))
        xs.extend([x for x in gx if x is not None])
        ys.extend([y for y in gy if y is not None])
    if not xs or not ys:
        xs = scoped["경도"].tolist()
        ys = scoped["위도"].tolist()
    lon_range = max(xs) - min(xs)
    lat_range = max(ys) - min(ys)
    pad_lon = max(lon_range * 0.025, 0.01)
    pad_lat = max(lat_range * 0.025, 0.01)
    bw_title = ""
    if bw_info:
        bw_title = f" | MGWR BW={bw_info.get('BW', '-'):.0f} NN ({bw_info.get('공간범위', '-')})"
    grid_size = (
        grid_geo.get("features", [{}])[0].get("properties", {}).get("cell_size_m", "-")
    )
    fig.update_layout(
        title=f"서울시 전체 법정동 경계 + {grid_size}m 격자 {risk_group or ''} {variable} MGWR {metric_label} ({color_label}){bw_title}",
        xaxis=dict(
            title="",
            range=[min(xs) - pad_lon, max(xs) + pad_lon],
            showgrid=False,
            showticklabels=False,
            zeroline=False,
        ),
        yaxis=dict(
            title="",
            range=[min(ys) - pad_lat, max(ys) + pad_lat],
            showgrid=False,
            showticklabels=False,
            zeroline=False,
            scaleanchor="x",
            scaleratio=1,
        ),
        height=760,
        plot_bgcolor="#fbfcff",
        paper_bgcolor="white",
        margin=dict(l=8, r=8, t=66, b=8),
        font=dict(color=COLORS["ink"]),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=0.01,
            x=0.01,
            bgcolor="rgba(255,255,255,0.86)",
        ),
        uirevision="mgwr_grid_seoul_map",
    )
    return fig


def gwr_variable_seoul_map(
    gwr_df: pd.DataFrame,
    variable: str,
    dong_boundaries: dict,
    gu_boundaries: dict | None = None,
    bw_info: dict | None = None,
    risk_group: str | None = None,
    model_label: str = "MGWR",
    metric: str = "contribution",
    color_scale_mode: str = "relative",
) -> go.Figure:
    coef_col = f"coef_{variable}"
    contribution_col = f"contrib_{variable}"
    z_col = (
        contribution_col
        if metric == "contribution" and contribution_col in gwr_df.columns
        else coef_col
    )
    metric_label = "지역기여도" if z_col == contribution_col else "평균계수"
    if z_col not in gwr_df.columns:
        return go.Figure()

    scoped = gwr_df.dropna(subset=["위도", "경도", "구", "동", z_col]).copy()
    if scoped.empty:
        return go.Figure()

    scoped[z_col] = pd.to_numeric(scoped[z_col], errors="coerce")
    scoped = scoped.dropna(subset=[z_col])
    if scoped.empty:
        return go.Figure()

    dong_effect = scoped.groupby(["구", "동"], as_index=False).agg(
        평균계수=(z_col, "mean"),
        절대평균계수=(z_col, lambda s: s.abs().mean()),
        평균MGWR계수=(coef_col, "mean")
        if coef_col in scoped.columns
        else (z_col, "mean"),
        표본수=("동", "size"),
        경도=("경도", "mean"),
        위도=("위도", "mean"),
    )
    fallback = (
        dong_effect.sort_values("표본수", ascending=False)
        .drop_duplicates("동")
        .set_index("동")
        .to_dict("index")
    )
    effect_by_key = {(row["구"], row["동"]): row for _, row in dong_effect.iterrows()}

    values = dong_effect["평균계수"]
    if color_scale_mode == "relative":
        center_value = float(values.median())
        color_values = values - center_value
        color_label = "평균 대비"
    else:
        center_value = 0.0
        color_values = values
        color_label = "0 기준"
    dong_effect["색상값"] = color_values
    effect_by_key = {(row["구"], row["동"]): row for _, row in dong_effect.iterrows()}
    fallback = (
        dong_effect.sort_values("표본수", ascending=False)
        .drop_duplicates("동")
        .set_index("동")
        .to_dict("index")
    )

    q = color_values.abs().quantile(0.95)
    z_abs = float(q) if pd.notna(q) and q > 0 else float(color_values.abs().max())
    if not z_abs or pd.isna(z_abs):
        z_abs = 1.0
    zmin, zmax = -z_abs, z_abs

    fig = go.Figure()
    used_dongs = set()
    for feature in dong_boundaries.get("features", []):
        props = feature.get("properties", {})
        gu_name = props.get("구")
        dong_name = props.get("법정동명") or props.get("EMD_KOR_NM")
        if not dong_name:
            continue
        row = effect_by_key.get((gu_name, dong_name))
        if row is None:
            row = fallback.get(dong_name)
        if row is None:
            continue

        xs, ys = polygon_line_coords(feature.get("geometry", {}))
        if not xs:
            continue
        used_dongs.add(dong_name)
        value = float(row["평균계수"])
        color_value = float(row["색상값"])
        fill = sample_diverging_color(color_value, zmin, zmax, alpha=0.58)
        border = sample_diverging_color(color_value, zmin, zmax, alpha=0.86)
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                fill="toself",
                fillcolor=fill,
                line=dict(color=border, width=0.9),
                name=f"{dong_name}",
                customdata=[
                    [
                        row.get("구", gu_name),
                        dong_name,
                        value,
                        row["절대평균계수"],
                        row["표본수"],
                        row["평균MGWR계수"],
                        color_value,
                    ]
                ]
                * len(xs),
                hovertemplate=(
                    "<b>%{customdata[0]} %{customdata[1]}</b><br>"
                    f"{variable} {metric_label}: " + "%{customdata[2]:.4f}<br>"
                    f"색상값({color_label}): " + "%{customdata[6]:.4f}<br>"
                    f"절대{metric_label}: " + "%{customdata[3]:.4f}<br>"
                    "MGWR 계수: %{customdata[5]:.4f}<br>"
                    "표본수: %{customdata[4]:,}"
                    "<extra></extra>"
                ),
                showlegend=False,
            )
        )

    boundary_x: list[float] = []
    boundary_y: list[float] = []
    for feature in dong_boundaries.get("features", []):
        props = feature.get("properties", {})
        dong_name = props.get("법정동명") or props.get("EMD_KOR_NM")
        if dong_name not in used_dongs:
            continue
        xs, ys = polygon_line_coords(feature.get("geometry", {}))
        boundary_x.extend(xs)
        boundary_y.extend(ys)
    if boundary_x:
        fig.add_trace(
            go.Scatter(
                x=boundary_x,
                y=boundary_y,
                mode="lines",
                line=dict(color="rgba(30,41,59,0.45)", width=0.7),
                name="법정동 경계",
                hoverinfo="skip",
            )
        )

    if gu_boundaries:
        add_gu_boundary_trace(fig, gu_boundaries, width=1.45, show_labels=True)

    gu_summary = dong_effect.groupby("구", as_index=False).agg(
        평균계수=("평균계수", "mean"),
        평균절대계수=("절대평균계수", "mean"),
        법정동수=("동", "nunique"),
        표본수=("표본수", "sum"),
        경도=("경도", "mean"),
        위도=("위도", "mean"),
    )
    top_pos = (
        dong_effect.sort_values("평균계수", ascending=False)
        .groupby("구")
        .head(1)[["구", "동", "평균계수"]]
        .rename(columns={"동": "양의영향동", "평균계수": "양의영향계수"})
    )
    top_neg = (
        dong_effect.sort_values("평균계수", ascending=True)
        .groupby("구")
        .head(1)[["구", "동", "평균계수"]]
        .rename(columns={"동": "음의영향동", "평균계수": "음의영향계수"})
    )
    gu_summary = gu_summary.merge(top_pos, on="구", how="left").merge(
        top_neg, on="구", how="left"
    )
    gu_summary["표시크기"] = (gu_summary["표본수"] ** 0.5 * 2.2).clip(18, 48)
    fig.add_trace(
        go.Scatter(
            x=gu_summary["경도"],
            y=gu_summary["위도"],
            mode="markers+text",
            text=gu_summary["구"],
            textposition="middle center",
            textfont=dict(size=13, color="#0f172a"),
            marker=dict(
                size=gu_summary["표시크기"],
                color="rgba(255,255,255,0.62)",
                line=dict(color="rgba(15,23,42,0.55)", width=1.5),
            ),
            customdata=gu_summary[
                [
                    "구",
                    "법정동수",
                    "표본수",
                    "평균계수",
                    "평균절대계수",
                    "양의영향동",
                    "양의영향계수",
                    "음의영향동",
                    "음의영향계수",
                ]
            ],
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                f"선택 변수: {variable}<br>"
                f"표시값: {metric_label}<br>"
                "법정동 수: %{customdata[1]:,}<br>"
                f"{model_label} 표본수: " + "%{customdata[2]:,}<br>"
                "구 평균계수: %{customdata[3]:.4f}<br>"
                "구 평균 절대계수: %{customdata[4]:.4f}<br>"
                "양의 영향 최대 동: %{customdata[5]} (%{customdata[6]:.4f})<br>"
                "음의 영향 최대 동: %{customdata[7]} (%{customdata[8]:.4f})"
                "<extra></extra>"
            ),
            name="구별 요약",
        )
    )

    top_labels = dong_effect.sort_values("표본수", ascending=False).head(25)
    fig.add_trace(
        go.Scatter(
            x=top_labels["경도"],
            y=top_labels["위도"],
            mode="text",
            text=top_labels["동"],
            textfont=dict(size=10, color="#1f2b3d"),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=scoped["경도"],
            y=scoped["위도"],
            mode="markers",
            marker=dict(
                size=4, color="rgba(31,43,61,0.26)", line=dict(color="white", width=0.3)
            ),
            customdata=scoped[["구", "동", "숙소명", z_col]],
            hovertemplate=(
                "<b>%{customdata[2]}</b><br>"
                "%{customdata[0]} %{customdata[1]}<br>"
                f"{variable} {metric_label}: " + "%{customdata[3]:.4f}"
                "<extra></extra>"
            ),
            name=f"{model_label} 표본 위치",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(
                colorscale="RdBu_r",
                cmin=zmin,
                cmax=zmax,
                color=[zmin, zmax],
                showscale=True,
                colorbar=dict(title=f"{variable}<br>{color_label}"),
            ),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    lon_range = scoped["경도"].max() - scoped["경도"].min()
    lat_range = scoped["위도"].max() - scoped["위도"].min()
    pad_lon = max(lon_range * 0.04, 0.01)
    pad_lat = max(lat_range * 0.04, 0.01)
    bw_title = ""
    if bw_info:
        bw_title = f" | MGWR BW={bw_info.get('BW', '-'):.0f} NN ({bw_info.get('공간범위', '-')})"
    group_title = f"{risk_group} " if risk_group else ""
    fig.update_layout(
        title=f"서울시 법정동별 {group_title}{variable} {model_label} {metric_label} ({color_label}){bw_title}",
        xaxis=dict(
            title="",
            range=[scoped["경도"].min() - pad_lon, scoped["경도"].max() + pad_lon],
            showgrid=False,
            showticklabels=False,
            zeroline=False,
        ),
        yaxis=dict(
            title="",
            range=[scoped["위도"].min() - pad_lat, scoped["위도"].max() + pad_lat],
            showgrid=False,
            showticklabels=False,
            zeroline=False,
            scaleanchor="x",
            scaleratio=1,
        ),
        height=720,
        plot_bgcolor="#fbfcff",
        paper_bgcolor="white",
        margin=dict(l=8, r=8, t=62, b=8),
        font=dict(color=COLORS["ink"]),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=0.01,
            x=0.01,
            bgcolor="rgba(255,255,255,0.86)",
        ),
        uirevision="gwr_variable_seoul_map",
    )
    return fig


def gwr_mgwr_map(gwr_df: pd.DataFrame, gu: str, dong_boundaries: dict) -> go.Figure:
    scoped = gwr_df[gwr_df["구"] == gu].dropna(subset=["위도", "경도"]).copy()
    if scoped.empty:
        return go.Figure()

    fig = go.Figure()
    scoped_dongs = set(scoped["동"].dropna().astype(str))
    boundary_x: list[float] = []
    boundary_y: list[float] = []
    for feature in dong_boundaries.get("features", []):
        props = feature.get("properties", {})
        dong_name = props.get("법정동명") or props.get("EMD_KOR_NM")
        if dong_name not in scoped_dongs:
            continue
        xs, ys = polygon_line_coords(feature.get("geometry", {}))
        boundary_x.extend(xs)
        boundary_y.extend(ys)
    if boundary_x:
        fig.add_trace(
            go.Scatter(
                x=boundary_x,
                y=boundary_y,
                mode="lines",
                line=dict(color="rgba(51,65,85,0.50)", width=1.1),
                name="법정동 경계",
                hoverinfo="skip",
            )
        )

    hull_points = convex_hull(
        list(zip(scoped["경도"].tolist(), scoped["위도"].tolist()))
    )
    if len(hull_points) >= 3:
        hull_x = [point[0] for point in hull_points] + [hull_points[0][0]]
        hull_y = [point[1] for point in hull_points] + [hull_points[0][1]]
        fig.add_trace(
            go.Scatter(
                x=hull_x,
                y=hull_y,
                mode="lines",
                fill="toself",
                fillcolor="rgba(148,163,184,0.10)",
                line=dict(color="rgba(71,85,105,0.55)", width=1.4),
                name=f"{gu} 분석 범위",
                hoverinfo="skip",
            )
        )

    coef_cols = [col for col in scoped.columns if col.startswith("coef_")]
    dong_summary = (
        scoped.groupby("동", as_index=False)
        .agg(
            경도=("경도", "mean"),
            위도=("위도", "mean"),
            표본수=("동", "size"),
        )
        .dropna(subset=["동"])
    )
    if coef_cols and not dong_summary.empty:
        dominant = (
            scoped.groupby("동")[coef_cols]
            .apply(lambda frame: frame.abs().mean().idxmax().replace("coef_", ""))
            .rename("주요설명변수")
            .reset_index()
        )
        dong_summary = dong_summary.merge(dominant, on="동", how="left")
        top_dongs = (
            dong_summary.sort_values("표본수", ascending=False).head(12)["동"].tolist()
        )
        dong_summary["표시크기"] = (dong_summary["표본수"] ** 0.5 * 11).clip(16, 54)
        dong_variable = dict(
            zip(
                dong_summary["동"].astype(str), dong_summary["주요설명변수"].astype(str)
            )
        )

        for feature in dong_boundaries.get("features", []):
            props = feature.get("properties", {})
            dong_name = props.get("법정동명") or props.get("EMD_KOR_NM")
            if dong_name not in dong_variable:
                continue
            variable = dong_variable[dong_name]
            color = GWR_VARIABLE_COLORS.get(variable, "#b3b3b3")
            xs, ys = polygon_line_coords(feature.get("geometry", {}))
            if not xs:
                continue
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines",
                    fill="toself",
                    fillcolor=hex_to_rgba(color, 0.24),
                    line=dict(color=hex_to_rgba(color, 0.52), width=0.8),
                    name=f"{dong_name} {variable}",
                    hovertemplate=(
                        f"<b>{dong_name}</b><br>"
                        f"주요 설명 변수: {variable}"
                        "<extra></extra>"
                    ),
                    showlegend=False,
                )
            )

        fig.add_trace(
            go.Scatter(
                x=dong_summary["경도"],
                y=dong_summary["위도"],
                mode="markers+text",
                text=dong_summary["동"].where(dong_summary["동"].isin(top_dongs), ""),
                textposition="top center",
                textfont=dict(size=12, color="#1f2b3d"),
                marker=dict(
                    size=dong_summary["표시크기"],
                    color=dong_summary["주요설명변수"]
                    .map(GWR_VARIABLE_COLORS)
                    .fillna("#b3b3b3"),
                    line=dict(color="white", width=2),
                    opacity=0.9,
                ),
                customdata=dong_summary[["동", "주요설명변수", "표본수"]],
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "주요 설명 변수: %{customdata[1]}<br>"
                    "표본수: %{customdata[2]:,}"
                    "<extra></extra>"
                ),
                name="법정동 주요 설명 변수",
            )
        )

        for variable, color in GWR_VARIABLE_COLORS.items():
            if variable in dong_summary["주요설명변수"].dropna().unique():
                fig.add_trace(
                    go.Scatter(
                        x=[None],
                        y=[None],
                        mode="markers",
                        marker=dict(size=12, color=color),
                        name=variable,
                        hoverinfo="skip",
                    )
                )

    fig.add_trace(
        go.Scatter(
            x=scoped["경도"],
            y=scoped["위도"],
            mode="markers",
            marker=dict(
                size=5, color="rgba(31,43,61,0.32)", line=dict(color="white", width=0.4)
            ),
            customdata=scoped[["동", "숙소명", "cluster_label"]],
            hovertemplate=(
                "<b>%{customdata[1]}</b><br>"
                "법정동: %{customdata[0]}<br>"
                "위험군: %{customdata[2]}"
                "<extra></extra>"
            ),
            name="GWR 표본 위치",
        )
    )

    lon_range = scoped["경도"].max() - scoped["경도"].min()
    lat_range = scoped["위도"].max() - scoped["위도"].min()
    pad_lon = max(lon_range * 0.08, 0.003)
    pad_lat = max(lat_range * 0.08, 0.003)
    fig.update_layout(
        title=f"{gu} 동별 주요 설명 변수 및 MGWR BW 해석",
        xaxis=dict(
            title="",
            range=[scoped["경도"].min() - pad_lon, scoped["경도"].max() + pad_lon],
            showgrid=False,
            showticklabels=False,
            zeroline=False,
        ),
        yaxis=dict(
            title="",
            range=[scoped["위도"].min() - pad_lat, scoped["위도"].max() + pad_lat],
            showgrid=False,
            showticklabels=False,
            zeroline=False,
            scaleanchor="x",
            scaleratio=1,
        ),
        height=650,
        plot_bgcolor="#fbfcff",
        paper_bgcolor="white",
        margin=dict(l=8, r=8, t=62, b=8),
        font=dict(color=COLORS["ink"]),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=0.01,
            x=0.01,
            bgcolor="rgba(255,255,255,0.86)",
        ),
    )
    return fig


def cluster_top_areas(df: pd.DataFrame) -> pd.DataFrame:
    top = (
        df.groupby(["cluster_label", "구", "동"])
        .size()
        .rename("숙박시설 수")
        .reset_index()
        .sort_values(["cluster_label", "숙박시설 수"], ascending=[True, False])
    )
    return top.groupby("cluster_label").head(3).reset_index(drop=True)


def risk_factor_chart(df: pd.DataFrame, gu: str) -> go.Figure:
    cols = [
        "소방위험도_점수",
        "단속위험도",
        "구조노후도",
        "도로폭위험도",
        "집중도",
        "주변건물수",
    ]
    scoped = df[df["구"] == gu].copy()
    rows = []
    for col in cols:
        gu_mean = scoped[col].mean()
        all_mean = df[col].mean()
        rows.append(
            {
                "위험요인": col,
                "선택구 평균": gu_mean,
                "전체 평균": all_mean,
                "전체평균대비": (gu_mean / all_mean * 100) if all_mean else 0,
            }
        )
    factor_df = pd.DataFrame(rows)
    factor_df["차이"] = factor_df["전체평균대비"] - 100
    factor_df = factor_df.sort_values("전체평균대비", ascending=True)
    fig = px.bar(
        factor_df,
        x="전체평균대비",
        y="위험요인",
        orientation="h",
        color="차이",
        color_continuous_scale=[[0, "#67c8a2"], [0.5, "#f3bd4f"], [1, "#ef7777"]],
        custom_data=["선택구 평균", "전체 평균", "전체평균대비"],
        text=factor_df["전체평균대비"].map(lambda v: f"{v:.0f}"),
    )
    fig.update_traces(
        textposition="outside",
        hovertemplate=(
            "<b>%{y}</b><br>"
            "선택 구 평균: %{customdata[0]:.3f}<br>"
            "전체 평균: %{customdata[1]:.3f}<br>"
            "전체 평균 대비: %{customdata[2]:.1f}%"
            "<extra></extra>"
        ),
    )
    fig.add_vline(x=100, line_width=2, line_dash="dash", line_color="#94a3b8")
    fig.update_layout(
        title=f"{gu} 위험요인 프로파일: 전체 평균=100",
        coloraxis_showscale=False,
        xaxis_title="전체 평균 대비 지수",
        yaxis_title="",
        xaxis=dict(
            range=[0, max(160, factor_df["전체평균대비"].max() * 1.18)],
            gridcolor=COLORS["line"],
        ),
    )
    return apply_chart_style(fig)


def risk_factor_table(df: pd.DataFrame, gu: str) -> pd.DataFrame:
    cols = [
        "소방위험도_점수",
        "단속위험도",
        "구조노후도",
        "도로폭위험도",
        "집중도",
        "주변건물수",
    ]
    scoped = df[df["구"] == gu].copy()
    rows = []
    for col in cols:
        gu_mean = scoped[col].mean()
        all_mean = df[col].mean()
        rows.append(
            {
                "위험요인": col,
                "선택구 평균": round(gu_mean, 3),
                "전체 평균": round(all_mean, 3),
                "전체평균대비": round((gu_mean / all_mean * 100) if all_mean else 0, 1),
            }
        )
    return pd.DataFrame(rows).sort_values("전체평균대비", ascending=False)


def point_segment_distance_m(
    lat: float, lon: float, a: tuple[float, float], b: tuple[float, float]
) -> float:
    mean_lat = math.radians((lat + a[0] + b[0]) / 3)
    px = lon * 111_320 * math.cos(mean_lat)
    py = lat * 110_540
    ax = a[1] * 111_320 * math.cos(mean_lat)
    ay = a[0] * 110_540
    bx = b[1] * 111_320 * math.cos(mean_lat)
    by = b[0] * 110_540
    dx = bx - ax
    dy = by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0, min(1, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def nearest_point_on_segment(
    lat: float, lon: float, a: tuple[float, float], b: tuple[float, float]
) -> tuple[float, float]:
    mean_lat = math.radians((lat + a[0] + b[0]) / 3)
    scale_x = 111_320 * math.cos(mean_lat)
    scale_y = 110_540
    px = lon * scale_x
    py = lat * scale_y
    ax = a[1] * scale_x
    ay = a[0] * scale_y
    bx = b[1] * scale_x
    by = b[0] * scale_y
    dx = bx - ax
    dy = by - ay
    if dx == 0 and dy == 0:
        return a
    t = max(0, min(1, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    return ((ay + t * dy) / scale_y, (ax + t * dx) / scale_x)


def point_distance_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    mean_lat = math.radians((a[0] + b[0]) / 2)
    dx = (a[1] - b[1]) * 111_320 * math.cos(mean_lat)
    dy = (a[0] - b[0]) * 110_540
    return math.hypot(dx, dy)


def build_road_graph(
    road_line_df: pd.DataFrame, gu: str
) -> tuple[
    dict[tuple[float, float], list[tuple[tuple[float, float], float]]],
    list[tuple[float, float]],
]:
    graph: dict[tuple[float, float], list[tuple[tuple[float, float], float]]] = {}
    for _, road in road_line_df[road_line_df["구"] == gu].iterrows():
        for line in road["road_lines"]:
            rounded = [(round(lat, 4), round(lon, 4)) for lat, lon in line]
            for start, end in zip(rounded, rounded[1:]):
                distance = point_distance_m(start, end)
                graph.setdefault(start, []).append((end, distance))
                graph.setdefault(end, []).append((start, distance))
    nodes = list(graph.keys())
    grid: dict[tuple[int, int], list[tuple[float, float]]] = {}
    cell_size = 0.0011
    for node in nodes:
        cell = (int(node[0] / cell_size), int(node[1] / cell_size))
        grid.setdefault(cell, []).append(node)
    for node in nodes:
        cell = (int(node[0] / cell_size), int(node[1] / cell_size))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for other in grid.get((cell[0] + dx, cell[1] + dy), []):
                    if other <= node:
                        continue
                    distance = point_distance_m(node, other)
                    if 0 < distance <= 120:
                        graph.setdefault(node, []).append((other, distance))
                        graph.setdefault(other, []).append((node, distance))
    return graph, list(graph.keys())


def nearest_graph_node(
    point: tuple[float, float], nodes: list[tuple[float, float]]
) -> tuple[float, float] | None:
    if not nodes:
        return None
    return min(nodes, key=lambda node: point_distance_m(point, node))


def nearest_fire_center(row: pd.Series, fire_df: pd.DataFrame) -> pd.Series | None:
    center_names = [
        row.get("최근접_안전센터", pd.NA),
        row.get("담당_안전센터", pd.NA),
    ]
    candidates = fire_df.dropna(subset=["위도", "경도"]).copy()
    for center_name in center_names:
        if pd.isna(center_name):
            continue
        matched = candidates[candidates["시설명"] == center_name]
        if not matched.empty:
            return matched.iloc[0]

    gu = str(row.get("구", ""))
    same_gu = candidates[
        candidates.get("관할구역", "").astype(str).str.contains(gu, na=False)
    ]
    if not same_gu.empty:
        candidates = same_gu
    if candidates.empty or pd.isna(row.get("위도")) or pd.isna(row.get("경도")):
        return None

    candidates = candidates.copy()
    candidates["_거리m"] = candidates.apply(
        lambda center: point_distance_m(
            (row["위도"], row["경도"]), (center["위도"], center["경도"])
        ),
        axis=1,
    )
    return candidates.nsmallest(1, "_거리m").iloc[0]


def shortest_graph_path(
    start_point: tuple[float, float],
    end_point: tuple[float, float],
    gu: str,
    road_line_df: pd.DataFrame,
) -> tuple[list[tuple[float, float]], float | None]:
    graph, nodes = build_road_graph(road_line_df, gu)
    start = nearest_graph_node(start_point, nodes)
    end = nearest_graph_node(end_point, nodes)
    if start is None or end is None:
        return [], None

    queue: list[tuple[float, tuple[float, float]]] = [(0.0, start)]
    distances = {start: 0.0}
    previous: dict[tuple[float, float], tuple[float, float]] = {}
    while queue:
        distance, node = heapq.heappop(queue)
        if node == end:
            break
        if distance > distances.get(node, float("inf")):
            continue
        for neighbor, weight in graph.get(node, []):
            new_distance = distance + weight
            if new_distance < distances.get(neighbor, float("inf")):
                distances[neighbor] = new_distance
                previous[neighbor] = node
                heapq.heappush(queue, (new_distance, neighbor))

    if end not in distances:
        fallback_distance = (
            point_distance_m(start_point, start)
            + point_distance_m(start, end)
            + point_distance_m(end_point, end)
        )
        return [start, end], fallback_distance

    path = [end]
    while path[-1] != start:
        path.append(previous[path[-1]])
    path.reverse()
    distance_with_access = (
        distances[end]
        + point_distance_m(start_point, start)
        + point_distance_m(end_point, end)
    )
    return path, distance_with_access


def adjacent_road_anchor(row: pd.Series) -> tuple[float, float] | None:
    road_lines = row.get("road_lines", [])
    if not isinstance(road_lines, list) or not road_lines:
        return None
    best_point = None
    best_distance = float("inf")
    for line in road_lines:
        for start, end in zip(line, line[1:]):
            point = nearest_point_on_segment(row["위도"], row["경도"], start, end)
            distance = point_distance_m((row["위도"], row["경도"]), point)
            if distance < best_distance:
                best_point = point
                best_distance = distance
    return best_point


def shortest_road_path(
    row: pd.Series,
    center_row: pd.Series,
    road_line_df: pd.DataFrame,
) -> tuple[list[tuple[float, float]], float | None]:
    start_point = (row["위도"], row["경도"])
    end_point = (center_row["위도"], center_row["경도"])
    anchor = adjacent_road_anchor(row) or start_point
    graph_path, graph_distance = shortest_graph_path(
        anchor, end_point, row["구"], road_line_df
    )
    if not graph_path:
        return [], None
    path = [start_point]
    if point_distance_m(start_point, anchor) > 5:
        path.append(anchor)
    path.extend(graph_path)
    path.append(end_point)
    distance = (graph_distance or 0) + point_distance_m(start_point, anchor)
    return path, distance


def dispatch_arrival_path(
    row: pd.Series,
    center_row: pd.Series,
    road_line_df: pd.DataFrame,
) -> tuple[list[tuple[float, float]], float | None, tuple[float, float] | None]:
    path, distance = shortest_road_path(row, center_row, road_line_df)
    anchor = adjacent_road_anchor(row)
    return list(reversed(path)) if path else [], distance, anchor


def nearest_road_width(row: pd.Series, road_line_df: pd.DataFrame) -> dict:
    candidates = road_line_df[road_line_df["구"] == row["구"]].copy()
    if candidates.empty or pd.isna(row.get("위도")) or pd.isna(row.get("경도")):
        return {}

    candidates["대표거리"] = (
        ((candidates["대표위도"] - row["위도"]) * 110_540) ** 2
        + (
            (candidates["대표경도"] - row["경도"])
            * 111_320
            * math.cos(math.radians(row["위도"]))
        )
        ** 2
    ) ** 0.5
    candidates = candidates.nsmallest(80, "대표거리")

    best = None
    best_distance = float("inf")
    for _, road in candidates.iterrows():
        for line in road["road_lines"]:
            for start, end in zip(line, line[1:]):
                distance = point_segment_distance_m(
                    row["위도"], row["경도"], start, end
                )
                if distance < best_distance:
                    best = road
                    best_distance = distance
    if best is None:
        return {}

    return {
        "인접도로명": best["도로명"],
        "도로폭표시": best["도로폭표시"],
        "공식도로폭m": best["공식도로폭m"],
        "공식도로폭최소m": best["공식도로폭최소m"],
        "공식도로폭최대m": best["공식도로폭최대m"],
        "도로폭공식구간수": best["도로폭공식구간수"],
        "공식선형길이m": best["공식선형길이m"],
        "폭출처": best["폭출처"],
        "road_lines": best["road_lines"],
        "도로폭매칭거리m": best_distance,
    }


def enrich_representative_roads(
    representatives: pd.DataFrame, road_line_df: pd.DataFrame
) -> pd.DataFrame:
    enriched = representatives.copy().reset_index(drop=True)
    road_keys = [
        "인접도로명",
        "도로폭표시",
        "공식도로폭m",
        "공식도로폭최소m",
        "공식도로폭최대m",
        "도로폭공식구간수",
        "공식선형길이m",
        "폭출처",
        "road_lines",
        "도로폭매칭거리m",
    ]
    for key in road_keys:
        if key not in enriched.columns:
            enriched[key] = pd.Series([pd.NA] * len(enriched), dtype="object")
    for idx, row in enriched.iterrows():
        road = nearest_road_width(row, road_line_df)
        for key, value in road.items():
            if key in road_keys:
                enriched.at[idx, key] = value
    return enriched


def representative_route_facilities(
    final_df: pd.DataFrame,
    route_df: pd.DataFrame,
    road_width_df: pd.DataFrame,
    road_line_df: pd.DataFrame,
    gu: str,
    cluster_label: str,
) -> pd.DataFrame:
    route_cols = [
        "구",
        "동",
        "업소명",
        "최근접_안전센터",
        "안전센터_유클리드m",
        "최근접_거리m",
        "이동시간초",
        "예상도착초",
        "담당_안전센터",
        "인접도로명",
        "도로폭_보정이동시간초",
        "도로폭_보정예상도착초",
    ]
    route_lookup = route_df[
        [col for col in route_cols if col in route_df.columns]
    ].copy()
    width_lookup = road_width_df[
        ["구", "도로명", "표시도로폭", "공식도로폭평균m", "공식구간수"]
    ].copy()
    width_lookup = width_lookup.rename(
        columns={
            "도로명": "인접도로명",
            "표시도로폭": "도로폭표시",
            "공식도로폭평균m": "공식도로폭m",
            "공식구간수": "도로폭공식구간수",
        }
    )
    scoped = final_df[
        (final_df["구"] == gu) & (final_df["cluster_label"] == cluster_label)
    ].copy()
    if scoped.empty:
        return scoped

    top_dongs = scoped["동"].dropna().value_counts().head(2)
    if top_dongs.empty:
        return scoped.iloc[0:0].copy()

    selected_parts = []
    for rank, (dong, facility_count) in enumerate(top_dongs.items(), start=1):
        take_n = 3 if rank == 1 else 2
        part = (
            scoped[scoped["동"] == dong]
            .sort_values(["최종위험점수_new"], ascending=False)
            .head(take_n)
            .copy()
        )
        part["대표동순위"] = rank
        part["동시설수"] = int(facility_count)
        selected_parts.append(part)

    scoped = (
        pd.concat(selected_parts, ignore_index=True)
        if selected_parts
        else scoped.iloc[0:0].copy()
    )
    if scoped.empty:
        return scoped

    scoped = scoped.merge(
        route_lookup,
        left_on=["구", "동", "숙소명"],
        right_on=["구", "동", "업소명"],
        how="left",
    )
    scoped = scoped.merge(width_lookup, on=["구", "인접도로명"], how="left")
    scoped["업소명"] = scoped["업소명"].fillna(scoped["숙소명"])
    scoped = scoped.dropna(subset=["위도", "경도"])
    if scoped.empty:
        return scoped

    scoped = scoped.sort_values(
        ["대표동순위", "최종위험점수_new"], ascending=[True, False]
    ).reset_index(drop=True)
    scoped = enrich_representative_roads(scoped, road_line_df)
    scoped["선택라벨"] = (
        scoped["동"].fillna("")
        + "("
        + scoped["동시설수"].astype("string")
        + "개 시설) · "
        + scoped["숙소명"].fillna("")
        + " · "
        + scoped["최종위험점수_new"].round(1).astype("string")
        + "점"
    )
    return scoped


def route_seconds_label(seconds: float | int | None) -> str:
    if pd.isna(seconds):
        return "-"
    minutes = float(seconds) / 60
    if minutes < 1:
        return f"{float(seconds):.0f}초"
    return f"{minutes:.1f}분"


def route_seconds_only_label(seconds: float | int | None) -> str:
    if pd.isna(seconds):
        return "-"
    return f"{float(seconds):,.0f}초"


def value_or_dash(value: object, suffix: str = "", decimals: int = 1) -> str:
    if pd.isna(value):
        return "-"
    if isinstance(value, (int, float)):
        return f"{float(value):,.{decimals}f}{suffix}"
    return f"{value}{suffix}"


def road_width_adjusted_seconds(
    distance_m: float | int | None,
    speed_factor: float | int | None,
    arrival_buffer_seconds: float = 60,
) -> tuple[float | pd.NA, float | pd.NA]:
    if pd.isna(distance_m):
        return pd.NA, pd.NA
    factor = 1 if pd.isna(speed_factor) else float(speed_factor)
    move_seconds = float(distance_m) * 0.12 * factor
    return move_seconds, move_seconds + arrival_buffer_seconds


def selected_route_comparison(
    row: pd.Series, fire_df: pd.DataFrame, road_line_df: pd.DataFrame
) -> dict:
    center_row = nearest_fire_center(row, fire_df)
    straight = row.get("안전센터_유클리드m", row.get("최근접_거리m", pd.NA))
    if center_row is None:
        return {
            "도로망추정거리m": pd.NA,
            "직선거리m": straight,
            "거리차m": pd.NA,
            "우회율": pd.NA,
            "도로망보정이동시간초": pd.NA,
            "도로망보정예상도착초": pd.NA,
        }
    _, road_distance = shortest_road_path(row, center_row, road_line_df)
    ratio = (
        (road_distance / straight)
        if road_distance and pd.notna(straight) and straight
        else pd.NA
    )
    diff = (road_distance - straight) if road_distance and pd.notna(straight) else pd.NA
    road_move_seconds, road_arrival_seconds = road_width_adjusted_seconds(
        road_distance, row.get("도로폭_속도계수", pd.NA)
    )
    return {
        "도로망추정거리m": road_distance,
        "직선거리m": straight,
        "거리차m": diff,
        "우회율": ratio,
        "도로망보정이동시간초": road_move_seconds,
        "도로망보정예상도착초": road_arrival_seconds,
    }


def route_width_markers(
    gu: str, path: list[tuple[float, float]], road_width_line_df: pd.DataFrame
) -> pd.DataFrame:
    if len(path) < 2:
        return pd.DataFrame()

    segment_lengths = [
        point_distance_m(start, end) for start, end in zip(path, path[1:])
    ]
    total = sum(segment_lengths)
    if total <= 0:
        return pd.DataFrame()

    targets = [total * ratio for ratio in (0.25, 0.5, 0.75)]
    rows = []
    passed = 0.0
    seg_idx = 0
    for target in targets:
        while (
            seg_idx < len(segment_lengths) - 1
            and passed + segment_lengths[seg_idx] < target
        ):
            passed += segment_lengths[seg_idx]
            seg_idx += 1
        start = path[seg_idx]
        end = path[seg_idx + 1]
        length = segment_lengths[seg_idx] or 1
        t = max(0, min(1, (target - passed) / length))
        lat = start[0] + (end[0] - start[0]) * t
        lon = start[1] + (end[1] - start[1]) * t
        width = nearest_road_width(
            pd.Series({"구": gu, "위도": lat, "경도": lon}), road_width_line_df
        )
        if not width:
            continue
        rows.append(
            {
                "위도": lat,
                "경도": lon,
                "도로명": width.get("인접도로명", "-"),
                "도로폭표시": width.get("도로폭표시", "-"),
                "공식도로폭m": width.get("공식도로폭m", pd.NA),
                "매칭거리m": width.get("도로폭매칭거리m", pd.NA),
            }
        )
    return pd.DataFrame(rows)


def dispatch_route_map(
    route_df: pd.DataFrame,
    fire_df: pd.DataFrame,
    road_width_line_df: pd.DataFrame,
    road_route_df: pd.DataFrame,
    selected_row: pd.Series,
) -> go.Figure:
    dong_points = (
        route_df[
            (route_df["구"] == selected_row["구"])
            & (route_df["동"] == selected_row["동"])
        ]
        .dropna(subset=["위도", "경도"])
        .copy()
    )
    center_row = nearest_fire_center(selected_row, fire_df)
    center = pd.DataFrame([center_row]) if center_row is not None else pd.DataFrame()
    center_name = (
        center_row["시설명"]
        if center_row is not None
        else selected_row.get("최근접_안전센터", "-")
    )

    road_width = selected_row.get("공식도로폭m", pd.NA)
    if pd.isna(road_width):
        road_width = 6
    line_width = 4 if pd.isna(road_width) else max(4, min(14, float(road_width) * 0.9))
    road_lines = selected_row.get("road_lines", [])
    road_path: list[tuple[float, float]] = []
    road_distance = None

    fig = go.Figure()
    if not dong_points.empty:
        fig.add_trace(
            go.Scattermapbox(
                lat=dong_points["위도"],
                lon=dong_points["경도"],
                mode="markers",
                name=f"{selected_row['동']} 숙박시설",
                marker=dict(size=7, color="rgba(100,116,139,0.34)"),
                customdata=dong_points[
                    ["업소명", "최근접_안전센터", "최근접_거리m", "공식도로폭m"]
                ],
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "최근접 안전센터: %{customdata[1]}<br>"
                    "거리: %{customdata[2]:,.0f}m<br>"
                    "공식 도로폭: %{customdata[3]:.1f}m"
                    "<extra></extra>"
                ),
            )
        )

    if not center.empty:
        center_row = center.iloc[0]
        if isinstance(road_lines, list) and road_lines:
            for idx, line in enumerate(road_lines):
                fig.add_trace(
                    go.Scattermapbox(
                        lat=[point[0] for point in line],
                        lon=[point[1] for point in line],
                        mode="lines",
                        name="숙소 바로 인접도로" if idx == 0 else "숙소 바로 인접도로",
                        showlegend=idx == 0,
                        line=dict(width=line_width, color="#f59e0b"),
                        hovertemplate=(
                            "<b>숙소 바로 인접도로</b><br>"
                            f"도로명: {selected_row.get('인접도로명', '-')}<br>"
                            f"도로폭: {selected_row.get('도로폭표시', '-')}<br>"
                            f"숙소-도로 매칭거리: {value_or_dash(selected_row.get('도로폭매칭거리m'), 'm', 0)}"
                            "<extra></extra>"
                        ),
                    )
                )
        fig.add_trace(
            go.Scattermapbox(
                lat=[selected_row["위도"], center_row["위도"]],
                lon=[selected_row["경도"], center_row["경도"]],
                mode="lines",
                name="출동 연결선",
                line=dict(width=max(4, line_width - 1), color="#2563eb"),
                hovertemplate=(
                    f"도로폭 반영 선두께<br>"
                    f"인접도로: {selected_row.get('인접도로명', '-')}<br>"
                    f"공식도로폭: {road_width}m"
                    "<extra></extra>"
                ),
            )
        )
        auto_path, auto_distance = shortest_graph_path(
            (selected_row["위도"], selected_row["경도"]),
            (center_row["위도"], center_row["경도"]),
            selected_row["구"],
            road_route_df,
        )
        if auto_path:
            fig.add_trace(
                go.Scattermapbox(
                    lat=[point[0] for point in auto_path],
                    lon=[point[1] for point in auto_path],
                    mode="lines",
                    name="자동 도로망 최단경로",
                    line=dict(width=3, color="rgba(100,116,139,0.55)"),
                    hovertemplate=(
                        "<b>자동 도로망 최단경로</b><br>"
                        f"추정거리: {auto_distance:,.0f}m<br>"
                        "인접도로를 강제하지 않은 비교 경로"
                        "<extra></extra>"
                    ),
                )
            )
        road_path, road_distance, road_anchor = dispatch_arrival_path(
            selected_row, center_row, road_route_df
        )
        if road_path:
            fig.add_trace(
                go.Scattermapbox(
                    lat=[point[0] for point in road_path],
                    lon=[point[1] for point in road_path],
                    mode="lines",
                    name="인접도로 진입 경로",
                    line=dict(width=6, color="#059669"),
                    hovertemplate=(
                        "<b>인접도로 진입 경로</b><br>"
                        f"추정거리: {road_distance:,.0f}m<br>"
                        f"진입도로: {selected_row.get('인접도로명', '-')}<br>"
                        f"직선거리: {selected_row.get('안전센터_유클리드m', selected_row.get('최근접_거리m', 0)):,.0f}m"
                        "<extra></extra>"
                    ),
                )
            )
        if road_anchor:
            fig.add_trace(
                go.Scattermapbox(
                    lat=[road_anchor[0]],
                    lon=[road_anchor[1]],
                    mode="markers+text",
                    name="인접도로 진입점",
                    text=["진입점"],
                    textposition="bottom center",
                    marker=dict(size=13, color="#0f766e", opacity=0.96),
                    hovertemplate=(
                        "<b>인접도로 진입점</b><br>"
                        f"도로명: {selected_row.get('인접도로명', '-')}<br>"
                        "도로망 경로가 이 지점을 통해 숙소로 진입합니다."
                        "<extra></extra>"
                    ),
                )
            )
        width_markers = (
            route_width_markers(selected_row["구"], road_path, road_width_line_df)
            if road_path
            else pd.DataFrame()
        )
        if not width_markers.empty:
            fig.add_trace(
                go.Scattermapbox(
                    lat=width_markers["위도"],
                    lon=width_markers["경도"],
                    mode="markers+text",
                    name="경로 도로폭",
                    text=width_markers["도로폭표시"],
                    textposition="top center",
                    marker=dict(size=12, color="#f59e0b", opacity=0.94),
                    customdata=width_markers[["도로명", "공식도로폭m", "매칭거리m"]],
                    hovertemplate=(
                        "<b>경로 주변 도로폭</b><br>"
                        "도로명: %{customdata[0]}<br>"
                        "도로폭: %{text}<br>"
                        "공식 평균: %{customdata[1]:.1f}m<br>"
                        "경로-도로폭선 매칭거리: %{customdata[2]:.0f}m"
                        "<extra></extra>"
                    ),
                )
            )
        fig.add_trace(
            go.Scattermapbox(
                lat=[selected_row["위도"]],
                lon=[selected_row["경도"]],
                mode="markers+text",
                name="숙소 인접 도로폭",
                text=[selected_row.get("도로폭표시", f"{road_width}m")],
                textposition="top center",
                marker=dict(size=10, color="#f59e0b", opacity=0.92),
                hovertemplate=(
                    "<b>숙소 인접 도로폭</b><br>"
                    f"도로명: {selected_row.get('인접도로명', '-')}<br>"
                    f"표시도로폭: {selected_row.get('도로폭표시', '-')}<br>"
                    f"공식 평균: {road_width}m"
                    "<extra></extra>"
                ),
            )
        )
        fig.add_trace(
            go.Scattermapbox(
                lat=[center_row["위도"]],
                lon=[center_row["경도"]],
                mode="markers+text",
                name="인접 안전센터",
                text=[center_name],
                textposition="top right",
                marker=dict(
                    size=17, color="#059669", symbol="fire-station", opacity=0.95
                ),
                hovertemplate="<b>%{text}</b><br>인접 안전센터<extra></extra>",
            )
        )

    fig.add_trace(
        go.Scattermapbox(
            lat=[selected_row["위도"]],
            lon=[selected_row["경도"]],
            mode="markers+text",
            name="선택 숙박시설",
            text=[selected_row["업소명"]],
            textposition="bottom right",
            marker=dict(size=18, color="#e11d48", opacity=0.96),
            customdata=[
                [
                    selected_row.get("최종위험점수_new", pd.NA),
                    selected_row.get("도로폭_보정예상도착초", pd.NA),
                    selected_row.get("최근접_거리m", pd.NA),
                ]
            ],
            hovertemplate=(
                "<b>%{text}</b><br>"
                "위험도: %{customdata[0]:.2f}<br>"
                "도로폭 보정 예상도착: %{customdata[1]:.0f}초<br>"
                "안전센터 거리: %{customdata[2]:,.0f}m"
                "<extra></extra>"
            ),
        )
    )

    lats = [selected_row["위도"]]
    lons = [selected_row["경도"]]
    if road_path:
        lats.extend([point[0] for point in road_path])
        lons.extend([point[1] for point in road_path])
    if isinstance(road_lines, list):
        for line in road_lines:
            lats.extend([point[0] for point in line])
            lons.extend([point[1] for point in line])
    if not center.empty:
        lats.append(center.iloc[0]["위도"])
        lons.append(center.iloc[0]["경도"])
    if not dong_points.empty:
        lats.extend(dong_points["위도"].tolist())
        lons.extend(dong_points["경도"].tolist())

    spread = max(max(lats) - min(lats), max(lons) - min(lons))
    zoom = (
        15 if spread < 0.01 else 14 if spread < 0.025 else 13 if spread < 0.05 else 12
    )
    fig.update_layout(
        title=f"{selected_row['동']} 출동경로 및 도로폭",
        mapbox=dict(
            style="carto-positron",
            center=dict(lat=sum(lats) / len(lats), lon=sum(lons) / len(lons)),
            zoom=zoom,
        ),
        height=460,
        margin=dict(l=0, r=0, t=48, b=0),
        paper_bgcolor="white",
        font=dict(color=COLORS["ink"]),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=0.01,
            x=0.01,
            bgcolor="rgba(255,255,255,0.86)",
            bordercolor="#e8eef7",
            borderwidth=1,
        ),
    )
    return fig


final_df, score_df, model_df, compare_df = load_0430()
gwr_local_df = load_gwr_local(final_df)
mgwr_cluster_local_df = load_mgwr_cluster_local()
dong_boundary_geo = load_dong_boundaries()
gu_boundary_geo = build_gu_boundaries(dong_boundary_geo)
grid_250_geo = load_seoul_grid(250)
grid_500_geo = load_seoul_grid(500)
license_df = load_license_sources()
route_df, fire_facility_df, road_width_df, road_line_df, road_route_df = (
    load_route_sources()
)
gu_list = sorted(final_df["구"].dropna().unique())

with st.sidebar:
    st.markdown(
        '<div class="sidebar-title">서울시 관광 지역 내 숙박 시설 화재 위험도 분석</div>',
        unsafe_allow_html=True,
    )
    st.header("필터")
    default_gu = gu_list.index("마포구") if "마포구" in gu_list else 0
    selected_gu = st.selectbox("구 선택", gu_list, index=default_gu)

gu_df = final_df[final_df["구"] == selected_gu].copy()
high_count = int((gu_df["cluster_label"] == "고위험군").sum())
new_2025 = int((gu_df["승인연도"] == 2025).sum())
avg_risk = gu_df["최종위험점수_new"].mean()
max_risk = gu_df["최종위험점수_new"].max()

selected_view = st.radio(
    "화면 선택",
    ["개요", "인허가", "위험지도", "클러스터", "공간모형", "GWR/MGWR"],
    horizontal=True,
    label_visibility="collapsed",
)

if selected_view == "개요":
    st.markdown(
        '<div class="section-title">선택 구 위험 프로파일</div>', unsafe_allow_html=True
    )
    st.markdown(
        '<div class="soft-note">0430 최종테이블의 선택 구 평균을 서울 10구 전체 평균과 비교합니다. 100보다 크면 전체 평균보다 높은 지표입니다.</div>',
        unsafe_allow_html=True,
    )
    left, right = st.columns([1.1, 0.9])
    with left:
        st.plotly_chart(risk_factor_chart(final_df, selected_gu), width="stretch")
        st.dataframe(
            risk_factor_table(final_df, selected_gu), hide_index=True, width="stretch"
        )
    with right:
        st.markdown(
            '<div class="section-title">선택 구 위험시설 TOP 10</div>',
            unsafe_allow_html=True,
        )
        rank_cols = [
            "구",
            "동",
            "숙소명",
            "업종",
            "cluster_label",
            "최종위험점수_new",
            "최근접_소화용수_거리등급",
        ]
        st.dataframe(
            gu_df.sort_values("최종위험점수_new", ascending=False)[rank_cols].head(10),
            hide_index=True,
            width="stretch",
        )

elif selected_view == "인허가":
    st.markdown(
        '<div class="section-title">신규 인허가 흐름</div>', unsafe_allow_html=True
    )
    top_left, top_mid, top_right = st.columns([1.25, 0.9, 0.95])
    with top_left:
        st.plotly_chart(
            license_cumulative_chart(license_df, selected_gu), width="stretch"
        )
    with top_mid:
        st.plotly_chart(
            license_dong_share_chart(license_df, selected_gu), width="stretch"
        )
    with top_right:
        st.plotly_chart(
            license_category_year_chart(license_df, selected_gu), width="stretch"
        )
    st.plotly_chart(license_heatmap_chart(license_df, selected_gu), width="stretch")

elif selected_view == "위험지도":
    st.markdown(
        '<div class="section-title">법정동 중심 위험 지도</div>', unsafe_allow_html=True
    )
    st.markdown(
        '<div class="soft-note">선택 구의 법정동 경계를 기준으로 구역을 나누고, 각 법정동에서 가장 많이 나타나는 위험군 색으로 채웁니다. 점은 개별 숙박시설 위치이며, 마우스를 올리면 법정동별 위험군 구성과 평균 위험도를 확인할 수 있습니다.</div>',
        unsafe_allow_html=True,
    )
    selected_clusters = st.multiselect(
        "지도 위험군",
        ["저위험군", "중위험군", "고위험군"],
        default=["저위험군", "중위험군", "고위험군"],
    )
    st.plotly_chart(
        dong_focus_map(final_df, selected_gu, selected_clusters), width="stretch"
    )

elif selected_view == "클러스터":
    st.markdown(
        '<div class="section-title">클러스터별 숙박시설 개수 상위 지역</div>',
        unsafe_allow_html=True,
    )
    top_area = cluster_top_areas(final_df)
    left, right = st.columns([0.95, 1.35])
    with left:
        st.dataframe(top_area, hide_index=True, width="stretch")
    with right:
        bar_df = top_area.copy()
        bar_df["지역"] = bar_df["구"] + " " + bar_df["동"]
        fig = px.bar(
            bar_df,
            x="숙박시설 수",
            y="지역",
            color="cluster_label",
            orientation="h",
            facet_col="cluster_label",
            color_discrete_map=CLUSTER_COLORS,
        )
        fig.update_layout(
            showlegend=False,
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=0, r=0, t=36, b=0),
            xaxis_title="숙박시설 수",
            yaxis_title="",
        )
        st.plotly_chart(fig, width="stretch")

elif selected_view == "GWR/MGWR":
    st.markdown(
        '<div class="section-title">저위험군/고위험군 MGWR 변수별 공간 패턴</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="soft-note">서울시 전체 법정동 경계를 배경으로 깔고, 그 위에 격자를 올려 선택 변수의 MGWR 지역 기여도를 색칠합니다. 빨강은 평균보다 위험도를 더 올리는 방향, 파랑은 낮추는 방향이며, 선택 변수의 MGWR BW는 그 효과가 국소적으로 작동하는지 광역적으로 작동하는지 설명합니다.</div>',
        unsafe_allow_html=True,
    )
    local_source_df = (
        mgwr_cluster_local_df if not mgwr_cluster_local_df.empty else gwr_local_df
    )
    if local_source_df.empty:
        st.warning("지역 계수 파일을 찾지 못했거나 좌표와 매칭하지 못했습니다.")
    else:
        control_l, control_r = st.columns([0.36, 0.64])
        with control_l:
            risk_group = st.selectbox("위험군", ["저위험군", "고위험군"], index=0)
            scoped_local_df = (
                local_source_df[local_source_df["cluster_label"] == risk_group].copy()
                if "cluster_label" in local_source_df.columns
                else local_source_df.copy()
            )
            variable_options = [
                col.replace("coef_", "")
                for col in scoped_local_df.columns
                if col.startswith("coef_") and col != "coef_intercept"
            ]
            selected_variable = (
                st.selectbox("영향도 변수", variable_options, index=0)
                if variable_options
                else None
            )
            display_metric = st.radio(
                "지도 표시값",
                ["지역 기여도", "MGWR 계수"],
                index=0,
                horizontal=True,
                help="계수가 거의 일정하면 MGWR 계수 지도는 같은 색으로 보입니다. 지역 기여도는 MGWR 계수에 해당 지점의 변수값(z-score)을 곱해 공간적 차이를 보여줍니다.",
            )
            map_unit_label = st.radio(
                "집계 단위",
                ["법정동", "격자"],
                index=0,
                horizontal=True,
                help="법정동은 행정구역별 해석에 좋고, 격자는 더 세밀한 공간 패턴 확인에 좋습니다.",
            )
            grid_size_label = st.radio(
                "격자 크기",
                ["500m", "250m"],
                index=0,
                horizontal=True,
                help="250m는 더 촘촘하지만 브라우저 렌더링이 무거울 수 있습니다.",
                disabled=(map_unit_label == "법정동"),
            )
            color_scale_label = st.radio(
                "색상 스케일",
                ["평균 대비", "0 기준"],
                index=0,
                horizontal=True,
                help="평균 대비는 같은 위험군 안에서 상대적으로 높은/낮은 지역을 보여줍니다. 0 기준은 계수나 기여도의 양/음 방향을 그대로 보여줍니다.",
            )
            bw_table = parse_bw_table(model_df, risk_group)
            selected_bw = None
            if selected_variable and not bw_table.empty:
                selected_rows = bw_table[bw_table["변수"] == selected_variable]
                if not selected_rows.empty:
                    selected_bw = selected_rows.iloc[0].to_dict()
                    st.metric(
                        "선택 변수 MGWR BW",
                        f"{selected_bw['BW']:.0f} NN",
                        f"{selected_bw['공간범위']} - {selected_bw['해석']}",
                    )
                st.markdown("**MGWR BW 해석**")
                st.dataframe(bw_table, hide_index=True, width="stretch")
            if not mgwr_cluster_local_df.empty:
                st.caption(
                    "지도 색은 기본적으로 MGWR 계수 × 지역 변수값(z-score)의 공간단위 평균입니다. BW는 법정동/격자별 값이 아니라 선택 위험군/변수의 MGWR 대역폭입니다."
                )
        with control_r:
            model_view = model_df.copy()
            if not model_view.empty:
                display_cols = [
                    col
                    for col in [
                        "위험군",
                        "대표모형",
                        "n",
                        "R2",
                        "Adj_R2",
                        "AICc",
                        "Residual_Moran_I",
                        "Moran_p",
                        "BW",
                    ]
                    if col in model_view.columns
                ]
                st.dataframe(
                    model_view[model_view["위험군"].isin(["저위험군", "고위험군"])][
                        display_cols
                    ],
                    hide_index=True,
                    width="stretch",
                )
        if scoped_local_df.empty or not variable_options:
            st.warning(f"{risk_group}에 표시할 지역 계수 데이터가 없습니다.")
        else:
            metric_key = (
                "coefficient" if display_metric == "MGWR 계수" else "contribution"
            )
            scale_key = "absolute" if color_scale_label == "0 기준" else "relative"
            if map_unit_label == "법정동":
                st.plotly_chart(
                    gwr_variable_seoul_map(
                        scoped_local_df,
                        selected_variable,
                        dong_boundary_geo,
                        gu_boundary_geo,
                        selected_bw,
                        risk_group=risk_group,
                        model_label="MGWR",
                        metric=metric_key,
                        color_scale_mode=scale_key,
                    ),
                    width="stretch",
                )
            else:
                selected_grid_geo = (
                    grid_250_geo if grid_size_label == "250m" else grid_500_geo
                )
                st.plotly_chart(
                    mgwr_grid_seoul_map(
                        scoped_local_df,
                        selected_variable,
                        dong_boundary_geo,
                        selected_grid_geo,
                        gu_boundary_geo,
                        selected_bw,
                        risk_group=risk_group,
                        metric=metric_key,
                        color_scale_mode=scale_key,
                    ),
                    width="stretch",
                )

elif selected_view == "공간모형":
    st.markdown(
        '<div class="section-title">구별 상위 대표 숙박시설 출동경로</div>',
        unsafe_allow_html=True,
    )
    control_l, control_r = st.columns([0.32, 0.68])
    with control_l:
        route_cluster = st.selectbox(
            "군집 선택",
            ["고위험군", "중위험군", "저위험군"],
            key="route_cluster",
        )

    representatives = representative_route_facilities(
        final_df, route_df, road_width_df, road_line_df, selected_gu, route_cluster
    )
    if representatives.empty:
        st.info(
            "선택한 구와 군집에서 상위 2개 동 대표 숙박시설 출동경로 데이터가 없습니다."
        )
    else:
        with control_r:
            selected_label = st.selectbox(
                "상위 동 대표 숙박시설 선택",
                representatives["선택라벨"].tolist(),
                key=f"route_facility_{selected_gu}_{route_cluster}",
            )
        selected_route = representatives[
            representatives["선택라벨"] == selected_label
        ].iloc[0]
        route_compare = selected_route_comparison(
            selected_route, fire_facility_df, road_route_df
        )

        map_col, info_col = st.columns([1.2, 0.8])
        with map_col:
            st.plotly_chart(
                dispatch_route_map(
                    route_df,
                    fire_facility_df,
                    road_line_df,
                    road_route_df,
                    selected_route,
                ),
                width="stretch",
            )
            compare_df_view = pd.DataFrame(
                [
                    {
                        "구분": "유클리드 직선거리",
                        "거리(m)": route_compare["직선거리m"],
                    },
                    {
                        "구분": "공식 도로망 추정거리",
                        "거리(m)": route_compare["도로망추정거리m"],
                    },
                    {
                        "구분": "도로망-유클리드 차이",
                        "거리(m)": route_compare["거리차m"],
                    },
                ]
            )
            st.dataframe(
                compare_df_view.assign(
                    **{
                        "거리(m)": compare_df_view["거리(m)"].map(
                            lambda v: value_or_dash(v, "", 0)
                        )
                    }
                ),
                hide_index=True,
                width="stretch",
            )
        with info_col:
            metric_cols = st.columns(2)
            metric_cols[0].metric(
                "최종위험점수", f"{selected_route.get('최종위험점수_new', 0):.2f}점"
            )
            metric_cols[1].metric(
                "직선거리", value_or_dash(route_compare["직선거리m"], "m", 0)
            )
            metric_cols[0].metric(
                "도로폭 보정 예상도착",
                route_seconds_only_label(selected_route.get("도로폭_보정예상도착초")),
            )
            metric_cols[1].metric(
                "도로망 추정거리",
                value_or_dash(route_compare["도로망추정거리m"], "m", 0),
            )
            metric_cols[0].metric(
                "공식 도로폭", value_or_dash(selected_route.get("공식도로폭m"), "m", 1)
            )
            metric_cols[1].metric(
                "우회율",
                "-"
                if pd.isna(route_compare["우회율"])
                else f"{route_compare['우회율']:.2f}배",
            )

            route_info = pd.DataFrame(
                [
                    {"항목": "법정동", "값": selected_route.get("동", "-")},
                    {
                        "항목": "인접 안전센터",
                        "값": selected_route.get("최근접_안전센터", "-"),
                    },
                    {
                        "항목": "담당 안전센터",
                        "값": selected_route.get("담당_안전센터", "-"),
                    },
                    {"항목": "인접도로명", "값": selected_route.get("인접도로명", "-")},
                    {
                        "항목": "도로폭 구간",
                        "값": selected_route.get("도로폭표시", "-"),
                    },
                    {
                        "항목": "공식 도로폭 평균",
                        "값": value_or_dash(selected_route.get("공식도로폭m"), "m", 2),
                    },
                    {
                        "항목": "공식 도로폭 구간수",
                        "값": value_or_dash(
                            selected_route.get("도로폭공식구간수"), "개", 0
                        ),
                    },
                    {
                        "항목": "숙소-도로 매칭거리",
                        "값": value_or_dash(
                            selected_route.get("도로폭매칭거리m"), "m", 0
                        ),
                    },
                    {
                        "항목": "직선거리(유클리드)",
                        "값": value_or_dash(route_compare["직선거리m"], "m", 0),
                    },
                    {
                        "항목": "도로망 추정거리",
                        "값": value_or_dash(route_compare["도로망추정거리m"], "m", 0),
                    },
                    {
                        "항목": "도로망/직선 거리비",
                        "값": "-"
                        if pd.isna(route_compare["우회율"])
                        else f"{route_compare['우회율']:.2f}배",
                    },
                    {
                        "항목": "도로폭 보정 이동시간",
                        "값": route_seconds_only_label(
                            selected_route.get("도로폭_보정이동시간초")
                        ),
                    },
                    {
                        "항목": "도로폭 보정 예상도착",
                        "값": route_seconds_only_label(
                            selected_route.get("도로폭_보정예상도착초")
                        ),
                    },
                    {
                        "항목": "도로망 기준 보정 이동시간",
                        "값": route_seconds_only_label(
                            route_compare["도로망보정이동시간초"]
                        ),
                    },
                    {
                        "항목": "도로망 기준 보정 예상도착",
                        "값": route_seconds_only_label(
                            route_compare["도로망보정예상도착초"]
                        ),
                    },
                    {"항목": "도로폭 출처", "값": "road_width_10gu 공식 도로폭 선형"},
                ]
            )
            st.dataframe(route_info, hide_index=True, width="stretch")

        preview_compare = [
            selected_route_comparison(row, fire_facility_df, road_route_df)[
                "도로망보정예상도착초"
            ]
            for _, row in representatives.iterrows()
        ]
        preview_cols = [
            "동",
            "숙소명",
            "최근접_안전센터",
            "최근접_거리m",
            "도로폭표시",
            "공식도로폭m",
            "도로폭_보정예상도착초",
            "최종위험점수_new",
        ]
        preview = representatives[preview_cols].copy()
        preview["도로망예상도착"] = [
            route_seconds_only_label(value) for value in preview_compare
        ]
        preview.columns = [
            "동",
            "숙소명",
            "인접 안전센터",
            "거리(m)",
            "도로폭구간",
            "공식도로폭(m)",
            "보정예상도착(초)",
            "최종위험점수",
            "도로망예상도착",
        ]
        st.dataframe(preview, hide_index=True, width="stretch")
