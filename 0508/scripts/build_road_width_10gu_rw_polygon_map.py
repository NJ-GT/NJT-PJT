# -*- coding: utf-8 -*-
"""
서울 10개구 도로폭을 행안부 실폭도로(TL_SPRD_RW) 폴리곤과 매칭해 면(폴리곤) 단위 인터랙티브 지도 생성.

기존 line_map / official_complete_map 과 차이:
    - 도로 표현 단위가 LineString 이 아닌 Polygon (실제 도로 폭/모양 그대로)
    - TL_SPRD_RW 폴리곤을 도로 중심선(TL_SPRD_MANAGE)과 공간 매칭:
        1) intersects 으로 직접 교차 (가장 신뢰)
        2) 매칭 안된 폴리곤은 NEAREST_MAX_DISTANCE_M(20m) 내 최근접 매칭
    - 폭 출처 4단계: API / 공식폭보완 / 공식만 / 실폭만 (실폭 폴리곤만 있고 중심선 매칭 실패)

입력:
    - road_width_10gu/data/seoul_road_width_viRoutDt_10*.csv             (API 도로폭)
    - road_width_10gu/data/official_road_shape_202603_seoul/11000/TL_SPRD_MANAGE.shp (도로 중심선)
    - road_width_10gu/data/official_road_shape_202603_seoul/11000/TL_SPRD_RW.shp     (실폭 폴리곤)

출력:
    - road_width_10gu/data/seoul_road_width_10gu_rw_polygon_map.html
    - road_width_10gu/data/seoul_road_width_10gu_rw_polygons.geojson
    - road_width_10gu/data/seoul_road_width_10gu_rw_polygon_match_summary.csv
    - road_width_10gu/data/seoul_road_width_10gu_rw_polygon_unmatched.csv
    - road_width_10gu/data/seoul_road_width_10gu_rw_polygon_api_only.csv
"""
from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
from shapely.geometry import MultiPolygon, mapping


# 경로
BASE_DIR = Path(__file__).resolve().parents[1] / "road_width_10gu"
DATA_DIR = BASE_DIR / "data"
INPUT_CSV = next(DATA_DIR.glob("seoul_road_width_viRoutDt_10*.csv"))
MANAGE_SHP = (
    DATA_DIR / "official_road_shape_202603_seoul" / "11000" / "TL_SPRD_MANAGE.shp"
)
RW_SHP = DATA_DIR / "official_road_shape_202603_seoul" / "11000" / "TL_SPRD_RW.shp"

OUTPUT_HTML = DATA_DIR / "seoul_road_width_10gu_rw_polygon_map.html"
OUTPUT_GEOJSON = DATA_DIR / "seoul_road_width_10gu_rw_polygons.geojson"
OUTPUT_SUMMARY_CSV = DATA_DIR / "seoul_road_width_10gu_rw_polygon_match_summary.csv"
OUTPUT_UNMATCHED_RW_CSV = DATA_DIR / "seoul_road_width_10gu_rw_polygon_unmatched.csv"
OUTPUT_API_ONLY_CSV = DATA_DIR / "seoul_road_width_10gu_rw_polygon_api_only.csv"

# 행안부 SHP 좌표계 (UTM-K)
SRC_CRS = "EPSG:5179"
# 직접 교차 매칭 실패 시 최근접 매칭 한계 거리 (m)
NEAREST_MAX_DISTANCE_M = 20.0
# GeoJSON 산출 시 폴리곤 단순화 tolerance (m) — 파일 크기 절감
SIMPLIFY_TOLERANCE_M = 0.8

# 시군구코드 5자리 → 자치구명 (서울 10구만)
SIG_TO_GU = {
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

# 자치구/도로폭/폭출처/매칭 정렬 — 시각화 일관성
GU_ORDER = [
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
WIDTH_ORDER = [
    "6m미만",
    "폭6-8m",
    "폭8-10m",
    "폭10-12m",
    "폭12-15m",
    "폭15-20m",
    "폭20-25m",
    "폭25-30m",
    "폭30-35m",
    "폭35-40m",
    "폭40-50m",
    "폭50-70m",
    "불가",
]
# 폭 출처 정렬 — 신뢰도 내림차순
SOURCE_ORDER = ["API", "공식폭보완", "공식만", "실폭만"]
# 매칭 방식 정렬 — 신뢰도 내림차순
MATCH_ORDER = ["교차", "근접", "미매칭"]
WIDTH_COLORS = {
    "6m미만": "#2f855a",
    "폭6-8m": "#63a15f",
    "폭8-10m": "#d6a419",
    "폭10-12m": "#e27b36",
    "폭12-15m": "#e05d3d",
    "폭15-20m": "#d64545",
    "폭20-25m": "#b83266",
    "폭25-30m": "#7f3c8d",
    "폭30-35m": "#2f6fb5",
    "폭35-40m": "#1e4e8c",
    "폭40-50m": "#2d3748",
    "폭50-70m": "#111827",
    "불가": "#7a7f87",
}


def normalize_name(value: object) -> str:
    """매칭 키 정규화 — 모든 공백 제거."""
    return "".join(str(value or "").strip().split())


def official_width_bucket(value: float | None) -> str:
    """공식 ROAD_BT(미터) → API 동일 카테고리 라벨 변환."""
    if value is None or math.isnan(value):
        return "불가"
    if value < 6:
        return "6m미만"
    if value < 8:
        return "폭6-8m"
    if value < 10:
        return "폭8-10m"
    if value < 12:
        return "폭10-12m"
    if value < 15:
        return "폭12-15m"
    if value < 20:
        return "폭15-20m"
    if value < 25:
        return "폭20-25m"
    if value < 30:
        return "폭25-30m"
    if value < 35:
        return "폭30-35m"
    if value < 40:
        return "폭35-40m"
    if value < 50:
        return "폭40-50m"
    if value < 70:
        return "폭50-70m"
    return "폭50-70m"


def as_float(value: object) -> float | None:
    """안전 float 변환 — NaN/실패 시 None."""
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return None
    return float(number)


def distance_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """위경도 두 점 사이 거리(m) — cosine 보정 평면 근사."""
    mean_lat = math.radians((lat1 + lat2) / 2)
    dx = (lng2 - lng1) * 111_320 * math.cos(mean_lat)
    dy = (lat2 - lat1) * 110_574
    return math.hypot(dx, dy)


def polygon_parts(geom: Any) -> list[Any]:
    """Shapely 지오메트리에서 모든 폴리곤 파트만 평탄화 (Polygon/MultiPolygon/GeometryCollection 처리)."""
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == "Polygon":
        return [geom]
    if geom.geom_type == "MultiPolygon":
        return list(geom.geoms)
    if geom.geom_type == "GeometryCollection":
        # 컬렉션은 재귀로 풀기
        parts: list[Any] = []
        for sub_geom in geom.geoms:
            parts.extend(polygon_parts(sub_geom))
        return parts
    return []


def extract_polygonal(geom: Any) -> Any | None:
    """지오메트리를 Polygon/MultiPolygon 으로 정규화 — 라인이 섞여있으면 폴리곤만 추출."""
    parts = polygon_parts(geom)
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return MultiPolygon(parts)


def geometry_to_lines(geom: Any) -> list[list[list[float]]]:
    """Shapely 라인 → [[x, y], ...] 리스트 (GeoJSON 좌표 순서) — 끝점 추출용."""
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == "LineString":
        return [[[float(x), float(y)] for x, y in geom.coords]]
    if geom.geom_type == "MultiLineString":
        return [[[float(x), float(y)] for x, y in part.coords] for part in geom.geoms]
    if geom.geom_type == "GeometryCollection":
        lines: list[list[list[float]]] = []
        for sub_geom in geom.geoms:
            lines.extend(geometry_to_lines(sub_geom))
        return lines
    return []


def choose_endpoints(
    lines: list[list[list[float]]],
) -> tuple[list[float], list[float]] | tuple[None, None]:
    """라인의 양 끝점 중 가장 멀리 떨어진 두 점 반환 (없으면 None, None)."""
    endpoints = []
    for line in lines:
        if len(line) >= 2:
            endpoints.extend([line[0], line[-1]])
    if len(endpoints) < 2:
        return None, None

    best_pair = (endpoints[0], endpoints[-1])
    best_dist = -1.0
    for i, start in enumerate(endpoints):
        for end in endpoints[i + 1 :]:
            dist = distance_m(start[1], start[0], end[1], end[0])
            if dist > best_dist:
                best_dist = dist
                best_pair = (start, end)
    return best_pair


def round_nested(value: Any, digits: int = 6) -> Any:
    """중첩 리스트 안 모든 숫자를 digits 자리로 반올림 — GeoJSON 좌표 정리용."""
    if isinstance(value, (float, int)):
        return round(float(value), digits)
    if isinstance(value, (list, tuple)):
        return [round_nested(item, digits) for item in value]
    return value


def clean_json_value(value: Any) -> Any:
    """JSON 직렬화 안전 변환 — NaN/None/numpy 스칼라 등을 표준 타입으로."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [clean_json_value(item) for item in value]
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        # numpy 스칼라 → 파이썬 기본 타입
        value = value.item()
    if isinstance(value, float):
        if math.isnan(value):
            return None
        return round(value, 6)
    if isinstance(value, (int, str, bool)):
        return value
    return str(value)


def load_input() -> pd.DataFrame:
    """API 도로폭 CSV 로드 + 첫 7개 컬럼 표준명 매핑 + (구|도로명) 키 부착."""
    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig", dtype=str).fillna("")
    seq_col, gu_col, road_col, kind_col, function_col, scale_col, width_col = (
        df.columns[:7]
    )
    df = df.rename(
        columns={
            seq_col: "순번",
            gu_col: "구",
            road_col: "도로명",
            kind_col: "도로구분",
            function_col: "도로기능",
            scale_col: "도로규모",
            width_col: "API도로폭",
        }
    )
    df["_key"] = df["구"].map(normalize_name) + "|" + df["도로명"].map(normalize_name)
    return df.drop_duplicates("_key", keep="first")


def load_manage() -> gpd.GeoDataFrame:
    """도로 중심선 SHP 로드 — 자치구 매칭 + 키 부착 + 폭/길이 숫자형 변환."""
    gdf = gpd.read_file(MANAGE_SHP, engine="pyogrio", encoding="cp949")
    gdf = gdf.set_crs(SRC_CRS, allow_override=True)
    gdf["도로구"] = gdf["SIG_CD"].astype(str).map(SIG_TO_GU).fillna("")
    gdf = gdf[gdf["도로구"] != ""].copy().reset_index(drop=True)
    gdf["mg_idx"] = range(len(gdf))
    gdf["_line_key"] = (
        gdf["도로구"].map(normalize_name) + "|" + gdf["RN"].map(normalize_name)
    )
    gdf["ROAD_BT_NUM"] = pd.to_numeric(gdf["ROAD_BT"], errors="coerce")
    gdf["ROAD_LT_NUM"] = pd.to_numeric(gdf["ROAD_LT"], errors="coerce")
    return gdf


def load_rw() -> gpd.GeoDataFrame:
    """실폭도로 폴리곤 SHP 로드 — 무효 지오메트리 보정 + 폴리곤만 추출."""
    # pyogrio 의 RuntimeWarning 무시 (일부 도형이 자기교차일 때 경고 발생)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        gdf = gpd.read_file(RW_SHP, engine="pyogrio", encoding="cp949")
    gdf = gdf.set_crs(SRC_CRS, allow_override=True)
    gdf["구"] = gdf["SIG_CD"].astype(str).map(SIG_TO_GU).fillna("")
    gdf = gdf[gdf["구"] != ""].copy().reset_index(drop=True)
    # make_valid → Polygon/MultiPolygon 만 추출 (라인이 섞이면 제거)
    gdf["geometry"] = gdf.geometry.make_valid().map(extract_polygonal)
    gdf = (
        gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy().reset_index(drop=True)
    )
    gdf["rw_idx"] = range(len(gdf))
    return gdf


def build_match_table(rw: gpd.GeoDataFrame, manage: gpd.GeoDataFrame) -> pd.DataFrame:
    """실폭 폴리곤 ↔ 도로 중심선 매칭 — 1) intersects 직접 교차, 2) 20m 내 최근접."""
    rw_key = rw[["rw_idx", "geometry"]]
    manage_key = manage[["mg_idx", "geometry"]]

    # 1단계: 직접 교차 매칭 — sjoin(intersects)
    direct = gpd.sjoin(
        rw_key, manage_key, predicate="intersects", how="inner"
    ).reset_index(drop=True)
    # 한 폴리곤이 여러 라인과 교차할 수 있어 — 교차 길이 큰 라인을 채택
    line_lookup = manage.set_index("mg_idx").geometry
    line_geoms = gpd.GeoSeries(
        line_lookup.reindex(direct["mg_idx"]).to_numpy(), crs=SRC_CRS
    )
    poly_geoms = gpd.GeoSeries(direct.geometry.to_numpy(), crs=SRC_CRS)
    direct["_intersect_len"] = poly_geoms.intersection(line_geoms).length
    direct["_distance_m"] = 0.0
    direct["매칭방식"] = "교차"
    # 폴리곤 1개당 1개의 best 라인 (교차 길이 가장 긴 것)
    direct_best = direct.sort_values(
        ["rw_idx", "_intersect_len"], ascending=[True, False]
    ).drop_duplicates("rw_idx", keep="first")[
        ["rw_idx", "mg_idx", "_intersect_len", "_distance_m", "매칭방식"]
    ]

    # 2단계: 직접 교차에 실패한 폴리곤만 — sjoin_nearest 로 최근접 라인
    direct_ids = set(direct_best["rw_idx"])
    unmatched = rw[~rw["rw_idx"].isin(direct_ids)].copy()
    nearest = gpd.sjoin_nearest(
        unmatched[["rw_idx", "geometry"]],
        manage_key,
        how="left",
        max_distance=NEAREST_MAX_DISTANCE_M,
        distance_col="_distance_m",
    ).reset_index(drop=True)
    # 라인 매칭이 된(거리 산출된) 행만
    nearest = nearest[nearest["mg_idx"].notna()].copy()
    nearest["mg_idx"] = nearest["mg_idx"].astype(int)
    nearest["_intersect_len"] = 0.0
    nearest["매칭방식"] = "근접"
    # 가장 가까운 라인 1개만 채택
    nearest_best = nearest.sort_values(
        ["rw_idx", "_distance_m"], ascending=[True, True]
    ).drop_duplicates("rw_idx", keep="first")[
        ["rw_idx", "mg_idx", "_intersect_len", "_distance_m", "매칭방식"]
    ]

    return pd.concat([direct_best, nearest_best], ignore_index=True)


def build_endpoint_lookup(manage: gpd.GeoDataFrame) -> dict[int, dict[str, Any]]:
    """중심선 mg_idx → 시작/끝 좌표(WGS84) 사전 — 클릭 시 끝점 강조용."""
    manage_wgs = manage[["mg_idx", "geometry"]].to_crs("EPSG:4326")
    lookup: dict[int, dict[str, Any]] = {}
    for row in manage_wgs.itertuples(index=False):
        lines = geometry_to_lines(row.geometry)
        start, end = choose_endpoints(lines)
        if start is None or end is None:
            lookup[int(row.mg_idx)] = {"start": None, "end": None}
            continue
        # Leaflet 은 [lat, lng] 순서 — start[1]=lat, start[0]=lng
        lookup[int(row.mg_idx)] = {
            "start": [round(start[1], 7), round(start[0], 7)],
            "end": [round(end[1], 7), round(end[0], 7)],
        }
    return lookup


def enrich_polygons(
    rw: gpd.GeoDataFrame,
    manage: gpd.GeoDataFrame,
    input_df: pd.DataFrame,
    match_table: pd.DataFrame,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame, pd.DataFrame]:
    """폴리곤에 매칭 + 메타 결합해 GeoJSON 직전 형태로 enrich."""
    meta_by_key = {row["_key"]: row for _, row in input_df.iterrows()}
    endpoint_lookup = build_endpoint_lookup(manage)

    # 매칭으로 mg_idx 가 부착된 폴리곤에 라인 메타(폭/길이/도로구) 결합
    line_attrs = manage[
        [
            "mg_idx",
            "도로구",
            "RN",
            "RN_CD",
            "RDS_MAN_NO",
            "ROAD_BT",
            "ROAD_LT",
            "ROAD_BT_NUM",
            "ROAD_LT_NUM",
            "_line_key",
        ]
    ].copy()
    gdf = rw.merge(match_table, on="rw_idx", how="left").merge(
        line_attrs, on="mg_idx", how="left"
    )

    rows: list[dict[str, Any]] = []
    for _, row in gdf.iterrows():
        has_line = pd.notna(row.get("mg_idx"))
        # 라인 매칭 안 됐으면 도로명을 '미매칭 실폭도로'
        road_name = (
            str(row["RN"]).strip()
            if has_line and pd.notna(row["RN"])
            else "미매칭 실폭도로"
        )
        rw_gu = str(row["구"])
        line_gu = (
            str(row["도로구"]).strip() if has_line and pd.notna(row["도로구"]) else ""
        )
        # API 매칭 — 라인의 구 또는 폴리곤의 구로 키 시도
        road_key = normalize_name(road_name)
        key_candidates = [
            f"{normalize_name(line_gu)}|{road_key}",
            f"{normalize_name(rw_gu)}|{road_key}",
        ]
        meta = next(
            (meta_by_key[key] for key in key_candidates if key in meta_by_key), None
        )

        official_width = as_float(row.get("ROAD_BT_NUM")) if has_line else None
        fallback_width = official_width_bucket(official_width)
        # 폭 출처 4단계 결정
        if not has_line:
            # 라인 매칭조차 실패한 폴리곤 — 폭 알 수 없음
            display_width = "불가"
            api_width = ""
            width_source = "실폭만"
            road_kind = ""
            road_function = ""
            road_scale = ""
        elif meta is not None:
            # API 메타 매칭 성공
            api_width = str(meta["API도로폭"]).strip() or "불가"
            if api_width != "불가":
                display_width = api_width
                width_source = "API"
            else:
                # API 폭이 '불가'면 공식 폭으로 대체
                display_width = fallback_width
                width_source = "공식폭보완"
            road_kind = str(meta["도로구분"])
            road_function = str(meta["도로기능"])
            road_scale = str(meta["도로규모"])
        else:
            # 라인은 있지만 API 메타가 없는 경우 — 공식 폭만 사용
            display_width = fallback_width
            api_width = ""
            width_source = "공식만"
            road_kind = "공식도로구간"
            road_function = ""
            road_scale = ""

        mg_idx = int(row["mg_idx"]) if has_line else None
        endpoints = (
            endpoint_lookup.get(mg_idx, {"start": None, "end": None})
            if mg_idx is not None
            else {"start": None, "end": None}
        )
        match_method = (
            str(row["매칭방식"]) if has_line and pd.notna(row["매칭방식"]) else "미매칭"
        )
        # JS 가 직접 사용할 properties dict
        rows.append(
            {
                "id": int(row["rw_idx"]),
                "rwSn": clean_json_value(row.get("RW_SN")),
                "gu": rw_gu,
                "roadGu": line_gu or rw_gu,
                "road": road_name,
                "roadKind": road_kind,
                "roadFunction": road_function,
                "roadScale": road_scale,
                "width": display_width,
                "apiWidth": api_width,
                "widthSource": width_source,
                "matchMethod": match_method,
                "matchDistanceM": round(float(row["_distance_m"]), 2)
                if has_line and pd.notna(row["_distance_m"])
                else None,
                "intersectLengthM": round(float(row["_intersect_len"]), 2)
                if has_line and pd.notna(row["_intersect_len"])
                else None,
                "officialWidthM": round(float(official_width), 2)
                if official_width is not None
                else None,
                "officialLengthM": round(float(row["ROAD_LT_NUM"]), 1)
                if has_line and pd.notna(row["ROAD_LT_NUM"])
                else None,
                "rdsManNo": clean_json_value(row.get("RDS_MAN_NO")),
                "rnCd": str(row["RN_CD"])
                if has_line and pd.notna(row["RN_CD"])
                else "",
                "start": endpoints["start"],
                "end": endpoints["end"],
                "color": WIDTH_COLORS.get(display_width, WIDTH_COLORS["불가"]),
            }
        )

    props = pd.DataFrame(rows)
    enriched = pd.concat(
        [gdf.reset_index(drop=True), props.reset_index(drop=True)], axis=1
    )

    # API 에는 있지만 공식 라인엔 없는 도로 — 후속 검토용
    matched_manage_keys = set(manage["_line_key"])
    api_only = input_df[~input_df["_key"].isin(matched_manage_keys)].copy()
    unmatched_rw = build_unmatched_rw_csv(enriched)
    return enriched, api_only, unmatched_rw


def build_unmatched_rw_csv(enriched: gpd.GeoDataFrame) -> pd.DataFrame:
    """미매칭 폴리곤 — 중심점 좌표 포함 CSV (수동 검토용)."""
    unmatched = enriched[enriched["matchMethod"] == "미매칭"].copy()
    if unmatched.empty:
        return pd.DataFrame(
            columns=["실폭도로일련번호", "시군구코드", "구", "중심위도", "중심경도"]
        )
    centroids = gpd.GeoSeries(unmatched.geometry.centroid, crs=SRC_CRS).to_crs(
        "EPSG:4326"
    )
    return pd.DataFrame(
        {
            "실폭도로일련번호": unmatched["RW_SN"].to_numpy(),
            "시군구코드": unmatched["SIG_CD"].to_numpy(),
            "구": unmatched["구"].to_numpy(),
            "중심위도": [round(point.y, 7) for point in centroids],
            "중심경도": [round(point.x, 7) for point in centroids],
        }
    )


def build_summary(enriched: gpd.GeoDataFrame) -> pd.DataFrame:
    """(구 × 매칭방식 × 폭출처) 단위 폴리곤 수 요약."""
    summary = (
        enriched.groupby(["gu", "matchMethod", "widthSource"], dropna=False)
        .size()
        .reset_index(name="폴리곤수")
        .rename(
            columns={"gu": "구", "matchMethod": "매칭방식", "widthSource": "폭출처"}
        )
        .sort_values(["구", "매칭방식", "폭출처"])
    )
    return summary


def build_geojson(enriched: gpd.GeoDataFrame) -> dict[str, Any]:
    """폴리곤을 단순화 + 위경도 변환 후 GeoJSON dict 반환 — 파일 크기 절감."""
    out = enriched.copy()
    # Douglas-Peucker 단순화 — 위상 보존
    out["geometry"] = out.geometry.simplify(
        SIMPLIFY_TOLERANCE_M, preserve_topology=True
    )
    out["geometry"] = out.geometry.make_valid().map(extract_polygonal)
    out = out[out.geometry.notna() & ~out.geometry.is_empty].copy()
    out = out.to_crs("EPSG:4326")

    # GeoJSON 의 properties 로 보존할 컬럼들
    prop_columns = [
        "id",
        "rwSn",
        "gu",
        "roadGu",
        "road",
        "roadKind",
        "roadFunction",
        "roadScale",
        "width",
        "apiWidth",
        "widthSource",
        "matchMethod",
        "matchDistanceM",
        "intersectLengthM",
        "officialWidthM",
        "officialLengthM",
        "rdsManNo",
        "rnCd",
        "start",
        "end",
        "color",
    ]
    features = []
    for _, row in out.iterrows():
        geom = mapping(row.geometry)
        # 좌표 모두 6자리 반올림 (파일 크기 추가 절감)
        geom["coordinates"] = round_nested(geom["coordinates"])
        properties = {col: clean_json_value(row[col]) for col in prop_columns}
        # start/end 는 list 가 아니면 None
        properties["start"] = row["start"] if isinstance(row["start"], list) else None
        properties["end"] = row["end"] if isinstance(row["end"], list) else None
        features.append({"type": "Feature", "geometry": geom, "properties": properties})
    return {"type": "FeatureCollection", "features": features}


def source_counts(enriched: gpd.GeoDataFrame) -> dict[str, int]:
    """폭 출처별 폴리곤 수 — HTML 패널 메타용."""
    return {name: int((enriched["widthSource"] == name).sum()) for name in SOURCE_ORDER}


def match_counts(enriched: gpd.GeoDataFrame) -> dict[str, int]:
    """매칭 방식별 폴리곤 수 — HTML 패널 메타용."""
    return {name: int((enriched["matchMethod"] == name).sum()) for name in MATCH_ORDER}


def build_html(enriched: gpd.GeoDataFrame, geojson: dict[str, Any]) -> str:
    """단독 Leaflet HTML 생성 — 검색/구/도로폭/폭출처/매칭 4종 필터 패널 포함.

    구현 메모:
        - 다른 road_width 시각화와 달리 raw 문자열 + __PLACEHOLDER__ 치환 방식 사용
          → JSON/숫자 임베드 시 f-string 의 { } 이스케이프 부담 회피.
    """
    src_counts = source_counts(enriched)
    m_counts = match_counts(enriched)
    geojson_json = json.dumps(geojson, ensure_ascii=False, separators=(",", ":"))
    template = r"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>서울 10개구 실폭도로 면지도</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <style>
    :root {
      --ink: #172033;
      --muted: #667085;
      --line: #d6deea;
      --panel: rgba(255,255,255,.965);
      --accent: #2563eb;
    }
    * { box-sizing: border-box; }
    html, body { height: 100%; margin: 0; }
    body {
      font-family: "Malgun Gothic", "Apple SD Gothic Neo", Arial, sans-serif;
      color: var(--ink);
      background: #edf2f7;
    }
    #map { position: fixed; inset: 0; }
    .panel {
      position: fixed;
      top: 16px;
      left: 16px;
      z-index: 1000;
      width: 408px;
      max-height: calc(100vh - 32px);
      overflow: auto;
      background: var(--panel);
      border: 1px solid rgba(116, 128, 150, .32);
      border-radius: 8px;
      box-shadow: 0 18px 40px rgba(28,39,64,.18);
      backdrop-filter: blur(8px);
    }
    .panel-header { padding: 16px; border-bottom: 1px solid var(--line); }
    h1 { margin: 0 0 8px; font-size: 19px; line-height: 1.28; font-weight: 800; letter-spacing: 0; }
    .note { margin: 7px 0 0; color: var(--muted); font-size: 12px; line-height: 1.45; }
    .meta { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 8px; margin-top: 12px; }
    .metric { min-width: 0; padding: 9px 7px; border: 1px solid var(--line); border-radius: 6px; background: #f8fafc; }
    .metric strong { display: block; font-size: 16px; line-height: 1.1; }
    .metric span { display: block; margin-top: 4px; font-size: 11px; color: var(--muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .panel-body { padding: 14px 16px 16px; }
    .field { margin-bottom: 14px; }
    .field label.title { display: block; margin-bottom: 7px; font-size: 12px; font-weight: 700; color: #35415a; }
    .search-row { display: grid; grid-template-columns: 1fr 40px; gap: 8px; }
    input[type="search"] { width: 100%; height: 36px; padding: 0 10px; border: 1px solid #bcc6d8; border-radius: 6px; font-size: 13px; outline: none; background: white; }
    input[type="search"]:focus { border-color: var(--accent); box-shadow: 0 0 0 3px rgba(37,99,235,.14); }
    button { height: 36px; border: 1px solid #bcc6d8; border-radius: 6px; background: white; color: #253148; font-weight: 800; font-size: 17px; cursor: pointer; line-height: 1; }
    button:hover { background: #f1f5fb; }
    .chips { display: flex; flex-wrap: wrap; gap: 7px; }
    .chip { display: inline-flex; align-items: center; gap: 6px; min-height: 30px; max-width: 100%; padding: 5px 8px; border: 1px solid #c8d0df; border-radius: 6px; background: white; font-size: 12px; line-height: 1.2; user-select: none; cursor: pointer; }
    .chip input { margin: 0; }
    .swatch { width: 10px; height: 10px; border-radius: 50%; flex: 0 0 auto; border: 1px solid rgba(0,0,0,.16); }
    .rows { display: grid; gap: 6px; font-size: 12px; }
    .bar-row { display: grid; grid-template-columns: 64px 1fr 42px; align-items: center; gap: 8px; min-height: 22px; }
    .bar-label, .bar-count { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .bar-track { height: 8px; border-radius: 999px; background: #e8edf5; overflow: hidden; }
    .bar-fill { height: 100%; min-width: 2px; border-radius: 999px; background: #4169a8; }
    .legend-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 6px 8px; font-size: 12px; }
    .legend-item { display: flex; align-items: center; min-width: 0; gap: 6px; }
    .legend-item span:last-child { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .leaflet-popup-content { font-family: "Malgun Gothic", "Apple SD Gothic Neo", Arial, sans-serif; font-size: 12px; margin: 12px; min-width: 276px; }
    .popup-title { font-size: 14px; font-weight: 800; margin-bottom: 6px; }
    .popup-table { width: 100%; border-collapse: collapse; }
    .popup-table th { width: 100px; text-align: left; color: #667085; font-weight: 700; padding: 3px 8px 3px 0; vertical-align: top; }
    .popup-table td { padding: 3px 0; vertical-align: top; }
    @media (max-width: 760px) {
      .panel { top: auto; left: 8px; right: 8px; bottom: 8px; width: auto; max-height: 54vh; }
      .panel-header { padding: 13px; }
      .panel-body { padding: 12px 13px 13px; }
      h1 { font-size: 17px; }
      .meta { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
  </style>
</head>
<body>
  <div id="map" aria-label="서울 10개구 실폭도로 면지도"></div>
  <aside class="panel">
    <div class="panel-header">
      <h1>서울 10개구 실폭도로</h1>
      <p class="note">행안부 TL_SPRD_RW 폴리곤 · 도로폭 API 우선 · 도로구간 중심선 공간 매칭</p>
      <div class="meta">
        <div class="metric"><strong id="visibleCount">0</strong><span>표시 면</span></div>
        <div class="metric"><strong>__DIRECT_COUNT__</strong><span>교차</span></div>
        <div class="metric"><strong>__NEAR_COUNT__</strong><span>근접</span></div>
        <div class="metric"><strong>__UNMATCHED_COUNT__</strong><span>미매칭</span></div>
      </div>
      <p class="note">API __API_COUNT__ · 공식폭보완 __OFFICIAL_FILL_COUNT__ · 공식만 __OFFICIAL_ONLY_COUNT__ · 실폭만 __RW_ONLY_COUNT__</p>
    </div>
    <div class="panel-body">
      <div class="field">
        <label class="title" for="search">검색</label>
        <div class="search-row">
          <input id="search" type="search" placeholder="도로명">
          <button id="reset" title="초기화" aria-label="초기화">↺</button>
        </div>
      </div>
      <div class="field">
        <label class="title">구</label>
        <div id="guFilters" class="chips"></div>
      </div>
      <div class="field">
        <label class="title">도로폭</label>
        <div id="widthFilters" class="chips"></div>
      </div>
      <div class="field">
        <label class="title">폭 출처</label>
        <div id="sourceFilters" class="chips"></div>
      </div>
      <div class="field">
        <label class="title">매칭</label>
        <div id="matchFilters" class="chips"></div>
      </div>
      <div class="field">
        <label class="title">구별 표시 면</label>
        <div id="guSummary" class="rows"></div>
      </div>
      <div class="field">
        <label class="title">도로폭 색상</label>
        <div id="legend" class="legend-grid"></div>
      </div>
    </div>
  </aside>
  <script>
    // 파이썬에서 임베드한 데이터 (placeholder 치환)
    const GEOJSON = __GEOJSON_DATA__;
    const GU_ORDER = __GU_ORDER__;
    const WIDTH_ORDER = __WIDTH_ORDER__;
    const SOURCE_ORDER = __SOURCE_ORDER__;
    const MATCH_ORDER = __MATCH_ORDER__;
    const WIDTH_COLORS = __WIDTH_COLORS__;
    const canvasRenderer = L.canvas({ padding: 0.5 });
    const map = L.map("map", { preferCanvas: true, zoomControl: false }).setView([37.55, 126.99], 11);
    L.control.zoom({ position: "bottomright" }).addTo(map);
    L.tileLayer("https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png", {
      maxZoom: 19,
      attribution: '&copy; OpenStreetMap contributors &copy; CARTO'
    }).addTo(map);

    let DATA = [];
    let firstRender = true;
    // GeoJSON 레이어 + 클릭 시 끝점 강조 레이어
    const geoLayer = L.geoJSON(null, { renderer: canvasRenderer, style: styleFeature, onEachFeature }).addTo(map);
    const endpointLayer = L.layerGroup().addTo(map);

    function escapeHtml(value) {
      // XSS 방지
      return String(value ?? "").replace(/[&<>"']/g, (char) => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
      }[char]));
    }
    function countBy(prop, order) {
      // 피처 prop 값별 카운트 — 사전 정의 순서 유지 + 0 제외
      const counts = new Map(order.map((name) => [name, 0]));
      DATA.forEach((feature) => {
        const value = feature.properties[prop] || "";
        counts.set(value, (counts.get(value) || 0) + 1);
      });
      return order.filter((name) => counts.get(name)).map((name) => ({ name, count: counts.get(name) }));
    }
    function createCheckbox(containerId, name, value, count, color) {
      // 칩형 체크박스 — 색상 점이 옵션
      const label = document.createElement("label");
      label.className = "chip";
      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.name = name;
      checkbox.value = value;
      checkbox.checked = true;
      checkbox.addEventListener("change", render);
      label.appendChild(checkbox);
      if (color) {
        const swatch = document.createElement("span");
        swatch.className = "swatch";
        swatch.style.background = color;
        label.appendChild(swatch);
      }
      const text = document.createElement("span");
      text.textContent = `${value} ${count.toLocaleString("ko-KR")}`;
      label.appendChild(text);
      document.getElementById(containerId).appendChild(label);
    }
    function setupFilters() {
      // 구/도로폭/폭출처/매칭 4종 필터 패널 동적 생성
      countBy("gu", GU_ORDER).forEach((item) => createCheckbox("guFilters", "gu", item.name, item.count));
      countBy("width", WIDTH_ORDER).forEach((item) => createCheckbox("widthFilters", "width", item.name, item.count, WIDTH_COLORS[item.name]));
      countBy("widthSource", SOURCE_ORDER).forEach((item) => createCheckbox("sourceFilters", "source", item.name, item.count));
      countBy("matchMethod", MATCH_ORDER).forEach((item) => createCheckbox("matchFilters", "match", item.name, item.count));
    }
    function getSelected(name) {
      return new Set([...document.querySelectorAll(`input[name="${name}"]:checked`)].map((el) => el.value));
    }
    function formatMaybe(value, suffix = "") {
      if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
      return Number(value).toLocaleString("ko-KR", { maximumFractionDigits: 1 }) + suffix;
    }
    function coordText(value) {
      // 위경도 좌표 표시 (소수 6자리)
      if (!Array.isArray(value)) return "-";
      return `${Number(value[0]).toFixed(6)}, ${Number(value[1]).toFixed(6)}`;
    }
    function popupHtml(feature) {
      // 폴리곤 클릭 시 팝업 — API/공식 폭 동시 표시 + 매칭 거리 + 시작/끝
      const p = feature.properties;
      const apiWidth = p.apiWidth || "-";
      return `
        <div class="popup-title">${escapeHtml(p.road)}</div>
        <table class="popup-table">
          <tr><th>구</th><td>${escapeHtml(p.gu)}</td></tr>
          <tr><th>표시 폭</th><td>${escapeHtml(p.width)} <small>(${escapeHtml(p.widthSource)})</small></td></tr>
          <tr><th>API 폭</th><td>${escapeHtml(apiWidth)}</td></tr>
          <tr><th>공식 폭</th><td>${formatMaybe(p.officialWidthM, " m")}</td></tr>
          <tr><th>매칭</th><td>${escapeHtml(p.matchMethod)} · ${formatMaybe(p.matchDistanceM, " m")}</td></tr>
          <tr><th>실폭번호</th><td>${escapeHtml(p.rwSn)}</td></tr>
          <tr><th>시작</th><td>${coordText(p.start)}</td></tr>
          <tr><th>끝</th><td>${coordText(p.end)}</td></tr>
        </table>`;
    }
    function styleFeature(feature) {
      // 폴리곤 색상/투명도 — 실폭만은 회색+옅게, 근접 매칭은 중간, 교차 매칭은 진하게
      const p = feature.properties;
      const color = p.color || WIDTH_COLORS["불가"];
      const isRwOnly = p.widthSource === "실폭만";
      const isNear = p.matchMethod === "근접";
      return {
        renderer: canvasRenderer,
        color: isRwOnly ? "#64748b" : color,
        weight: isNear ? 0.65 : 0.48,
        opacity: isRwOnly ? 0.35 : 0.55,
        fillColor: color,
        fillOpacity: isRwOnly ? 0.20 : (isNear ? 0.48 : 0.62)
      };
    }
    function onEachFeature(feature, layer) {
      // 마우스오버 강조 + 클릭 시 끝점 표시
      const p = feature.properties;
      layer
        .bindTooltip(`[${p.gu}] ${p.road} | ${p.width} | ${p.widthSource} | ${p.matchMethod}`, { sticky: true })
        .bindPopup(popupHtml(feature), { maxWidth: 380 });
      layer.on("mouseover", (event) => {
        event.target.setStyle({ color: "#111827", weight: 2.2, opacity: 0.95, fillOpacity: 0.88 });
      });
      layer.on("mouseout", (event) => {
        geoLayer.resetStyle(event.target);
      });
      layer.on("click", () => showEndpoints(p));
    }
    function showEndpoints(p) {
      // 시작-끝 직선(점선) + 시작점(검정)/끝점(빨강) 강조 마커
      endpointLayer.clearLayers();
      if (!Array.isArray(p.start) || !Array.isArray(p.end)) return;
      L.polyline([p.start, p.end], { color: "#111827", weight: 2.2, opacity: 0.85, dashArray: "5 4" }).addTo(endpointLayer);
      L.circleMarker(p.start, { radius: 7, color: "#ffffff", weight: 2, fillColor: "#111827", fillOpacity: 0.96 })
        .bindTooltip(`시작: ${p.road}`, { permanent: true, direction: "top", offset: [0, -8] })
        .addTo(endpointLayer);
      L.circleMarker(p.end, { radius: 7, color: "#ffffff", weight: 2, fillColor: "#f43f5e", fillOpacity: 0.96 })
        .bindTooltip(`끝: ${p.road}`, { permanent: true, direction: "top", offset: [0, -8] })
        .addTo(endpointLayer);
    }
    function filteredFeatures() {
      // 4종 체크박스 + 검색어 모두 만족하는 피처만
      const selectedGu = getSelected("gu");
      const selectedWidth = getSelected("width");
      const selectedSource = getSelected("source");
      const selectedMatch = getSelected("match");
      const query = document.getElementById("search").value.trim().toLowerCase();
      return DATA.filter((feature) => {
        const p = feature.properties;
        if (!selectedGu.has(p.gu)) return false;
        if (!selectedWidth.has(p.width)) return false;
        if (!selectedSource.has(p.widthSource)) return false;
        if (!selectedMatch.has(p.matchMethod)) return false;
        if (!query) return true;
        return `${p.road} ${p.gu} ${p.width} ${p.widthSource} ${p.matchMethod}`.toLowerCase().includes(query);
      });
    }
    function updateSummary(features) {
      const guCounts = new Map();
      features.forEach((feature) => {
        const gu = feature.properties.gu;
        guCounts.set(gu, (guCounts.get(gu) || 0) + 1);
      });
      document.getElementById("visibleCount").textContent = features.length.toLocaleString("ko-KR");
      const maxCount = Math.max(1, ...GU_ORDER.map((gu) => guCounts.get(gu) || 0));
      document.getElementById("guSummary").innerHTML = GU_ORDER.map((gu) => {
        const count = guCounts.get(gu) || 0;
        const pct = (count / maxCount) * 100;
        return `
          <div class="bar-row">
            <div class="bar-label">${gu}</div>
            <div class="bar-track"><div class="bar-fill" style="width:${pct}%"></div></div>
            <div class="bar-count">${count.toLocaleString("ko-KR")}</div>
          </div>`;
      }).join("");
    }
    function renderLegend() {
      const widthCounts = countBy("width", WIDTH_ORDER);
      document.getElementById("legend").innerHTML = widthCounts.map((item) => `
        <div class="legend-item">
          <span class="swatch" style="background:${WIDTH_COLORS[item.name]}"></span>
          <span>${item.name} ${item.count.toLocaleString("ko-KR")}</span>
        </div>`).join("");
    }
    function render() {
      const features = filteredFeatures();
      geoLayer.clearLayers();
      endpointLayer.clearLayers();
      geoLayer.addData({ type: "FeatureCollection", features });
      updateSummary(features);
      // 첫 렌더링에만 fitBounds (사용자 줌 보존)
      if (firstRender && features.length) {
        map.fitBounds(geoLayer.getBounds(), { padding: [26, 26] });
        firstRender = false;
      }
    }
    function boot() {
      DATA = GEOJSON.features;
      setupFilters();
      renderLegend();
      render();
    }
    document.getElementById("search").addEventListener("input", render);
    document.getElementById("reset").addEventListener("click", () => {
      document.getElementById("search").value = "";
      document.querySelectorAll('input[type="checkbox"]').forEach((el) => el.checked = true);
      render();
    });
    boot();
  </script>
</body>
</html>
"""
    # placeholder → 실제값 치환 (raw 문자열에서 JSON/숫자 안전하게 삽입)
    replacements = {
        "__GEOJSON_DATA__": geojson_json,
        "__GU_ORDER__": json.dumps(GU_ORDER, ensure_ascii=False),
        "__WIDTH_ORDER__": json.dumps(WIDTH_ORDER, ensure_ascii=False),
        "__SOURCE_ORDER__": json.dumps(SOURCE_ORDER, ensure_ascii=False),
        "__MATCH_ORDER__": json.dumps(MATCH_ORDER, ensure_ascii=False),
        "__WIDTH_COLORS__": json.dumps(WIDTH_COLORS, ensure_ascii=False),
        "__DIRECT_COUNT__": f"{m_counts['교차']:,}",
        "__NEAR_COUNT__": f"{m_counts['근접']:,}",
        "__UNMATCHED_COUNT__": f"{m_counts['미매칭']:,}",
        "__API_COUNT__": f"{src_counts['API']:,}",
        "__OFFICIAL_FILL_COUNT__": f"{src_counts['공식폭보완']:,}",
        "__OFFICIAL_ONLY_COUNT__": f"{src_counts['공식만']:,}",
        "__RW_ONLY_COUNT__": f"{src_counts['실폭만']:,}",
    }
    for token, value in replacements.items():
        template = template.replace(token, value)
    return template


def main() -> int:
    """메인 — API/중심선/실폭 로드 → 매칭 → enrich → GeoJSON/CSV/HTML 일괄 산출."""
    input_df = load_input()
    manage = load_manage()
    rw = load_rw()
    match_table = build_match_table(rw, manage)
    enriched, api_only, unmatched_rw = enrich_polygons(
        rw, manage, input_df, match_table
    )
    geojson = build_geojson(enriched)

    # GeoJSON 저장 — separators 로 공백 제거 (파일 크기 절감)
    OUTPUT_GEOJSON.write_text(
        json.dumps(geojson, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )
    OUTPUT_HTML.write_text(build_html(enriched, geojson), encoding="utf-8")
    build_summary(enriched).to_csv(
        OUTPUT_SUMMARY_CSV, index=False, encoding="utf-8-sig"
    )
    unmatched_rw.to_csv(OUTPUT_UNMATCHED_RW_CSV, index=False, encoding="utf-8-sig")
    api_only.drop(columns=[col for col in ["_key"] if col in api_only.columns]).to_csv(
        OUTPUT_API_ONLY_CSV, index=False, encoding="utf-8-sig"
    )

    src_counts = source_counts(enriched)
    m_counts = match_counts(enriched)
    print(f"RW polygons: {len(enriched)}")
    print(f"Direct matched polygons: {m_counts['교차']}")
    print(
        f"Nearest matched polygons: {m_counts['근접']} (<= {NEAREST_MAX_DISTANCE_M:g}m)"
    )
    print(f"Unmatched RW polygons: {m_counts['미매칭']}")
    print(
        "Width sources: "
        f"API={src_counts['API']}, official_fill={src_counts['공식폭보완']}, "
        f"official_only={src_counts['공식만']}, rw_only={src_counts['실폭만']}"
    )
    print(f"API-only roads: {len(api_only)}")
    print(f"HTML: {OUTPUT_HTML}")
    print(f"GeoJSON: {OUTPUT_GEOJSON}")
    print(f"Summary CSV: {OUTPUT_SUMMARY_CSV}")
    print(f"Unmatched RW CSV: {OUTPUT_UNMATCHED_RW_CSV}")
    print(f"API-only CSV: {OUTPUT_API_ONLY_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
