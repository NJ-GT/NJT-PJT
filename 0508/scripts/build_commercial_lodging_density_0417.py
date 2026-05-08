# -*- coding: utf-8 -*-
"""
서울 10개구 상가·숙박 밀집도 격자/법정동 GeoJSON + 인터랙티브 HTML 지도 생성.

목적:
    - 건물 SHP(AL_D010) 의 상가성 건물과 숙박시설 CSV 를 50,000㎡ 격자/법정동 두 단위로 집계해
      입체화재하중 밀도/개수 밀도 등 비교 지표 산출 + 단독 HTML 지도로 시각화.

핵심 결정 (gis_analysis.py 와 동일 컨벤션):
    - 상가 분류: 건물용도코드 03000(제1종근린생활시설) / 04000(제2종근린생활시설) / 07000(판매시설)
                 또는 용도명에 '근린생활시설' / '판매시설' 키워드 포함
    - 입체화재하중 분자 = 건축면적 × max(지상층수+지하층수, 1) (건축면적 0 이면 연면적)
    - 격자: EPSG:5186 좌표계 50,000㎡ 정사각형(약 223.6m × 223.6m)

입력:
    - data/AL_D010_11_20260409/AL_D010_11_20260409_filtered.shp  (건물 SHP)
    - data/통합숙박시설최종안0415.csv                            (숙박 CSV)
    - data/[오피셜]법정동승인일자_공간정보0415.geojson           (법정동 경계)

출력 (모두 data/상가숙소밀집도_10개구_0417/):
    - 서울10개구_상가숙소_법정동별_밀집도.csv / .geojson
    - 서울10개구_상가숙소_격자별_밀집도.csv / .geojson
    - 서울10개구_상가숙소_시각화컬럼정의.csv  (각 컬럼 설명/원천)
    - 서울10개구_상가숙소_밀집도지도.html      (Leaflet 단독 HTML)
    - 서울10개구_상가숙소_산출요약.json
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import geopandas as gpd
import pandas as pd
from pyproj import Transformer
from shapely.geometry import box


# 경로
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
BUILDING_DIR = DATA_DIR / "AL_D010_11_20260409"
OUT_DIR = DATA_DIR / "상가숙소밀집도_10개구_0417"

BUILDING_SHP = BUILDING_DIR / "AL_D010_11_20260409_filtered.shp"
LODGING_CSV = DATA_DIR / "통합숙박시설최종안0415.csv"
LEGAL_DONG_GEOJSON = DATA_DIR / "[오피셜]법정동승인일자_공간정보0415.geojson"

# 격자 면적 — 50,000㎡ → 한 변 ≈ 223.6m
GRID_AREA_M2 = 50_000.0
GRID_SIDE_M = math.sqrt(GRID_AREA_M2)

# 시군구코드 5자리 → 자치구명 (서울 10구만)
GU_MAP = {
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

# 상가성 건물 분류 — gis_analysis.py 와 동일 컨벤션
# 03000=제1종근린생활시설, 04000=제2종근린생활시설, 07000=판매시설
COMMERCIAL_CODES = {"03000", "04000", "07000"}
# 용도명 키워드 fallback — 코드가 비어있을 때 대비
COMMERCIAL_KEYWORDS = ("근린생활시설", "판매시설")

# AL_D010 SHP 의 A0~A29 컬럼 → 의미 있는 한글 컬럼명 (인덱스 순서대로 매핑)
BUILDING_COLUMNS = [
    "건물유형코드",
    "건물ID",
    "건물관리번호",
    "법정동코드",
    "주소",
    "지번",
    "주부속구분",
    "일반집합구분",
    "건물용도코드",
    "건물용도명",
    "구조코드",
    "구조명",
    "건폐율",
    "사용승인일",
    "연면적",
    "건축면적",
    "높이",
    "기타",
    "용적률",
    "지번코드",
    "대장구분",
    "건물식별번호",
    "갱신일자",
    "행정구역코드",
    "예비1",
    "예비2",
    "지상층수",
    "지하층수",
    "생성일자",
    "총층수",
]


def log(message: str) -> None:
    """flush 진행 로그 — 긴 SHP 처리 중 단계 진행 가시화."""
    print(message, flush=True)


def fix_mojibake(value: object) -> str:
    """SHP 한글 깨짐(latin1↔cp949) 복구 — 실패 시 원본 유지."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    try:
        # latin1 로 잘못 디코딩된 cp949 바이트를 복구
        return text.encode("latin1").decode("cp949")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


def normalize_code(value: object, width: int) -> str:
    """행정코드(시군구/법정동) 를 width 자리 0패딩 — '.0' 접미사도 정리."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    if not text or text.lower() == "nan":
        return ""
    return text.zfill(width)


def split_gu_dong(address: str) -> tuple[str, str]:
    """주소 문자열에서 (구, 동) 추출 — '서울특별시 OO구 XX동 ...' 패턴."""
    parts = str(address).split()
    gu = parts[1] if len(parts) >= 2 else ""
    dong = parts[2] if len(parts) >= 3 else ""
    return gu, dong


def add_grid_columns(
    df: pd.DataFrame, x_col: str, y_col: str, origin_x: float, origin_y: float
) -> pd.DataFrame:
    """좌표 → 격자 행/열/ID 부여 — origin_xy 를 (1,1) 로 두고 GRID_SIDE_M 단위 칸 인덱스."""
    result = df.copy()
    # +1 — 1-based 인덱싱 (G50K_R0001_C0001 가 좌하단)
    result["그리드열"] = ((result[x_col] - origin_x) // GRID_SIDE_M).astype("Int64") + 1
    result["그리드행"] = ((result[y_col] - origin_y) // GRID_SIDE_M).astype("Int64") + 1
    result["그리드ID"] = result.apply(
        lambda r: f"G50K_R{int(r['그리드행']):04d}_C{int(r['그리드열']):04d}", axis=1
    )
    return result


def load_buildings() -> gpd.GeoDataFrame:
    """건물 SHP 로드 + 한글 깨짐 복구 + 상가 분류 + 입체화재하중 분자 + 위경도 변환."""
    log("1. 10개구 건물 SHP 로드 및 상가 분류")
    gdf = gpd.read_file(BUILDING_SHP)
    # A0~A29 → 한글 컬럼명 일괄 변경
    gdf = gdf.rename(columns={f"A{i}": name for i, name in enumerate(BUILDING_COLUMNS)})

    # 깨질 수 있는 텍스트 컬럼만 mojibake 복구
    text_cols = [
        "건물ID",
        "건물관리번호",
        "법정동코드",
        "주소",
        "지번",
        "주부속구분",
        "일반집합구분",
        "건물용도코드",
        "건물용도명",
        "구조코드",
        "구조명",
        "대장구분",
        "건물식별번호",
        "행정구역코드",
        "예비1",
        "예비2",
    ]
    for col in text_cols:
        if col in gdf.columns:
            gdf[col] = gdf[col].map(fix_mojibake)

    # 법정동코드 10자리 → 시군구코드 5자리 → 자치구명
    gdf["법정동코드"] = gdf["법정동코드"].map(lambda v: normalize_code(v, 10))
    gdf["시군구코드"] = gdf["법정동코드"].str[:5]
    gdf["구"] = gdf["시군구코드"].map(GU_MAP)
    # 주소 fallback — 코드가 매칭되지 않으면 주소에서 직접 구/동 추출
    gu_dong = gdf["주소"].map(split_gu_dong)
    gdf["주소_구"] = gu_dong.map(lambda x: x[0])
    gdf["법정동명"] = gu_dong.map(lambda x: x[1])
    gdf["구"] = gdf["구"].fillna(gdf["주소_구"])

    # 면적/층수 숫자형 + 결측 0
    for col in ["연면적", "건축면적", "지상층수", "지하층수"]:
        gdf[col] = pd.to_numeric(gdf[col], errors="coerce").fillna(0.0)
    # 0층 보정 — 지상층수 0이면 1로 (필로티/지층만 있는 경우)
    gdf.loc[gdf["지상층수"] == 0, "지상층수"] = 1.0
    gdf["총층수"] = (gdf["지상층수"] + gdf["지하층수"]).clip(lower=1)
    # 입체화재하중 분자 = 건축면적 × 총층수 (없으면 연면적 fallback)
    floor_area_proxy = gdf["건축면적"] * gdf["총층수"]
    gdf["입체화재하중_분자"] = floor_area_proxy.where(
        floor_area_proxy > 0, gdf["연면적"]
    )

    # 상가 분류 — 코드 매칭 또는 용도명 키워드
    use_code = gdf["건물용도코드"].astype(str).str.zfill(5)
    use_name = gdf["건물용도명"].fillna("").astype(str)
    keyword_mask = use_name.apply(
        lambda x: any(keyword in x for keyword in COMMERCIAL_KEYWORDS)
    )
    gdf["상가여부"] = use_code.isin(COMMERCIAL_CODES) | keyword_mask

    # 중심점 → EPSG:5186 평면좌표 (격자 계산용) + WGS84 (지도 표시용)
    centroids = gdf.geometry.centroid
    gdf["x_EPSG5186"] = centroids.x
    gdf["y_EPSG5186"] = centroids.y
    to_wgs84 = Transformer.from_crs("EPSG:5186", "EPSG:4326", always_xy=True)
    lng, lat = to_wgs84.transform(
        gdf["x_EPSG5186"].to_numpy(), gdf["y_EPSG5186"].to_numpy()
    )
    gdf["경도"] = lng
    gdf["위도"] = lat

    # 서울 10구 외 행 제거
    return gdf[gdf["시군구코드"].isin(GU_MAP.keys())].copy()


def load_lodgings() -> pd.DataFrame:
    """숙박 CSV 로드 + 자치구 코드 정규화 + 입체화재하중 분자 + EPSG:5186 변환."""
    log("2. 통합숙박시설 CSV 로드")
    df = pd.read_csv(LODGING_CSV, encoding="utf-8-sig")
    df["시군구코드"] = df["시군구코드"].map(lambda v: normalize_code(v, 5))
    df = df[df["시군구코드"].isin(GU_MAP.keys())].copy()
    df["구"] = df["시군구코드"].map(GU_MAP)
    # 법정동코드는 5자리(동) + 시군구 5자리 → 10자리로 결합
    df["법정동코드5"] = df["법정동코드"].map(lambda v: normalize_code(v, 5))
    df["법정동코드"] = df["시군구코드"] + df["법정동코드5"]
    # 대지위치에서 '서울특별시 OO구 XX동' 패턴으로 동 추출
    df["법정동명"] = (
        df["대지위치"]
        .fillna("")
        .astype(str)
        .str.extract(r"서울특별시\s+\S+\s+([^\s]+)", expand=False)
        .fillna("")
    )

    # 면적/층수 숫자형
    for col in ["위도", "경도", "연면적(㎡)", "지상층수", "지하층수"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    df.loc[df["지상층수"] == 0, "지상층수"] = 1.0
    df["총층수"] = (df["지상층수"] + df["지하층수"]).clip(lower=1)
    # 숙박은 건축면적이 없어 연면적 자체를 분자로 사용
    df["숙박_입체화재하중_분자"] = df["연면적(㎡)"]

    # 위경도 → EPSG:5186 (격자 계산용)
    to_5186 = Transformer.from_crs("EPSG:4326", "EPSG:5186", always_xy=True)
    x, y = to_5186.transform(df["경도"].to_numpy(), df["위도"].to_numpy())
    df["x_EPSG5186"] = x
    df["y_EPSG5186"] = y
    return df


def load_legal_dongs() -> gpd.GeoDataFrame:
    """법정동 경계 GeoJSON 로드 + 코드 정규화 + 면적(m²/ha) 산출."""
    log("3. 법정동 경계 로드")
    gdf = gpd.read_file(LEGAL_DONG_GEOJSON)
    # 법정동코드 10자리 우선, 없으면 EMD_CD 8자리 + '00' (관리구역 미정)
    gdf["법정동코드_정규"] = gdf.apply(
        lambda r: (
            normalize_code(r.get("법정동코드"), 10)
            or (normalize_code(r.get("EMD_CD"), 8) + "00")
        ),
        axis=1,
    )
    gdf["시군구코드"] = gdf["법정동코드_정규"].str[:5]
    gdf = gdf[gdf["시군구코드"].isin(GU_MAP.keys())].copy()
    gdf["구"] = gdf["시군구코드"].map(GU_MAP)
    gdf["법정동명"] = gdf["법정동명"].fillna(gdf["EMD_KOR_NM"])
    # 면적은 5186 평면좌표에서 계산 — 위경도 면적은 의미 없음
    gdf["면적_m2"] = gdf.to_crs("EPSG:5186").geometry.area
    gdf["면적_ha"] = gdf["면적_m2"] / 10_000.0
    keep_cols = [
        "EMD_CD",
        "EMD_KOR_NM",
        "법정동코드_정규",
        "시군구코드",
        "구",
        "법정동명",
        "면적_m2",
        "면적_ha",
        "geometry",
    ]
    return gdf[keep_cols].copy()


def aggregate_by_dong(
    buildings: gpd.GeoDataFrame, lodgings: pd.DataFrame, legal: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """법정동별 상가/숙박 통계 + 밀도 지표 계산."""
    log("4. 법정동별 상가/숙박 밀집도 계산")
    commercial = buildings[buildings["상가여부"]].copy()
    # 3개 그룹별 통계 — 전체건물 / 상가 / 숙박
    all_building_stats = buildings.groupby("법정동코드", dropna=False).agg(
        전체건물수=("건물ID", "count"),
        전체연면적합계_m2=("연면적", "sum"),
    )
    commercial_stats = commercial.groupby("법정동코드", dropna=False).agg(
        상가수=("건물ID", "count"),
        상가연면적합계_m2=("연면적", "sum"),
        상가입체화재하중_분자=("입체화재하중_분자", "sum"),
    )
    lodging_stats = lodgings.groupby("법정동코드", dropna=False).agg(
        숙박시설수=("사업장명", "count"),
        숙박연면적합계_m2=("연면적(㎡)", "sum"),
        숙박입체화재하중_분자=("숙박_입체화재하중_분자", "sum"),
    )

    # 경계 GDF 에 3종 통계 좌측 결합
    out = legal.copy()
    out = out.merge(
        all_building_stats, how="left", left_on="법정동코드_정규", right_index=True
    )
    out = out.merge(
        commercial_stats, how="left", left_on="법정동코드_정규", right_index=True
    )
    out = out.merge(
        lodging_stats, how="left", left_on="법정동코드_정규", right_index=True
    )

    # 결측은 0 으로 (해당 동에 데이터 없음)
    numeric_cols = [
        "전체건물수",
        "전체연면적합계_m2",
        "상가수",
        "상가연면적합계_m2",
        "상가입체화재하중_분자",
        "숙박시설수",
        "숙박연면적합계_m2",
        "숙박입체화재하중_분자",
    ]
    out[numeric_cols] = out[numeric_cols].fillna(0.0)
    out["복합입체화재하중_분자"] = (
        out["상가입체화재하중_분자"] + out["숙박입체화재하중_분자"]
    )

    # ha당 밀도 / 면적당 입체화재하중 밀도 / 합산 개수 등 파생지표
    out["상가수_per_ha"] = out["상가수"] / out["면적_ha"]
    out["숙박시설수_per_ha"] = out["숙박시설수"] / out["면적_ha"]
    out["상가연면적_per_ha"] = out["상가연면적합계_m2"] / out["면적_ha"]
    out["숙박연면적_per_ha"] = out["숙박연면적합계_m2"] / out["면적_ha"]
    out["상가_입체화재하중밀도"] = out["상가입체화재하중_분자"] / out["면적_m2"]
    out["숙박_입체화재하중밀도"] = out["숙박입체화재하중_분자"] / out["면적_m2"]
    out["복합_입체화재하중밀도"] = out["복합입체화재하중_분자"] / out["면적_m2"]
    out["상가숙박_개수"] = out["상가수"] + out["숙박시설수"]
    out["상가숙박_개수_per_ha"] = out["상가숙박_개수"] / out["면적_ha"]

    # 정수형/소수점 정리
    for col in ["전체건물수", "상가수", "숙박시설수", "상가숙박_개수"]:
        out[col] = out[col].round().astype(int)
    for col in out.select_dtypes(include="number").columns:
        out[col] = out[col].round(6)
    return out


def dominant_area(points: pd.DataFrame) -> pd.DataFrame:
    """격자별 최빈 (구, 법정동) 산출 + 격자에 속한 법정동 종류 수."""
    grouped = (
        points.groupby(["그리드ID", "구", "법정동코드", "법정동명"], dropna=False)
        .size()
        .reset_index(name="point_count")
        .sort_values(["그리드ID", "point_count"], ascending=[True, False])
    )
    # 그리드별 가장 많이 속한 (구, 동) 1행 — 대표값
    primary = grouped.drop_duplicates("그리드ID").rename(
        columns={
            "구": "주요구",
            "법정동코드": "주요법정동코드",
            "법정동명": "주요법정동명",
        }
    )
    # 한 격자가 여러 법정동을 걸치는 경우 그 종류 수
    dong_count = (
        grouped.groupby("그리드ID")["법정동코드"].nunique().rename("포함법정동수")
    )
    return primary.merge(dong_count, on="그리드ID", how="left")


def aggregate_by_grid(
    buildings: gpd.GeoDataFrame, lodgings: pd.DataFrame
) -> gpd.GeoDataFrame:
    """50,000㎡ 정사각 격자별 상가/숙박 통계 — GeoDataFrame(폴리곤) 반환."""
    log("5. 50,000㎡ 격자별 상가/숙박 밀집도 계산")
    # 모든 점의 최소 좌표를 격자 원점(좌하단)으로 — 연속성 보장
    all_x = pd.concat(
        [buildings["x_EPSG5186"], lodgings["x_EPSG5186"]], ignore_index=True
    )
    all_y = pd.concat(
        [buildings["y_EPSG5186"], lodgings["y_EPSG5186"]], ignore_index=True
    )
    origin_x = math.floor(all_x.min() / GRID_SIDE_M) * GRID_SIDE_M
    origin_y = math.floor(all_y.min() / GRID_SIDE_M) * GRID_SIDE_M

    # 각 점에 그리드 인덱스 부착
    buildings_g = add_grid_columns(
        buildings, "x_EPSG5186", "y_EPSG5186", origin_x, origin_y
    )
    lodgings_g = add_grid_columns(
        lodgings, "x_EPSG5186", "y_EPSG5186", origin_x, origin_y
    )

    commercial = buildings_g[buildings_g["상가여부"]].copy()
    # 그리드별 3종 통계
    all_building_stats = buildings_g.groupby("그리드ID", dropna=False).agg(
        전체건물수=("건물ID", "count"),
    )
    commercial_stats = commercial.groupby("그리드ID", dropna=False).agg(
        상가수=("건물ID", "count"),
        상가연면적합계_m2=("연면적", "sum"),
        상가입체화재하중_분자=("입체화재하중_분자", "sum"),
    )
    lodging_stats = lodgings_g.groupby("그리드ID", dropna=False).agg(
        숙박시설수=("사업장명", "count"),
        숙박연면적합계_m2=("연면적(㎡)", "sum"),
        숙박입체화재하중_분자=("숙박_입체화재하중_분자", "sum"),
    )

    # 상가 또는 숙박이 1개라도 있는 그리드만 남김
    grid_keys = pd.Index(commercial_stats.index).union(lodging_stats.index)
    out = pd.DataFrame(index=grid_keys)
    out.index.name = "그리드ID"
    out = out.merge(all_building_stats, how="left", left_index=True, right_index=True)
    out = out.merge(commercial_stats, how="left", left_index=True, right_index=True)
    out = out.merge(lodging_stats, how="left", left_index=True, right_index=True)
    out = out.fillna(0.0).reset_index()

    # ID 에서 행/열 추출 + bbox 좌표 산출
    row_col = out["그리드ID"].str.extract(r"G50K_R(\d+)_C(\d+)").astype(int)
    out["그리드행"] = row_col[0]
    out["그리드열"] = row_col[1]
    out["x_min_EPSG5186"] = origin_x + (out["그리드열"] - 1) * GRID_SIDE_M
    out["y_min_EPSG5186"] = origin_y + (out["그리드행"] - 1) * GRID_SIDE_M
    out["x_max_EPSG5186"] = out["x_min_EPSG5186"] + GRID_SIDE_M
    out["y_max_EPSG5186"] = out["y_min_EPSG5186"] + GRID_SIDE_M
    out["중심x_EPSG5186"] = (out["x_min_EPSG5186"] + out["x_max_EPSG5186"]) / 2
    out["중심y_EPSG5186"] = (out["y_min_EPSG5186"] + out["y_max_EPSG5186"]) / 2

    # 중심점 위경도 변환 (지도 표시용)
    to_wgs84 = Transformer.from_crs("EPSG:5186", "EPSG:4326", always_xy=True)
    lng, lat = to_wgs84.transform(
        out["중심x_EPSG5186"].to_numpy(), out["중심y_EPSG5186"].to_numpy()
    )
    out["중심경도"] = lng
    out["중심위도"] = lat

    # 그리드 면적은 GRID_AREA_M2 고정 (격자 자체가 정사각형)
    out["기준면적_m2"] = GRID_AREA_M2
    out["기준면적_ha"] = GRID_AREA_M2 / 10_000.0
    out["복합입체화재하중_분자"] = (
        out["상가입체화재하중_분자"] + out["숙박입체화재하중_분자"]
    )
    out["상가수_per_ha"] = out["상가수"] / out["기준면적_ha"]
    out["숙박시설수_per_ha"] = out["숙박시설수"] / out["기준면적_ha"]
    out["상가연면적_per_ha"] = out["상가연면적합계_m2"] / out["기준면적_ha"]
    out["숙박연면적_per_ha"] = out["숙박연면적합계_m2"] / out["기준면적_ha"]
    out["상가_입체화재하중밀도"] = out["상가입체화재하중_분자"] / GRID_AREA_M2
    out["숙박_입체화재하중밀도"] = out["숙박입체화재하중_분자"] / GRID_AREA_M2
    out["복합_입체화재하중밀도"] = out["복합입체화재하중_분자"] / GRID_AREA_M2
    out["상가숙박_개수"] = out["상가수"] + out["숙박시설수"]
    out["상가숙박_개수_per_ha"] = out["상가숙박_개수"] / out["기준면적_ha"]
    # 격자 안에 데이터 자체가 0인지(공백 표시용)
    out["건물데이터_0여부"] = out["전체건물수"].eq(0).map({True: "Y", False: "N"})

    # 격자 안 점들로 대표 (구, 동) 추정
    points = pd.concat(
        [
            commercial[["그리드ID", "구", "법정동코드", "법정동명"]],
            lodgings_g[["그리드ID", "구", "법정동코드", "법정동명"]],
        ],
        ignore_index=True,
    )
    out = out.merge(dominant_area(points), how="left", on="그리드ID")

    # bbox 좌표를 shapely box 폴리곤으로 변환 → GeoDataFrame
    geometry = [
        box(
            row.x_min_EPSG5186,
            row.y_min_EPSG5186,
            row.x_max_EPSG5186,
            row.y_max_EPSG5186,
        )
        for row in out.itertuples(index=False)
    ]
    gdf = gpd.GeoDataFrame(out, geometry=geometry, crs="EPSG:5186").to_crs("EPSG:4326")
    # 정수형 / 소수점 정리
    for col in [
        "전체건물수",
        "상가수",
        "숙박시설수",
        "상가숙박_개수",
        "포함법정동수",
        "그리드행",
        "그리드열",
    ]:
        if col in gdf.columns:
            gdf[col] = gdf[col].fillna(0).round().astype(int)
    for col in gdf.select_dtypes(include="number").columns:
        gdf[col] = gdf[col].round(6)
    return gdf


def write_outputs(dong_gdf: gpd.GeoDataFrame, grid_gdf: gpd.GeoDataFrame) -> None:
    """CSV/GeoJSON/HTML/컬럼정의 일괄 산출 + 요약 JSON."""
    log("6. CSV/GeoJSON/HTML/컬럼정의 산출")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dong_out = dong_gdf.copy()
    # 출력 시 '법정동코드' 컬럼명을 정규화된 값으로 통일
    dong_out["법정동코드"] = dong_out["법정동코드_정규"]

    # 출력 컬럼 순서 — 발표 자료/대시보드 사용 편의
    dong_cols = [
        "구",
        "법정동코드",
        "법정동명",
        "면적_m2",
        "면적_ha",
        "전체건물수",
        "전체연면적합계_m2",
        "상가수",
        "상가연면적합계_m2",
        "상가입체화재하중_분자",
        "숙박시설수",
        "숙박연면적합계_m2",
        "숙박입체화재하중_분자",
        "복합입체화재하중_분자",
        "상가수_per_ha",
        "숙박시설수_per_ha",
        "상가연면적_per_ha",
        "숙박연면적_per_ha",
        "상가_입체화재하중밀도",
        "숙박_입체화재하중밀도",
        "복합_입체화재하중밀도",
        "상가숙박_개수",
        "상가숙박_개수_per_ha",
    ]
    grid_cols = [
        "그리드ID",
        "주요구",
        "주요법정동코드",
        "주요법정동명",
        "포함법정동수",
        "그리드행",
        "그리드열",
        "기준면적_m2",
        "기준면적_ha",
        "전체건물수",
        "상가수",
        "상가연면적합계_m2",
        "상가입체화재하중_분자",
        "숙박시설수",
        "숙박연면적합계_m2",
        "숙박입체화재하중_분자",
        "복합입체화재하중_분자",
        "상가수_per_ha",
        "숙박시설수_per_ha",
        "상가연면적_per_ha",
        "숙박연면적_per_ha",
        "상가_입체화재하중밀도",
        "숙박_입체화재하중밀도",
        "복합_입체화재하중밀도",
        "상가숙박_개수",
        "상가숙박_개수_per_ha",
        "건물데이터_0여부",
        "중심위도",
        "중심경도",
        "x_min_EPSG5186",
        "y_min_EPSG5186",
        "x_max_EPSG5186",
        "y_max_EPSG5186",
    ]

    dong_csv = OUT_DIR / "서울10개구_상가숙소_법정동별_밀집도.csv"
    grid_csv = OUT_DIR / "서울10개구_상가숙소_격자별_밀집도.csv"
    dong_geojson = OUT_DIR / "서울10개구_상가숙소_법정동별_밀집도.geojson"
    grid_geojson = OUT_DIR / "서울10개구_상가숙소_격자별_밀집도.geojson"

    # 4개 산출물 — CSV 2개 + GeoJSON 2개
    dong_out[dong_cols].to_csv(dong_csv, index=False, encoding="utf-8-sig")
    grid_gdf[grid_cols].to_csv(grid_csv, index=False, encoding="utf-8-sig")
    dong_out[dong_cols + ["geometry"]].to_file(dong_geojson, driver="GeoJSON")
    grid_gdf[grid_cols + ["geometry"]].to_file(grid_geojson, driver="GeoJSON")

    # 컬럼 정의 + HTML 지도
    write_column_dictionary(OUT_DIR / "서울10개구_상가숙소_시각화컬럼정의.csv")
    write_html_map(
        grid_geojson, dong_geojson, OUT_DIR / "서울10개구_상가숙소_밀집도지도.html"
    )

    # 요약 JSON
    summary = {
        "법정동_행수": int(len(dong_gdf)),
        "격자_행수": int(len(grid_gdf)),
        "상가수_합계": int(dong_gdf["상가수"].sum()),
        "숙박시설수_합계": int(dong_gdf["숙박시설수"].sum()),
        "출력폴더": str(OUT_DIR),
        "상가분류기준": "건물용도코드 03000/04000/07000 또는 용도명에 근린생활시설/판매시설 포함",
        "격자기준": "EPSG:5186 좌표계 50,000㎡ 정사각 격자",
    }
    (OUT_DIR / "서울10개구_상가숙소_산출요약.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log(json.dumps(summary, ensure_ascii=False, indent=2))


def write_column_dictionary(path: Path) -> None:
    """각 컬럼의 의미·계산식·시각화 용도를 정리한 데이터 사전 CSV 작성."""
    rows = [
        (
            "그리드ID",
            "격자별",
            "50,000㎡ 격자의 고유 ID",
            "EPSG:5186 좌표로 계산한 행/열",
            "지도 셀 식별",
        ),
        (
            "주요구",
            "격자별",
            "격자 안 점 데이터가 가장 많이 속한 자치구",
            "상가+숙박 포인트 최빈 구",
            "툴팁/필터",
        ),
        (
            "주요법정동코드",
            "격자별",
            "격자 안 점 데이터가 가장 많이 속한 법정동코드",
            "상가+숙박 포인트 최빈 동",
            "툴팁/필터",
        ),
        (
            "주요법정동명",
            "격자별",
            "격자 안 점 데이터가 가장 많이 속한 법정동명",
            "상가+숙박 포인트 최빈 동",
            "툴팁/필터",
        ),
        (
            "포함법정동수",
            "격자별",
            "격자 안에 포함된 포인트 기준 법정동 종류 수",
            "상가+숙박 포인트 법정동 고유 개수",
            "경계 교차 확인",
        ),
        ("기준면적_m2", "격자별", "밀집도 분모 면적", "50,000㎡ 고정", "밀도 계산"),
        (
            "면적_m2",
            "법정동별",
            "법정동 경계 면적",
            "법정동 경계 GeoJSON을 EPSG:5186으로 투영 후 면적",
            "동별 밀도 계산",
        ),
        (
            "전체건물수",
            "공통",
            "해당 구역 내 전체 건물 수",
            "AL_D010 건물 SHP",
            "데이터 공백 점검",
        ),
        (
            "상가수",
            "공통",
            "상가성 건물 수",
            "건물용도코드 03000/04000/07000 또는 용도명 키워드",
            "상가 밀집도",
        ),
        (
            "상가연면적합계_m2",
            "공통",
            "상가성 건물 연면적 합계",
            "AL_D010 연면적",
            "상가 규모",
        ),
        (
            "상가입체화재하중_분자",
            "공통",
            "상가 건물의 입체화재하중 분자",
            "Σ(건축면적 × max(지상층수+지하층수, 1)); 건축면적이 0이면 연면적 사용",
            "상가 밀도 색상",
        ),
        (
            "숙박시설수",
            "공통",
            "숙박시설 수",
            "통합숙박시설최종안0415.csv",
            "숙박 밀집도",
        ),
        (
            "숙박연면적합계_m2",
            "공통",
            "숙박시설 연면적 합계",
            "통합숙박시설최종안0415.csv 연면적(㎡)",
            "숙박 규모",
        ),
        (
            "숙박입체화재하중_분자",
            "공통",
            "숙박시설 입체화재하중 분자",
            "숙박시설 연면적(㎡) 합계",
            "숙박 밀도 색상",
        ),
        (
            "복합입체화재하중_분자",
            "공통",
            "상가+숙박 입체화재하중 분자",
            "상가입체화재하중_분자 + 숙박입체화재하중_분자",
            "복합 밀도 색상",
        ),
        ("상가수_per_ha", "공통", "ha당 상가 수", "상가수 / 면적_ha", "상가 개수 밀도"),
        (
            "숙박시설수_per_ha",
            "공통",
            "ha당 숙박시설 수",
            "숙박시설수 / 면적_ha",
            "숙박 개수 밀도",
        ),
        (
            "상가연면적_per_ha",
            "공통",
            "ha당 상가 연면적",
            "상가연면적합계_m2 / 면적_ha",
            "상가 규모 밀도",
        ),
        (
            "숙박연면적_per_ha",
            "공통",
            "ha당 숙박 연면적",
            "숙박연면적합계_m2 / 면적_ha",
            "숙박 규모 밀도",
        ),
        (
            "상가_입체화재하중밀도",
            "공통",
            "상가 입체화재하중 밀도",
            "상가입체화재하중_분자 / 기준면적_m2 또는 면적_m2",
            "상가 색상 지표",
        ),
        (
            "숙박_입체화재하중밀도",
            "공통",
            "숙박 입체화재하중 밀도",
            "숙박입체화재하중_분자 / 기준면적_m2 또는 면적_m2",
            "숙박 색상 지표",
        ),
        (
            "복합_입체화재하중밀도",
            "공통",
            "상가+숙박 복합 입체화재하중 밀도",
            "복합입체화재하중_분자 / 기준면적_m2 또는 면적_m2",
            "기본 지도 색상 지표",
        ),
        (
            "상가숙박_개수_per_ha",
            "공통",
            "ha당 상가+숙박 개수",
            "(상가수+숙박시설수) / 면적_ha",
            "개수 기반 비교",
        ),
        (
            "건물데이터_0여부",
            "격자별",
            "격자 안 전체 건물 수가 0인지 여부",
            "전체건물수 == 0",
            "데이터 공백 표시",
        ),
        (
            "중심위도",
            "격자별",
            "격자 중심 위도",
            "EPSG:5186 중심점을 WGS84로 변환",
            "지도 중심/라벨",
        ),
        (
            "중심경도",
            "격자별",
            "격자 중심 경도",
            "EPSG:5186 중심점을 WGS84로 변환",
            "지도 중심/라벨",
        ),
    ]
    pd.DataFrame(
        rows, columns=["컬럼명", "적용파일", "설명", "계산식_원천", "시각화용도"]
    ).to_csv(path, index=False, encoding="utf-8-sig")


def write_html_map(
    grid_geojson_path: Path, dong_geojson_path: Path, out_path: Path
) -> None:
    """단독 Leaflet 지도 HTML 작성 — 4개 색상 지표 토글 + 법정동 경계 오버레이."""
    grid = json.loads(grid_geojson_path.read_text(encoding="utf-8"))
    dong = json.loads(dong_geojson_path.read_text(encoding="utf-8"))
    center = [37.55, 126.98]
    # 색상 지표 후보 — 셀렉트박스로 토글
    metrics = [
        ("복합_입체화재하중밀도", "복합 밀도"),
        ("상가_입체화재하중밀도", "상가 밀도"),
        ("숙박_입체화재하중밀도", "숙박 밀도"),
        ("상가숙박_개수_per_ha", "개수/ha"),
    ]
    # f-string 안에서 CSS/JS 의 { } 는 모두 { { } } 로 이스케이프
    html = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>서울 10개구 상가·숙박 밀집도</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <style>
    html,body,#map{{height:100%;margin:0;font-family:Arial,'Malgun Gothic',sans-serif;background:#111;color:#eee}}
    #panel{{position:absolute;z-index:1000;top:16px;left:16px;background:rgba(20,22,24,.94);border:1px solid rgba(255,255,255,.14);border-radius:8px;padding:12px 14px;min-width:260px;box-shadow:0 10px 30px rgba(0,0,0,.28)}}
    #panel h1{{font-size:16px;margin:0 0 10px;font-weight:700}}
    label{{display:block;font-size:12px;color:#c8c8c8;margin-bottom:6px}}
    select{{width:100%;height:34px;border-radius:6px;border:1px solid #555;background:#191b1f;color:#fff;padding:0 8px}}
    .stats{{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:10px}}
    .stat{{background:#272a2f;border-radius:6px;padding:8px}}
    .stat b{{display:block;font-size:15px}}
    .stat span{{font-size:11px;color:#aaa}}
    .legend{{position:absolute;z-index:1000;bottom:22px;left:16px;background:rgba(20,22,24,.94);border:1px solid rgba(255,255,255,.14);border-radius:8px;padding:10px 12px;font-size:12px}}
    .bar{{width:180px;height:10px;border-radius:5px;background:linear-gradient(90deg,#1f2329,#d7eef3,#7fcdbb,#2c7fb8,#253494);margin:7px 0}}
    .note{{font-size:11px;color:#aaa;margin-top:5px;line-height:1.35}}
    .leaflet-tooltip{{font-family:Arial,'Malgun Gothic',sans-serif}}
  </style>
</head>
<body>
<div id="map"></div>
<div id="panel">
  <h1>서울 10개구 상가·숙박 밀집도</h1>
  <label for="metric">색상 지표</label>
  <select id="metric">
    {"".join(f'<option value="{value}">{label}</option>' for value, label in metrics)}
  </select>
  <div class="stats">
    <div class="stat"><b id="gridCount">-</b><span>격자</span></div>
    <div class="stat"><b id="dongCount">-</b><span>법정동</span></div>
    <div class="stat"><b id="shopCount">-</b><span>상가</span></div>
    <div class="stat"><b id="lodgingCount">-</b><span>숙박</span></div>
  </div>
</div>
<div class="legend">
  <div id="legendTitle">복합 밀도</div>
  <div class="bar"></div>
  <div><span id="legendMin">0</span><span style="float:right" id="legendMax">-</span></div>
  <div class="note">상위 2% 극단값은 같은 진한 색으로 표시</div>
</div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
// 파이썬에서 임베드한 GeoJSON 데이터 — 페이지 로드 즉시 사용
const GRID = {json.dumps(grid, ensure_ascii=False)};
const DONG = {json.dumps(dong, ensure_ascii=False)};
const METRICS = {json.dumps(dict(metrics), ensure_ascii=False)};
const map = L.map('map', {{center:{center}, zoom:11, preferCanvas:true}});
L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
  attribution:'&copy; OpenStreetMap &copy; CARTO', subdomains:'abcd', maxZoom:20
}}).addTo(map);

const fmt = n => Number(n || 0).toLocaleString('ko-KR', {{maximumFractionDigits: 4}});
function quantile(values, q) {{
  // 분위수 계산 — 상위 2% 극단값 캡 산정용 (p98)
  const sorted = values.filter(v => Number.isFinite(v) && v > 0).sort((a,b) => a-b);
  if (!sorted.length) return 0;
  const idx = Math.min(sorted.length - 1, Math.max(0, Math.floor((sorted.length - 1) * q)));
  return sorted[idx];
}}
function color(v, max) {{
  // sqrt 변환 후 5단계 색상 — 작은 값도 시각적으로 구분
  if (!v || max <= 0) return '#1f2329';
  const t = Math.sqrt(Math.min(1, v / max));
  if (t > .86) return '#253494';
  if (t > .68) return '#2c7fb8';
  if (t > .50) return '#41b6c4';
  if (t > .32) return '#7fcdbb';
  return '#d7eef3';
}}
let gridLayer;
function render(metric) {{
  // 지표 변경 시 그리드 재채색 + 범례 갱신
  if (gridLayer) map.removeLayer(gridLayer);
  const values = GRID.features.map(f => Number(f.properties[metric] || 0));
  const max = quantile(values, .98) || Math.max(...values, 0);
  gridLayer = L.geoJSON(GRID, {{
    style: f => {{
      const p = f.properties;
      return {{color:'#252525',weight:.45,fillColor:color(Number(p[metric] || 0), max),fillOpacity:.72}};
    }},
    onEachFeature: (f, layer) => {{
      const p = f.properties;
      layer.bindTooltip(
        `<b>${{p.그리드ID}}</b><br>${{p.주요구 || ''}} ${{p.주요법정동명 || ''}}<br>` +
        `상가 ${{fmt(p.상가수)}} · 숙박 ${{fmt(p.숙박시설수)}}<br>` +
        `${{METRICS[metric]}}: ${{fmt(p[metric])}}`,
        {{sticky:true}}
      );
    }}
  }}).addTo(map);
  document.getElementById('legendTitle').textContent = METRICS[metric];
  document.getElementById('legendMax').textContent = 'p98 ' + fmt(max);
}}
// 법정동 경계는 항상 흰 라인으로 오버레이
L.geoJSON(DONG, {{style:{{color:'#ffffff',weight:.7,fillOpacity:0,opacity:.35}}}}).addTo(map);
document.getElementById('metric').addEventListener('change', e => render(e.target.value));
document.getElementById('gridCount').textContent = GRID.features.length.toLocaleString('ko-KR');
document.getElementById('dongCount').textContent = DONG.features.length.toLocaleString('ko-KR');
document.getElementById('shopCount').textContent = GRID.features.reduce((s,f)=>s+Number(f.properties.상가수||0),0).toLocaleString('ko-KR');
document.getElementById('lodgingCount').textContent = GRID.features.reduce((s,f)=>s+Number(f.properties.숙박시설수||0),0).toLocaleString('ko-KR');
render('복합_입체화재하중밀도');
map.fitBounds(L.geoJSON(DONG).getBounds(), {{padding:[20,20]}});
</script>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")


def main() -> None:
    """메인 — 건물/숙박/법정동 로드 → 동·격자 집계 → CSV/GeoJSON/HTML 일괄 산출."""
    buildings = load_buildings()
    lodgings = load_lodgings()
    legal = load_legal_dongs()
    dong_gdf = aggregate_by_dong(buildings, lodgings, legal)
    grid_gdf = aggregate_by_grid(buildings, lodgings)
    write_outputs(dong_gdf, grid_gdf)


if __name__ == "__main__":
    main()
