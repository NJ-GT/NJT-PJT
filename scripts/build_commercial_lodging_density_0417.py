# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import math
from pathlib import Path

import geopandas as gpd
import pandas as pd
from pyproj import Transformer
from shapely.geometry import box


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
BUILDING_DIR = DATA_DIR / "AL_D010_11_20260409"
OUT_DIR = DATA_DIR / "상가숙소밀집도_10개구_0417"

BUILDING_SHP = BUILDING_DIR / "AL_D010_11_20260409_filtered.shp"
LODGING_CSV = DATA_DIR / "통합숙박시설최종안0415.csv"
LEGAL_DONG_GEOJSON = DATA_DIR / "[오피셜]법정동승인일자_공간정보0415.geojson"

GRID_AREA_M2 = 50_000.0
GRID_SIDE_M = math.sqrt(GRID_AREA_M2)

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

# Existing project convention from scripts/gis_analysis.py:
# 03000=제1종근린생활시설, 04000=제2종근린생활시설, 07000=판매시설.
COMMERCIAL_CODES = {"03000", "04000", "07000"}
COMMERCIAL_KEYWORDS = ("근린생활시설", "판매시설")

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
]


def log(message: str) -> None:
    print(message, flush=True)


def fix_mojibake(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    try:
        return text.encode("latin1").decode("cp949")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


def normalize_code(value: object, width: int) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    if not text or text.lower() == "nan":
        return ""
    return text.zfill(width)


def split_gu_dong(address: str) -> tuple[str, str]:
    parts = str(address).split()
    gu = parts[1] if len(parts) >= 2 else ""
    dong = parts[2] if len(parts) >= 3 else ""
    return gu, dong


def add_grid_columns(df: pd.DataFrame, x_col: str, y_col: str, origin_x: float, origin_y: float) -> pd.DataFrame:
    result = df.copy()
    result["그리드열"] = ((result[x_col] - origin_x) // GRID_SIDE_M).astype("Int64") + 1
    result["그리드행"] = ((result[y_col] - origin_y) // GRID_SIDE_M).astype("Int64") + 1
    result["그리드ID"] = result.apply(
        lambda r: f"G50K_R{int(r['그리드행']):04d}_C{int(r['그리드열']):04d}", axis=1
    )
    return result


def load_buildings() -> gpd.GeoDataFrame:
    log("1. 10개구 건물 SHP 로드 및 상가 분류")
    gdf = gpd.read_file(BUILDING_SHP)
    gdf = gdf.rename(columns={f"A{i}": name for i, name in enumerate(BUILDING_COLUMNS)})

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

    gdf["법정동코드"] = gdf["법정동코드"].map(lambda v: normalize_code(v, 10))
    gdf["시군구코드"] = gdf["법정동코드"].str[:5]
    gdf["구"] = gdf["시군구코드"].map(GU_MAP)
    gu_dong = gdf["주소"].map(split_gu_dong)
    gdf["주소_구"] = gu_dong.map(lambda x: x[0])
    gdf["법정동명"] = gu_dong.map(lambda x: x[1])
    gdf["구"] = gdf["구"].fillna(gdf["주소_구"])

    for col in ["연면적", "건축면적", "지상층수", "지하층수"]:
        gdf[col] = pd.to_numeric(gdf[col], errors="coerce").fillna(0.0)
    gdf["총층수"] = (gdf["지상층수"] + gdf["지하층수"]).clip(lower=1)
    floor_area_proxy = gdf["건축면적"] * gdf["총층수"]
    gdf["입체화재하중_분자"] = floor_area_proxy.where(floor_area_proxy > 0, gdf["연면적"])

    use_code = gdf["건물용도코드"].astype(str).str.zfill(5)
    use_name = gdf["건물용도명"].fillna("").astype(str)
    keyword_mask = use_name.apply(lambda x: any(keyword in x for keyword in COMMERCIAL_KEYWORDS))
    gdf["상가여부"] = use_code.isin(COMMERCIAL_CODES) | keyword_mask

    centroids = gdf.geometry.centroid
    gdf["x_EPSG5186"] = centroids.x
    gdf["y_EPSG5186"] = centroids.y
    to_wgs84 = Transformer.from_crs("EPSG:5186", "EPSG:4326", always_xy=True)
    lng, lat = to_wgs84.transform(gdf["x_EPSG5186"].to_numpy(), gdf["y_EPSG5186"].to_numpy())
    gdf["경도"] = lng
    gdf["위도"] = lat

    return gdf[gdf["시군구코드"].isin(GU_MAP.keys())].copy()


def load_lodgings() -> pd.DataFrame:
    log("2. 통합숙박시설 CSV 로드")
    df = pd.read_csv(LODGING_CSV, encoding="utf-8-sig")
    df["시군구코드"] = df["시군구코드"].map(lambda v: normalize_code(v, 5))
    df = df[df["시군구코드"].isin(GU_MAP.keys())].copy()
    df["구"] = df["시군구코드"].map(GU_MAP)
    df["법정동코드5"] = df["법정동코드"].map(lambda v: normalize_code(v, 5))
    df["법정동코드"] = df["시군구코드"] + df["법정동코드5"]
    df["법정동명"] = df["대지위치"].fillna("").astype(str).str.extract(r"서울특별시\s+\S+\s+([^\s]+)", expand=False).fillna("")

    for col in ["위도", "경도", "연면적(㎡)", "지상층수", "지하층수"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    df["총층수"] = (df["지상층수"] + df["지하층수"]).clip(lower=1)
    df["숙박_입체화재하중_분자"] = df["연면적(㎡)"]

    to_5186 = Transformer.from_crs("EPSG:4326", "EPSG:5186", always_xy=True)
    x, y = to_5186.transform(df["경도"].to_numpy(), df["위도"].to_numpy())
    df["x_EPSG5186"] = x
    df["y_EPSG5186"] = y
    return df


def load_legal_dongs() -> gpd.GeoDataFrame:
    log("3. 법정동 경계 로드")
    gdf = gpd.read_file(LEGAL_DONG_GEOJSON)
    gdf["법정동코드_정규"] = gdf.apply(
        lambda r: normalize_code(r.get("법정동코드"), 10)
        or (normalize_code(r.get("EMD_CD"), 8) + "00"),
        axis=1,
    )
    gdf["시군구코드"] = gdf["법정동코드_정규"].str[:5]
    gdf = gdf[gdf["시군구코드"].isin(GU_MAP.keys())].copy()
    gdf["구"] = gdf["시군구코드"].map(GU_MAP)
    gdf["법정동명"] = gdf["법정동명"].fillna(gdf["EMD_KOR_NM"])
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


def aggregate_by_dong(buildings: gpd.GeoDataFrame, lodgings: pd.DataFrame, legal: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    log("4. 법정동별 상가/숙박 밀집도 계산")
    commercial = buildings[buildings["상가여부"]].copy()
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

    out = legal.copy()
    out = out.merge(all_building_stats, how="left", left_on="법정동코드_정규", right_index=True)
    out = out.merge(commercial_stats, how="left", left_on="법정동코드_정규", right_index=True)
    out = out.merge(lodging_stats, how="left", left_on="법정동코드_정규", right_index=True)

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
    out["복합입체화재하중_분자"] = out["상가입체화재하중_분자"] + out["숙박입체화재하중_분자"]

    out["상가수_per_ha"] = out["상가수"] / out["면적_ha"]
    out["숙박시설수_per_ha"] = out["숙박시설수"] / out["면적_ha"]
    out["상가연면적_per_ha"] = out["상가연면적합계_m2"] / out["면적_ha"]
    out["숙박연면적_per_ha"] = out["숙박연면적합계_m2"] / out["면적_ha"]
    out["상가_입체화재하중밀도"] = out["상가입체화재하중_분자"] / out["면적_m2"]
    out["숙박_입체화재하중밀도"] = out["숙박입체화재하중_분자"] / out["면적_m2"]
    out["복합_입체화재하중밀도"] = out["복합입체화재하중_분자"] / out["면적_m2"]
    out["상가숙박_개수"] = out["상가수"] + out["숙박시설수"]
    out["상가숙박_개수_per_ha"] = out["상가숙박_개수"] / out["면적_ha"]

    for col in ["전체건물수", "상가수", "숙박시설수", "상가숙박_개수"]:
        out[col] = out[col].round().astype(int)
    for col in out.select_dtypes(include="number").columns:
        out[col] = out[col].round(6)
    return out


def dominant_area(points: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        points.groupby(["그리드ID", "구", "법정동코드", "법정동명"], dropna=False)
        .size()
        .reset_index(name="point_count")
        .sort_values(["그리드ID", "point_count"], ascending=[True, False])
    )
    primary = grouped.drop_duplicates("그리드ID").rename(
        columns={"구": "주요구", "법정동코드": "주요법정동코드", "법정동명": "주요법정동명"}
    )
    dong_count = grouped.groupby("그리드ID")["법정동코드"].nunique().rename("포함법정동수")
    return primary.merge(dong_count, on="그리드ID", how="left")


def aggregate_by_grid(buildings: gpd.GeoDataFrame, lodgings: pd.DataFrame) -> gpd.GeoDataFrame:
    log("5. 50,000㎡ 격자별 상가/숙박 밀집도 계산")
    all_x = pd.concat([buildings["x_EPSG5186"], lodgings["x_EPSG5186"]], ignore_index=True)
    all_y = pd.concat([buildings["y_EPSG5186"], lodgings["y_EPSG5186"]], ignore_index=True)
    origin_x = math.floor(all_x.min() / GRID_SIDE_M) * GRID_SIDE_M
    origin_y = math.floor(all_y.min() / GRID_SIDE_M) * GRID_SIDE_M

    buildings_g = add_grid_columns(buildings, "x_EPSG5186", "y_EPSG5186", origin_x, origin_y)
    lodgings_g = add_grid_columns(lodgings, "x_EPSG5186", "y_EPSG5186", origin_x, origin_y)

    commercial = buildings_g[buildings_g["상가여부"]].copy()
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

    grid_keys = pd.Index(commercial_stats.index).union(lodging_stats.index)
    out = pd.DataFrame(index=grid_keys)
    out.index.name = "그리드ID"
    out = out.merge(all_building_stats, how="left", left_index=True, right_index=True)
    out = out.merge(commercial_stats, how="left", left_index=True, right_index=True)
    out = out.merge(lodging_stats, how="left", left_index=True, right_index=True)
    out = out.fillna(0.0).reset_index()

    row_col = out["그리드ID"].str.extract(r"G50K_R(\d+)_C(\d+)").astype(int)
    out["그리드행"] = row_col[0]
    out["그리드열"] = row_col[1]
    out["x_min_EPSG5186"] = origin_x + (out["그리드열"] - 1) * GRID_SIDE_M
    out["y_min_EPSG5186"] = origin_y + (out["그리드행"] - 1) * GRID_SIDE_M
    out["x_max_EPSG5186"] = out["x_min_EPSG5186"] + GRID_SIDE_M
    out["y_max_EPSG5186"] = out["y_min_EPSG5186"] + GRID_SIDE_M
    out["중심x_EPSG5186"] = (out["x_min_EPSG5186"] + out["x_max_EPSG5186"]) / 2
    out["중심y_EPSG5186"] = (out["y_min_EPSG5186"] + out["y_max_EPSG5186"]) / 2

    to_wgs84 = Transformer.from_crs("EPSG:5186", "EPSG:4326", always_xy=True)
    lng, lat = to_wgs84.transform(out["중심x_EPSG5186"].to_numpy(), out["중심y_EPSG5186"].to_numpy())
    out["중심경도"] = lng
    out["중심위도"] = lat

    out["기준면적_m2"] = GRID_AREA_M2
    out["기준면적_ha"] = GRID_AREA_M2 / 10_000.0
    out["복합입체화재하중_분자"] = out["상가입체화재하중_분자"] + out["숙박입체화재하중_분자"]
    out["상가수_per_ha"] = out["상가수"] / out["기준면적_ha"]
    out["숙박시설수_per_ha"] = out["숙박시설수"] / out["기준면적_ha"]
    out["상가연면적_per_ha"] = out["상가연면적합계_m2"] / out["기준면적_ha"]
    out["숙박연면적_per_ha"] = out["숙박연면적합계_m2"] / out["기준면적_ha"]
    out["상가_입체화재하중밀도"] = out["상가입체화재하중_분자"] / GRID_AREA_M2
    out["숙박_입체화재하중밀도"] = out["숙박입체화재하중_분자"] / GRID_AREA_M2
    out["복합_입체화재하중밀도"] = out["복합입체화재하중_분자"] / GRID_AREA_M2
    out["상가숙박_개수"] = out["상가수"] + out["숙박시설수"]
    out["상가숙박_개수_per_ha"] = out["상가숙박_개수"] / out["기준면적_ha"]
    out["건물데이터_0여부"] = out["전체건물수"].eq(0).map({True: "Y", False: "N"})

    points = pd.concat(
        [
            commercial[["그리드ID", "구", "법정동코드", "법정동명"]],
            lodgings_g[["그리드ID", "구", "법정동코드", "법정동명"]],
        ],
        ignore_index=True,
    )
    out = out.merge(dominant_area(points), how="left", on="그리드ID")

    geometry = [
        box(row.x_min_EPSG5186, row.y_min_EPSG5186, row.x_max_EPSG5186, row.y_max_EPSG5186)
        for row in out.itertuples(index=False)
    ]
    gdf = gpd.GeoDataFrame(out, geometry=geometry, crs="EPSG:5186").to_crs("EPSG:4326")
    for col in ["전체건물수", "상가수", "숙박시설수", "상가숙박_개수", "포함법정동수", "그리드행", "그리드열"]:
        if col in gdf.columns:
            gdf[col] = gdf[col].fillna(0).round().astype(int)
    for col in gdf.select_dtypes(include="number").columns:
        gdf[col] = gdf[col].round(6)
    return gdf


def write_outputs(dong_gdf: gpd.GeoDataFrame, grid_gdf: gpd.GeoDataFrame) -> None:
    log("6. CSV/GeoJSON/HTML/컬럼정의 산출")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dong_out = dong_gdf.copy()
    dong_out["법정동코드"] = dong_out["법정동코드_정규"]

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

    dong_out[dong_cols].to_csv(dong_csv, index=False, encoding="utf-8-sig")
    grid_gdf[grid_cols].to_csv(grid_csv, index=False, encoding="utf-8-sig")
    dong_out[dong_cols + ["geometry"]].to_file(dong_geojson, driver="GeoJSON")
    grid_gdf[grid_cols + ["geometry"]].to_file(grid_geojson, driver="GeoJSON")

    write_column_dictionary(OUT_DIR / "서울10개구_상가숙소_시각화컬럼정의.csv")
    write_html_map(grid_geojson, dong_geojson, OUT_DIR / "서울10개구_상가숙소_밀집도지도.html")

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
    rows = [
        ("그리드ID", "격자별", "50,000㎡ 격자의 고유 ID", "EPSG:5186 좌표로 계산한 행/열", "지도 셀 식별"),
        ("주요구", "격자별", "격자 안 점 데이터가 가장 많이 속한 자치구", "상가+숙박 포인트 최빈 구", "툴팁/필터"),
        ("주요법정동코드", "격자별", "격자 안 점 데이터가 가장 많이 속한 법정동코드", "상가+숙박 포인트 최빈 동", "툴팁/필터"),
        ("주요법정동명", "격자별", "격자 안 점 데이터가 가장 많이 속한 법정동명", "상가+숙박 포인트 최빈 동", "툴팁/필터"),
        ("포함법정동수", "격자별", "격자 안에 포함된 포인트 기준 법정동 종류 수", "상가+숙박 포인트 법정동 고유 개수", "경계 교차 확인"),
        ("기준면적_m2", "격자별", "밀집도 분모 면적", "50,000㎡ 고정", "밀도 계산"),
        ("면적_m2", "법정동별", "법정동 경계 면적", "법정동 경계 GeoJSON을 EPSG:5186으로 투영 후 면적", "동별 밀도 계산"),
        ("전체건물수", "공통", "해당 구역 내 전체 건물 수", "AL_D010 건물 SHP", "데이터 공백 점검"),
        ("상가수", "공통", "상가성 건물 수", "건물용도코드 03000/04000/07000 또는 용도명 키워드", "상가 밀집도"),
        ("상가연면적합계_m2", "공통", "상가성 건물 연면적 합계", "AL_D010 연면적", "상가 규모"),
        ("상가입체화재하중_분자", "공통", "상가 건물의 입체화재하중 분자", "Σ(건축면적 × max(지상층수+지하층수, 1)); 건축면적이 0이면 연면적 사용", "상가 밀도 색상"),
        ("숙박시설수", "공통", "숙박시설 수", "통합숙박시설최종안0415.csv", "숙박 밀집도"),
        ("숙박연면적합계_m2", "공통", "숙박시설 연면적 합계", "통합숙박시설최종안0415.csv 연면적(㎡)", "숙박 규모"),
        ("숙박입체화재하중_분자", "공통", "숙박시설 입체화재하중 분자", "숙박시설 연면적(㎡) 합계", "숙박 밀도 색상"),
        ("복합입체화재하중_분자", "공통", "상가+숙박 입체화재하중 분자", "상가입체화재하중_분자 + 숙박입체화재하중_분자", "복합 밀도 색상"),
        ("상가수_per_ha", "공통", "ha당 상가 수", "상가수 / 면적_ha", "상가 개수 밀도"),
        ("숙박시설수_per_ha", "공통", "ha당 숙박시설 수", "숙박시설수 / 면적_ha", "숙박 개수 밀도"),
        ("상가연면적_per_ha", "공통", "ha당 상가 연면적", "상가연면적합계_m2 / 면적_ha", "상가 규모 밀도"),
        ("숙박연면적_per_ha", "공통", "ha당 숙박 연면적", "숙박연면적합계_m2 / 면적_ha", "숙박 규모 밀도"),
        ("상가_입체화재하중밀도", "공통", "상가 입체화재하중 밀도", "상가입체화재하중_분자 / 기준면적_m2 또는 면적_m2", "상가 색상 지표"),
        ("숙박_입체화재하중밀도", "공통", "숙박 입체화재하중 밀도", "숙박입체화재하중_분자 / 기준면적_m2 또는 면적_m2", "숙박 색상 지표"),
        ("복합_입체화재하중밀도", "공통", "상가+숙박 복합 입체화재하중 밀도", "복합입체화재하중_분자 / 기준면적_m2 또는 면적_m2", "기본 지도 색상 지표"),
        ("상가숙박_개수_per_ha", "공통", "ha당 상가+숙박 개수", "(상가수+숙박시설수) / 면적_ha", "개수 기반 비교"),
        ("건물데이터_0여부", "격자별", "격자 안 전체 건물 수가 0인지 여부", "전체건물수 == 0", "데이터 공백 표시"),
        ("중심위도", "격자별", "격자 중심 위도", "EPSG:5186 중심점을 WGS84로 변환", "지도 중심/라벨"),
        ("중심경도", "격자별", "격자 중심 경도", "EPSG:5186 중심점을 WGS84로 변환", "지도 중심/라벨"),
    ]
    pd.DataFrame(rows, columns=["컬럼명", "적용파일", "설명", "계산식_원천", "시각화용도"]).to_csv(
        path, index=False, encoding="utf-8-sig"
    )


def write_html_map(grid_geojson_path: Path, dong_geojson_path: Path, out_path: Path) -> None:
    grid = json.loads(grid_geojson_path.read_text(encoding="utf-8"))
    dong = json.loads(dong_geojson_path.read_text(encoding="utf-8"))
    center = [37.55, 126.98]
    metrics = [
        ("복합_입체화재하중밀도", "복합 밀도"),
        ("상가_입체화재하중밀도", "상가 밀도"),
        ("숙박_입체화재하중밀도", "숙박 밀도"),
        ("상가숙박_개수_per_ha", "개수/ha"),
    ]
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
    {''.join(f'<option value="{value}">{label}</option>' for value, label in metrics)}
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
const GRID = {json.dumps(grid, ensure_ascii=False)};
const DONG = {json.dumps(dong, ensure_ascii=False)};
const METRICS = {json.dumps(dict(metrics), ensure_ascii=False)};
const map = L.map('map', {{center:{center}, zoom:11, preferCanvas:true}});
L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
  attribution:'&copy; OpenStreetMap &copy; CARTO', subdomains:'abcd', maxZoom:20
}}).addTo(map);

const fmt = n => Number(n || 0).toLocaleString('ko-KR', {{maximumFractionDigits: 4}});
function quantile(values, q) {{
  const sorted = values.filter(v => Number.isFinite(v) && v > 0).sort((a,b) => a-b);
  if (!sorted.length) return 0;
  const idx = Math.min(sorted.length - 1, Math.max(0, Math.floor((sorted.length - 1) * q)));
  return sorted[idx];
}}
function color(v, max) {{
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
    buildings = load_buildings()
    lodgings = load_lodgings()
    legal = load_legal_dongs()
    dong_gdf = aggregate_by_dong(buildings, lodgings, legal)
    grid_gdf = aggregate_by_grid(buildings, lodgings)
    write_outputs(dong_gdf, grid_gdf)


if __name__ == "__main__":
    main()
