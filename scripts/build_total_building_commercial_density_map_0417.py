# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import geopandas as gpd
import pandas as pd
from pyproj import Transformer


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
BUILDING_DIR = DATA_DIR / "AL_D010_11_20260409"
GRID_SOURCE_DIR = DATA_DIR / "상가숙소밀집도_10개구_0417"
OUT_DIR = DATA_DIR / "건물상가밀집도_10개구_0417"

BUILDING_SHP = BUILDING_DIR / "AL_D010_11_20260409_filtered.shp"
BASE_GRID_GEOJSON = GRID_SOURCE_DIR / "서울10개구_상가숙소_격자별_밀집도.geojson"

OUT_HTML = OUT_DIR / "서울10개구_건물상가_50,000sqm_밀도지도.html"
OUT_BUILDING_POINTS_JS = OUT_DIR / "actual_buildings_10gu_0417.js"
OUT_CSV = OUT_DIR / "서울10개구_건물상가_격자별_시각화.csv"
OUT_GEOJSON = OUT_DIR / "서울10개구_건물상가_격자별_밀도.geojson"
OUT_COLUMNS = OUT_DIR / "서울10개구_건물상가_시각화컬럼정의.csv"
OUT_SUMMARY = OUT_DIR / "서울10개구_건물상가_산출요약.json"
ZERO_DENSITY_EXCLUDE_CSV = OUT_DIR / "서울10개구_밀도0_제외건물목록_388.csv"
REGISTRY_CORRECTION_CSV = OUT_DIR / "서울10개구_건물있지만_밀도0_표제부보정후보_건물목록.csv"

GRID_AREA_M2 = 50_000.0
GRID_AREA_HA = GRID_AREA_M2 / 10_000.0
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

COMMERCIAL_CODES = {"03000", "04000", "07000"}
COMMERCIAL_KEYWORDS = ("근린생활시설", "판매시설")

BUILDING_COLUMNS = {
    "A0": "건물유형코드",
    "A1": "건물ID",
    "A2": "건물관리번호",
    "A3": "법정동코드",
    "A4": "주소",
    "A5": "지번",
    "A8": "건물용도코드",
    "A9": "건물용도명",
    "A14": "연면적",
    "A15": "건축면적",
    "A23": "행정구역코드",
    "A26": "지상층수",
    "A27": "지하층수",
    "A29": "총층수",
}


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


def to_float(value: object) -> float:
    try:
        if value is None or value == "":
            return 0.0
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return 0.0
        return number
    except (TypeError, ValueError):
        return 0.0


def quantile(values: list[float], q: float) -> float:
    clean = sorted(v for v in values if v > 0)
    if not clean:
        return 1.0
    pos = (len(clean) - 1) * q
    low = math.floor(pos)
    high = math.ceil(pos)
    if low == high:
        return clean[int(pos)]
    return clean[low] + (clean[high] - clean[low]) * (pos - low)


def load_zero_density_exclude_keys() -> set[tuple[str, str]]:
    if not ZERO_DENSITY_EXCLUDE_CSV.exists():
        return set()
    df = pd.read_csv(ZERO_DENSITY_EXCLUDE_CSV, encoding="utf-8-sig", dtype={"건물ID": str})
    if "그리드ID" not in df.columns or "건물ID" not in df.columns:
        return set()
    return set(zip(df["그리드ID"].astype(str), df["건물ID"].astype(str)))


def load_registry_corrections() -> pd.DataFrame:
    if not REGISTRY_CORRECTION_CSV.exists():
        return pd.DataFrame()

    dtype = {
        "그리드ID": str,
        "건물ID": str,
        "표제부선택후보_관리건축물대장PK": str,
        "표제부선택후보_주용도코드": str,
    }
    df = pd.read_csv(REGISTRY_CORRECTION_CSV, encoding="utf-8-sig", dtype=dtype)
    required = {
        "그리드ID",
        "건물ID",
        "표제부선택후보_건물명",
        "표제부선택후보_주용도코드",
        "표제부선택후보_주용도코드명",
        "표제부선택후보_건축면적_m2",
        "표제부선택후보_연면적_m2",
        "표제부선택후보_지상층수",
        "표제부선택후보_지하층수",
        "표제부선택후보_관리건축물대장PK",
        "표제부선택후보_표제부_보정분자_후보",
    }
    if not required.issubset(df.columns):
        return pd.DataFrame()

    df["_보정분자"] = pd.to_numeric(df["표제부선택후보_표제부_보정분자_후보"], errors="coerce").fillna(0.0)
    df = df[df["_보정분자"] > 0].copy()
    if df.empty:
        return df

    df = df.drop_duplicates(["그리드ID", "건물ID"], keep="first")
    return df[
        [
            "그리드ID",
            "건물ID",
            "표제부선택후보_건물명",
            "표제부선택후보_주용도코드",
            "표제부선택후보_주용도코드명",
            "표제부선택후보_건축면적_m2",
            "표제부선택후보_연면적_m2",
            "표제부선택후보_지상층수",
            "표제부선택후보_지하층수",
            "표제부선택후보_관리건축물대장PK",
        ]
    ].copy()


def apply_registry_corrections(gdf: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, int]:
    gdf["건물명"] = ""
    gdf["관리건축물대장PK"] = ""
    gdf["표제부보정여부"] = False

    corrections = load_registry_corrections()
    if corrections.empty:
        return gdf, 0

    corrected = gdf.merge(corrections, on=["그리드ID", "건물ID"], how="left")
    mask = corrected["표제부선택후보_연면적_m2"].notna()
    if not mask.any():
        return corrected, 0

    numeric_pairs = {
        "연면적": "표제부선택후보_연면적_m2",
        "건축면적": "표제부선택후보_건축면적_m2",
        "지상층수": "표제부선택후보_지상층수",
        "지하층수": "표제부선택후보_지하층수",
    }
    for target, source in numeric_pairs.items():
        corrected.loc[mask, target] = pd.to_numeric(corrected.loc[mask, source], errors="coerce").fillna(0.0)

    corrected.loc[mask, "건물용도코드"] = corrected.loc[mask, "표제부선택후보_주용도코드"].map(
        lambda value: normalize_code(value, 5)
    )
    corrected.loc[mask, "건물용도명"] = corrected.loc[mask, "표제부선택후보_주용도코드명"].fillna("").astype(str)
    corrected.loc[mask, "건물명"] = corrected.loc[mask, "표제부선택후보_건물명"].fillna("").astype(str)
    corrected.loc[mask, "관리건축물대장PK"] = (
        corrected.loc[mask, "표제부선택후보_관리건축물대장PK"].fillna("").astype(str)
    )
    corrected.loc[mask, "표제부보정여부"] = True

    helper_cols = [col for col in corrections.columns if col not in {"그리드ID", "건물ID"}]
    corrected = corrected.drop(columns=helper_cols)
    return corrected, int(mask.sum())


def calculate_building_metrics(gdf: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, int]:
    gdf["건축면적있음"] = gdf["건축면적"] > 0
    gdf["연면적있음"] = gdf["연면적"] > 0
    gdf["층수있음"] = (gdf["지상층수"] + gdf["지하층수"]) > 0
    gdf["총층수"] = (gdf["지상층수"] + gdf["지하층수"]).clip(lower=1)
    gdf["건축면적추정여부"] = (~gdf["건축면적있음"]) & gdf["연면적있음"] & gdf["층수있음"]
    gdf["계산건축면적"] = gdf["건축면적"].where(
        gdf["건축면적있음"],
        (gdf["연면적"] / gdf["총층수"]).where(gdf["건축면적추정여부"], 0.0),
    )
    gdf["건축면적산정가능여부"] = gdf["계산건축면적"] > 0
    floor_proxy = gdf["건축면적"] * gdf["총층수"]
    estimated_floor_proxy = gdf["계산건축면적"] * gdf["총층수"]
    gdf["건축면적x층수_원값"] = estimated_floor_proxy.clip(lower=0.0)
    capped_floor_proxy = gdf["건축면적x층수_원값"].where(
        ~((gdf["연면적"] > 0) & (gdf["건축면적x층수_원값"] > gdf["연면적"])),
        gdf["연면적"],
    )
    gdf["건축면적x층수_분자"] = capped_floor_proxy.where(gdf["건축면적산정가능여부"], 0.0)
    use_floor_area = (
        (gdf["연면적"] > 0)
        & (
            (gdf["건축면적"] <= 0)
            | (gdf["건축면적"] > gdf["연면적"])
            | (floor_proxy > gdf["연면적"])
        )
    )
    gdf["면적보정여부"] = use_floor_area
    gdf["전체건물입체화재하중_분자_원본"] = floor_proxy.where(floor_proxy > 0, gdf["연면적"])
    gdf["전체건물입체화재하중_분자"] = floor_proxy.where(~use_floor_area, gdf["연면적"])
    gdf["전체건물입체화재하중_분자"] = gdf["전체건물입체화재하중_분자"].where(
        gdf["전체건물입체화재하중_분자"] > 0,
        gdf["연면적"],
    )
    gdf["면적계산가능여부"] = gdf["전체건물입체화재하중_분자"] > 0

    use_name = gdf["건물용도명"].fillna("").astype(str)
    keyword_mask = use_name.apply(lambda text: any(keyword in text for keyword in COMMERCIAL_KEYWORDS))
    gdf["상가여부"] = gdf["건물용도코드"].isin(COMMERCIAL_CODES) | keyword_mask
    gdf["상가입체화재하중_분자"] = gdf["전체건물입체화재하중_분자"].where(gdf["상가여부"], 0.0)
    gdf["상가건축면적x층수_분자"] = gdf["건축면적x층수_분자"].where(gdf["상가여부"], 0.0)
    gdf["상가연면적"] = gdf["연면적"].where(gdf["상가여부"], 0.0)
    gdf["상가수_flag"] = gdf["상가여부"].astype(int)

    zero_numerator_building_count = int((~gdf["면적계산가능여부"]).sum())
    return gdf, zero_numerator_building_count


def load_base_grid() -> tuple[dict, pd.DataFrame, float, float]:
    with BASE_GRID_GEOJSON.open("r", encoding="utf-8") as f:
        geojson = json.load(f)

    rows = []
    for feature in geojson["features"]:
        props = feature["properties"]
        rows.append(
            {
                "그리드ID": props["그리드ID"],
                "주요구": props.get("주요구", ""),
                "주요법정동코드": props.get("주요법정동코드", ""),
                "주요법정동명": props.get("주요법정동명", ""),
                "포함법정동수": to_float(props.get("포함법정동수", 0)),
                "그리드행": int(to_float(props["그리드행"])),
                "그리드열": int(to_float(props["그리드열"])),
                "기준면적_m2": GRID_AREA_M2,
                "기준면적_ha": GRID_AREA_HA,
                "숙박시설수": to_float(props.get("숙박시설수", 0)),
                "숙박연면적합계_m2": to_float(props.get("숙박연면적합계_m2", 0)),
                "숙박입체화재하중_분자": to_float(props.get("숙박입체화재하중_분자", 0)),
                "중심위도": to_float(props.get("중심위도", 0)),
                "중심경도": to_float(props.get("중심경도", 0)),
                "x_min_EPSG5186": to_float(props["x_min_EPSG5186"]),
                "y_min_EPSG5186": to_float(props["y_min_EPSG5186"]),
                "x_max_EPSG5186": to_float(props["x_max_EPSG5186"]),
                "y_max_EPSG5186": to_float(props["y_max_EPSG5186"]),
            }
        )

    grid_df = pd.DataFrame(rows)
    origin_x = (grid_df["x_min_EPSG5186"] - (grid_df["그리드열"] - 1) * GRID_SIDE_M).median()
    origin_y = (grid_df["y_min_EPSG5186"] - (grid_df["그리드행"] - 1) * GRID_SIDE_M).median()
    return geojson, grid_df, float(origin_x), float(origin_y)


def build_building_points(gdf: gpd.GeoDataFrame) -> list[list[object]]:
    print("2. 실제 건물 위치 레이어 데이터 생성", flush=True)
    to_wgs84 = Transformer.from_crs("EPSG:5186", "EPSG:4326", always_xy=True)
    lng_values, lat_values = to_wgs84.transform(gdf["x_EPSG5186"].to_numpy(), gdf["y_EPSG5186"].to_numpy())

    points = []
    rows = zip(
        lat_values,
        lng_values,
        gdf["상가여부"],
        gdf["연면적"],
        gdf["총층수"],
        gdf["구"],
        gdf["법정동명"],
        gdf["건물용도명"],
        gdf["그리드ID"],
        gdf["건물ID"].astype(str),
        gdf["건축면적"],
        gdf["계산건축면적"],
        gdf["건축면적추정여부"],
        gdf["건축면적x층수_원값"],
        gdf["건축면적x층수_분자"],
        gdf["건물명"],
        gdf["표제부보정여부"],
        gdf["관리건축물대장PK"].astype(str),
    )
    for (
        lat,
        lng,
        is_commercial,
        area,
        floors,
        gu,
        dong,
        use_name,
        grid_id,
        building_id,
        building_area,
        calc_building_area,
        is_estimated,
        footprint_floor_raw,
        footprint_floor,
        building_name,
        is_registry_corrected,
        registry_pk,
    ) in rows:
        if not (math.isfinite(float(lat)) and math.isfinite(float(lng))):
            continue
        points.append(
            [
                round(float(lat), 6),
                round(float(lng), 6),
                1 if bool(is_commercial) else 0,
                round(to_float(area), 1),
                int(to_float(floors)),
                str(gu or ""),
                str(dong or ""),
                str(use_name or ""),
                str(grid_id or ""),
                building_id,
                round(to_float(building_area), 1),
                round(to_float(calc_building_area), 1),
                1 if bool(is_estimated) else 0,
                round(to_float(footprint_floor_raw), 1),
                round(to_float(footprint_floor), 1),
                str(building_name or ""),
                1 if bool(is_registry_corrected) else 0,
                str(registry_pk or ""),
            ]
        )
    return points


def load_and_aggregate_buildings(grid_ids: set[str], origin_x: float, origin_y: float) -> tuple[pd.DataFrame, dict, list[list[object]]]:
    print("1. 10개구 건물 원본 로드 및 전체/상가 분자 계산", flush=True)
    gdf = gpd.read_file(BUILDING_SHP)
    gdf = gdf.rename(columns=BUILDING_COLUMNS)

    gdf["법정동코드"] = gdf["법정동코드"].map(lambda value: normalize_code(value, 10))
    gdf["구코드"] = gdf["법정동코드"].str[:5]
    gdf = gdf[gdf["구코드"].isin(GU_MAP)].copy()
    source_building_count = int(len(gdf))

    gdf["구"] = gdf["구코드"].map(GU_MAP)
    gdf["주소"] = gdf["주소"].map(fix_mojibake)
    gdf["법정동명"] = gdf["주소"].str.split().str[2].fillna("")
    gdf["건물용도코드"] = gdf["건물용도코드"].map(lambda value: normalize_code(value, 5))
    gdf["건물용도명"] = gdf["건물용도명"].map(fix_mojibake)

    for col in ["연면적", "건축면적", "지상층수", "지하층수"]:
        gdf[col] = pd.to_numeric(gdf[col], errors="coerce").fillna(0.0)
    gdf.loc[gdf["지상층수"] == 0, "지상층수"] = 1.0

    centroids = gdf.geometry.centroid
    gdf["x_EPSG5186"] = centroids.x
    gdf["y_EPSG5186"] = centroids.y
    gdf["그리드열"] = ((gdf["x_EPSG5186"] - origin_x) // GRID_SIDE_M).astype("Int64") + 1
    gdf["그리드행"] = ((gdf["y_EPSG5186"] - origin_y) // GRID_SIDE_M).astype("Int64") + 1
    gdf["그리드ID"] = gdf.apply(
        lambda row: f"G50K_R{int(row['그리드행']):04d}_C{int(row['그리드열']):04d}",
        axis=1,
    )

    unmatched = int((~gdf["그리드ID"].isin(grid_ids)).sum())
    gdf = gdf[gdf["그리드ID"].isin(grid_ids)].copy()
    gdf, registry_correction_building_count = apply_registry_corrections(gdf)
    exclude_keys = load_zero_density_exclude_keys()
    if exclude_keys:
        exclude_mask = [
            ((grid_id, building_id) in exclude_keys) and (not is_corrected)
            for grid_id, building_id, is_corrected in zip(
                gdf["그리드ID"].astype(str),
                gdf["건물ID"].astype(str),
                gdf["표제부보정여부"],
            )
        ]
        excluded_zero_density_building_count = int(sum(exclude_mask))
        gdf = gdf.loc[[not value for value in exclude_mask]].copy()
    else:
        excluded_zero_density_building_count = 0

    gdf, zero_numerator_building_count = calculate_building_metrics(gdf)

    building_points = build_building_points(gdf)

    agg = (
        gdf.groupby("그리드ID")
        .agg(
            전체건물수=("건물ID", "count"),
            건축면적입력건물수=("건축면적있음", "sum"),
            건축면적누락건물수=("건축면적있음", lambda values: int((~values).sum())),
            건축면적역추산건물수=("건축면적추정여부", "sum"),
            건축면적산정불가건물수=("건축면적산정가능여부", lambda values: int((~values).sum())),
            연면적입력건물수=("연면적있음", "sum"),
            층수입력건물수=("층수있음", "sum"),
            전체건물연면적합계_m2=("연면적", "sum"),
            전체건축면적합계_m2=("건축면적", "sum"),
            계산건축면적합계_m2=("계산건축면적", "sum"),
            건축면적x층수_분자=("건축면적x층수_분자", "sum"),
            전체건물입체화재하중_분자_원본=("전체건물입체화재하중_분자_원본", "sum"),
            전체건물입체화재하중_분자=("전체건물입체화재하중_분자", "sum"),
            면적계산가능건물수=("면적계산가능여부", "sum"),
            면적보정건물수=("면적보정여부", "sum"),
            표제부보정건물수=("표제부보정여부", "sum"),
            상가수=("상가수_flag", "sum"),
            상가연면적합계_m2=("상가연면적", "sum"),
            상가건축면적x층수_분자=("상가건축면적x층수_분자", "sum"),
            상가입체화재하중_분자=("상가입체화재하중_분자", "sum"),
        )
        .reset_index()
    )

    summary = {
        "원본_10개구_건물수": source_building_count,
        "격자_미매칭_건물수": unmatched,
        "면적분자0_전체후보건물수": zero_numerator_building_count,
        "밀도0_지정제외건물수": excluded_zero_density_building_count,
        "표제부보정건물수": registry_correction_building_count,
        "집계_건물수": int(agg["전체건물수"].sum()),
        "집계_상가수": int(agg["상가수"].sum()),
        "면적보정건물수": int(agg["면적보정건물수"].sum()),
        "표제부보정집계건물수": int(agg["표제부보정건물수"].sum()),
        "실제건물표시_건물수": len(building_points),
    }
    return agg, summary, building_points


def build_grid_dataset(base_geojson: dict, base_df: pd.DataFrame, agg: pd.DataFrame) -> tuple[dict, pd.DataFrame, dict]:
    print("2. 50,000㎡ 격자에 전체건물/상가 지표 결합", flush=True)
    out = base_df.merge(agg, on="그리드ID", how="left")
    numeric_fill = [
        "전체건물수",
        "건축면적입력건물수",
        "건축면적누락건물수",
        "건축면적역추산건물수",
        "건축면적산정불가건물수",
        "연면적입력건물수",
        "층수입력건물수",
        "전체건물연면적합계_m2",
        "전체건축면적합계_m2",
        "계산건축면적합계_m2",
        "건축면적x층수_분자",
        "전체건물입체화재하중_분자_원본",
        "전체건물입체화재하중_분자",
        "면적계산가능건물수",
        "면적보정건물수",
        "표제부보정건물수",
        "숙박시설수",
        "숙박연면적합계_m2",
        "숙박입체화재하중_분자",
        "상가수",
        "상가연면적합계_m2",
        "상가건축면적x층수_분자",
        "상가입체화재하중_분자",
    ]
    for col in numeric_fill:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    out["전체건물수_per_ha"] = out["전체건물수"] / GRID_AREA_HA
    out["상가수_per_ha"] = out["상가수"] / GRID_AREA_HA
    out["숙박시설수_per_ha"] = out["숙박시설수"] / GRID_AREA_HA
    out["상가숙박_개수"] = out["상가수"] + out["숙박시설수"]
    out["상가숙박_개수_per_ha"] = out["상가숙박_개수"] / GRID_AREA_HA
    out["상가숙박_면적층수_분자"] = out["상가건축면적x층수_분자"] + out["숙박입체화재하중_분자"]
    out["상가숙박_면적층수_밀도"] = out["상가숙박_면적층수_분자"] / GRID_AREA_M2
    out["건축면적x층수_밀도"] = out["건축면적x층수_분자"] / GRID_AREA_M2
    out["전체건물_입체화재하중밀도"] = out["전체건물입체화재하중_분자"] / GRID_AREA_M2
    out["상가건축면적x층수_밀도"] = out["상가건축면적x층수_분자"] / GRID_AREA_M2
    out["상가_입체화재하중밀도"] = out["상가입체화재하중_분자"] / GRID_AREA_M2
    out["건물데이터_0여부"] = out["전체건물수"].map(lambda count: "Y" if count <= 0 else "N")
    out["면적데이터_0여부"] = out.apply(
        lambda row: "Y" if row["전체건물수"] > 0 and row["전체건물입체화재하중_분자"] <= 0 else "N",
        axis=1,
    )

    round_cols = [
        "전체건물연면적합계_m2",
        "전체건축면적합계_m2",
        "계산건축면적합계_m2",
        "건축면적x층수_분자",
        "전체건물입체화재하중_분자",
        "상가연면적합계_m2",
        "숙박연면적합계_m2",
        "상가건축면적x층수_분자",
        "숙박입체화재하중_분자",
        "상가숙박_면적층수_분자",
        "상가입체화재하중_분자",
        "전체건물수_per_ha",
        "상가수_per_ha",
        "숙박시설수_per_ha",
        "상가숙박_개수_per_ha",
        "상가숙박_면적층수_밀도",
        "건축면적x층수_밀도",
        "전체건물_입체화재하중밀도",
        "상가건축면적x층수_밀도",
        "상가_입체화재하중밀도",
    ]
    for col in round_cols:
        out[col] = out[col].round(6)

    features_by_id = {feature["properties"]["그리드ID"]: feature for feature in base_geojson["features"]}
    output_features = []
    output_columns = [
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
        "건축면적입력건물수",
        "건축면적누락건물수",
        "건축면적역추산건물수",
        "건축면적산정불가건물수",
        "연면적입력건물수",
        "층수입력건물수",
        "전체건물수_per_ha",
        "전체건물연면적합계_m2",
        "전체건축면적합계_m2",
        "계산건축면적합계_m2",
        "건축면적x층수_분자",
        "건축면적x층수_밀도",
        "전체건물입체화재하중_분자_원본",
        "전체건물입체화재하중_분자",
        "면적계산가능건물수",
        "면적보정건물수",
        "표제부보정건물수",
        "전체건물_입체화재하중밀도",
        "상가수",
        "상가수_per_ha",
        "숙박시설수",
        "숙박시설수_per_ha",
        "상가숙박_개수",
        "상가숙박_개수_per_ha",
        "상가연면적합계_m2",
        "숙박연면적합계_m2",
        "상가건축면적x층수_분자",
        "상가건축면적x층수_밀도",
        "숙박입체화재하중_분자",
        "상가숙박_면적층수_분자",
        "상가숙박_면적층수_밀도",
        "상가입체화재하중_분자",
        "상가_입체화재하중밀도",
        "건물데이터_0여부",
        "면적데이터_0여부",
        "중심위도",
        "중심경도",
        "x_min_EPSG5186",
        "y_min_EPSG5186",
        "x_max_EPSG5186",
        "y_max_EPSG5186",
    ]
    for row in out[output_columns].to_dict(orient="records"):
        feature = features_by_id[row["그리드ID"]]
        output_features.append(
            {
                "type": "Feature",
                "geometry": feature["geometry"],
                "properties": row,
            }
        )

    metrics = {
        "footprint_floor_density": out["건축면적x층수_밀도"].tolist(),
        "building_fire_density": out["전체건물_입체화재하중밀도"].tolist(),
        "commercial_footprint_floor_density": out["상가건축면적x층수_밀도"].tolist(),
        "commercial_fire_density": out["상가_입체화재하중밀도"].tolist(),
        "building_count": out["전체건물수"].tolist(),
        "commercial_count": out["상가수"].tolist(),
        "lodging_count": out["숙박시설수"].tolist(),
        "commercial_lodging_count": out["상가숙박_개수"].tolist(),
        "building_per_ha": out["전체건물수_per_ha"].tolist(),
        "commercial_per_ha": out["상가수_per_ha"].tolist(),
        "lodging_per_ha": out["숙박시설수_per_ha"].tolist(),
        "commercial_lodging_per_ha": out["상가숙박_개수_per_ha"].tolist(),
        "commercial_lodging_area_floor_density": out["상가숙박_면적층수_밀도"].tolist(),
    }
    caps = {
        key: {
            "max": round(max(values) if values else 0.0, 6),
            "p98": round(max(quantile(values, 0.98), 1.0), 6),
        }
        for key, values in metrics.items()
    }

    geojson = {"type": "FeatureCollection", "features": output_features}
    summary = {
        "격자수": int(len(out)),
        "건물데이터_있는_격자수": int((out["전체건물수"] > 0).sum()),
        "상가_있는_격자수": int((out["상가수"] > 0).sum()),
        "전체건물수_합계": int(out["전체건물수"].sum()),
        "상가수_합계": int(out["상가수"].sum()),
        "숙박시설수_합계": int(out["숙박시설수"].sum()),
        "상가숙박_개수_합계": int(out["상가숙박_개수"].sum()),
        "상가숙박_면적층수_분자_합계": round(float(out["상가숙박_면적층수_분자"].sum()), 3),
        "건축면적입력건물수_합계": int(out["건축면적입력건물수"].sum()),
        "건축면적역추산건물수_합계": int(out["건축면적역추산건물수"].sum()),
        "건축면적산정불가건물수_합계": int(out["건축면적산정불가건물수"].sum()),
        "면적계산가능건물수_합계": int(out["면적계산가능건물수"].sum()),
        "면적보정건물수_합계": int(out["면적보정건물수"].sum()),
        "표제부보정건물수_합계": int(out["표제부보정건물수"].sum()),
        "건물있지만_면적분자0_격자수": int(((out["전체건물수"] > 0) & (out["전체건물입체화재하중_분자"] <= 0)).sum()),
        "전체건물입체화재하중_분자_원본_합계": round(float(out["전체건물입체화재하중_분자_원본"].sum()), 3),
        "계산건축면적합계_m2": round(float(out["계산건축면적합계_m2"].sum()), 3),
        "건축면적x층수_분자_합계": round(float(out["건축면적x층수_분자"].sum()), 3),
        "전체건물입체화재하중_분자_합계": round(float(out["전체건물입체화재하중_분자"].sum()), 3),
        "전체건물입체화재하중_분자_감소량": round(
            float(out["전체건물입체화재하중_분자_원본"].sum() - out["전체건물입체화재하중_분자"].sum()),
            3,
        ),
        "상가건축면적x층수_분자_합계": round(float(out["상가건축면적x층수_분자"].sum()), 3),
        "상가입체화재하중_분자_합계": round(float(out["상가입체화재하중_분자"].sum()), 3),
        "건축면적x층수_밀도_최대": caps["footprint_floor_density"]["max"],
        "전체건물_입체화재하중밀도_최대": caps["building_fire_density"]["max"],
        "상가건축면적x층수_밀도_최대": caps["commercial_footprint_floor_density"]["max"],
        "상가_입체화재하중밀도_최대": caps["commercial_fire_density"]["max"],
        "기준면적_m2": GRID_AREA_M2,
        "건축면적x층수_계산식": "건축면적x층수_밀도 = Σ(계산건축면적 × 총층수 기반 분자) / 50,000㎡; 건축면적이 없고 연면적/층수가 있으면 계산건축면적=연면적/총층수로 역추산, 건축면적×총층수가 연면적보다 크면 연면적으로 상한 처리",
        "계산식": "전체건물_입체화재하중밀도 = Σ(보정분자) / 50,000㎡; 보정분자 = 연면적>0이고 건축면적<=0 또는 건축면적>연면적 또는 건축면적×총층수>연면적이면 연면적, 그 외 건축면적×총층수",
    }
    return geojson, out[output_columns], {"caps": caps, "summary": summary}


def write_outputs(geojson: dict, csv_df: pd.DataFrame, summary: dict, building_points: list[list[object]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    with OUT_GEOJSON.open("w", encoding="utf-8") as f:
        json.dump(geojson, f, ensure_ascii=False, separators=(",", ":"))
    with OUT_SUMMARY.open("w", encoding="utf-8-sig") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with OUT_BUILDING_POINTS_JS.open("w", encoding="utf-8") as f:
        f.write("window.BUILDING_POINT_FIELDS=['lat','lng','commercial','area_m2','floors','gu','dong','use_name','grid_id','building_id','building_area_m2','calc_building_area_m2','building_area_estimated','building_area_floor_raw_m2','building_area_floor_numerator_m2','building_name','registry_corrected','registry_pk'];\n")
        f.write("window.BUILDING_POINTS=")
        json.dump(building_points, f, ensure_ascii=False, separators=(",", ":"))
        f.write(";\n")

    rows = [
        ("그리드ID", "격자", "50,000㎡ 격자 고유 ID", "라벨/팝업"),
        ("주요구", "위치", "격자가 주로 포함하는 자치구", "팝업/필터"),
        ("주요법정동명", "위치", "격자가 주로 포함하는 법정동", "팝업/필터"),
        ("기준면적_m2", "분모", "격자 기준면적. 모든 격자 50,000㎡", "밀도 계산"),
        ("전체건물수", "전체건물", "격자 안 전체 건물 개수", "건물수 모드/팝업"),
        ("건축면적입력건물수", "검증", "건축면적이 0보다 큰 건물 수", "호버/팝업"),
        ("건축면적역추산건물수", "검증", "건축면적이 0이고 연면적/층수로 건축면적을 역추산한 건물 수", "호버/팝업"),
        ("건축면적산정불가건물수", "검증", "건축면적도 없고 연면적/층수 역추산도 불가능한 건물 수", "호버/팝업"),
        ("연면적입력건물수", "검증", "연면적이 0보다 큰 건물 수", "호버/팝업"),
        ("층수입력건물수", "검증", "지상층수+지하층수가 0보다 큰 건물 수", "호버/팝업"),
        ("전체건물수_per_ha", "전체건물", "전체건물수 / 5ha", "건물수/ha 모드"),
        ("전체건축면적합계_m2", "전체건물", "격자 안 전체 건물 원본 건축면적 합계", "팝업/검증"),
        ("계산건축면적합계_m2", "전체건물", "원본 건축면적과 연면적/층수 역추산 건축면적을 합친 값", "팝업/검증"),
        ("건축면적x층수_분자", "전체건물", "Σ(계산건축면적 × 총층수 기반 분자). 건축면적이 없고 연면적/층수가 있으면 연면적/총층수로 역추산", "건축면적×층수 밀도 계산"),
        ("건축면적x층수_밀도", "전체건물", "건축면적x층수_분자 / 50,000㎡", "기본 지도 색상/라벨"),
        ("전체건물입체화재하중_분자_원본", "전체건물", "Σ(건축면적 × 총층수), 건축면적×총층수가 0이면 연면적 사용", "보정 전 비교"),
        ("전체건물입체화재하중_분자", "전체건물", "보정분자 합계. 이상 면적은 연면적으로 대체", "전체건물 보정밀도 계산"),
        ("면적보정건물수", "검증", "건축면적<=0, 건축면적>연면적, 건축면적×총층수>연면적 중 하나에 해당해 연면적으로 대체한 건물수", "팝업/검증"),
        ("표제부보정건물수", "검증", "SHP 원본 면적/층수 0 건물 중 표제부 후보로 면적·층수·건물명을 보정한 건물수", "팝업/실제건물 레이어"),
        ("전체건물_입체화재하중밀도", "전체건물", "전체건물입체화재하중_분자 / 50,000㎡", "기본 지도 색상/라벨"),
        ("상가수", "상가", "건물용도코드 03000/04000/07000 또는 용도명 근린생활시설/판매시설 건물수", "상가수 모드"),
        ("상가수_per_ha", "상가", "상가수 / 5ha", "상가수/ha 모드"),
        ("숙박시설수", "숙박", "격자 안 숙박시설 수", "숙박시설수 모드/팝업"),
        ("숙박시설수_per_ha", "숙박", "숙박시설수 / 5ha", "숙박시설수/ha 모드"),
        ("상가숙박_개수", "상가+숙박", "상가수 + 숙박시설수", "상가+숙박 개수 모드"),
        ("상가숙박_개수_per_ha", "상가+숙박", "(상가수 + 숙박시설수) / 5ha", "상가+숙박 개수 밀도 모드"),
        ("숙박연면적합계_m2", "숙박", "격자 안 숙박시설 연면적 합계", "팝업/규모 비교"),
        ("상가건축면적x층수_분자", "상가", "상가 건물의 Σ(건축면적 × 총층수)", "상가 건축면적×층수 밀도 계산"),
        ("상가건축면적x층수_밀도", "상가", "상가건축면적x층수_분자 / 50,000㎡", "상가 건축면적×층수 모드"),
        ("숙박입체화재하중_분자", "숙박", "숙박시설 연면적 합계 기반 분자", "숙박 규모 비교"),
        ("상가숙박_면적층수_분자", "상가+숙박", "상가건축면적x층수_분자 + 숙박입체화재하중_분자", "면적·층수 기반 보조 지표"),
        ("상가숙박_면적층수_밀도", "상가+숙박", "상가숙박_면적층수_분자 / 50,000㎡", "면적·층수 기반 보조 지표"),
        ("상가입체화재하중_분자", "상가", "Σ(상가 건축면적 × 총층수), 0이면 연면적 사용", "상가 밀도 계산"),
        ("상가_입체화재하중밀도", "상가", "상가입체화재하중_분자 / 50,000㎡", "상가 밀도 모드"),
        ("건물데이터_0여부", "검증", "전체건물수 0개 여부", "빨간 격자 표시"),
        ("중심위도", "좌표", "격자 중심 위도", "라벨/집계점"),
        ("중심경도", "좌표", "격자 중심 경도", "라벨/집계점"),
    ]
    with OUT_COLUMNS.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["컬럼명", "구분", "설명", "시각화_사용처"])
        writer.writerows(rows)


def build_html(geojson: dict, caps: dict, summary: dict) -> str:
    grid_json = json.dumps(geojson, ensure_ascii=False, separators=(",", ":"))
    caps_json = json.dumps(caps, ensure_ascii=False, separators=(",", ":"))
    summary_json = json.dumps(summary, ensure_ascii=False, separators=(",", ":"))

    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>서울 10개구 상가+숙박 개수 밀도</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.css">
<script src="https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
html, body, #map {{
  width: 100%;
  height: 100%;
  margin: 0;
  padding: 0;
  font-family: "Malgun Gothic", Arial, sans-serif;
}}
.leaflet-container {{ font-size: 1rem; background: #eef2f5; }}
.title-panel {{
  position: fixed;
  top: 12px;
  left: 64px;
  z-index: 9999;
  width: min(430px, calc(100vw - 430px));
  min-width: 360px;
  padding: 12px 16px;
  border: 1px solid #cbd5e1;
  border-radius: 8px;
  background: rgba(255,255,255,.94);
  box-shadow: 0 6px 18px rgba(15,23,42,.14);
  color: #1f2937;
}}
.title-panel h1 {{
  margin: 0 0 5px;
  font-size: 20px;
  line-height: 1.25;
  letter-spacing: 0;
}}
.title-panel .sub {{
  font-size: 13px;
  color: #475569;
  line-height: 1.45;
}}
.legend-panel {{
  position: fixed;
  left: 18px;
  bottom: 28px;
  z-index: 9999;
  width: 330px;
  padding: 12px 14px;
  border: 1px solid #cbd5e1;
  border-radius: 8px;
  background: rgba(255,255,255,.94);
  box-shadow: 0 6px 18px rgba(15,23,42,.18);
  color: #374151;
  font-size: 14px;
  line-height: 1.6;
}}
.legend-panel strong {{
  display: block;
  margin-bottom: 4px;
  color: #1f2937;
}}
.swatch {{
  display: inline-block;
  width: 16px;
  height: 12px;
  margin-right: 8px;
  border: 1px solid rgba(15,23,42,.28);
  vertical-align: -1px;
}}
.metric-control {{
  padding: 12px 14px;
  border: 1px solid #cbd5e1;
  border-radius: 8px;
  background: rgba(255,255,255,.96);
  box-shadow: 0 6px 18px rgba(15,23,42,.16);
  color: #1f2937;
  font-family: "Malgun Gothic", Arial, sans-serif;
  font-size: 13px;
  line-height: 1.6;
}}
.metric-control .control-title {{
  margin-bottom: 6px;
  font-weight: 800;
}}
.metric-control label {{
  display: block;
  white-space: nowrap;
  cursor: pointer;
}}
.metric-control input {{
  margin-right: 6px;
}}
.density-label-icon {{
  background: transparent;
  border: 0;
}}
.density-label {{
  display: block;
  min-width: 38px;
  padding: 2px 5px;
  border: 1px solid rgba(15, 23, 42, .28);
  border-radius: 4px;
  background: rgba(255, 255, 255, .9);
  color: #0f172a;
  font-size: 10.5px;
  font-weight: 800;
  line-height: 1.15;
  text-align: center;
  box-shadow: 0 1px 4px rgba(15, 23, 42, .22);
  pointer-events: none;
}}
.density-label.zero {{
  opacity: .62;
  font-weight: 700;
}}
.popup-table {{
  border-collapse: collapse;
  min-width: 292px;
  font-size: 13px;
}}
.popup-table th {{
  padding: 4px 10px 4px 0;
  text-align: left;
  color: #475569;
  white-space: nowrap;
}}
.popup-table td {{
  padding: 4px 0;
  text-align: right;
  color: #111827;
  font-weight: 700;
}}
.actual-building-status {{
  padding: 9px 12px;
  border: 1px solid #cbd5e1;
  border-radius: 8px;
  background: rgba(255,255,255,.94);
  box-shadow: 0 6px 18px rgba(15,23,42,.16);
  color: #1f2937;
  font-family: "Malgun Gothic", Arial, sans-serif;
  font-size: 12px;
  line-height: 1.45;
}}
.building-dot {{
  display: inline-block;
  width: 9px;
  height: 9px;
  margin-right: 8px;
  border-radius: 999px;
  vertical-align: 0;
}}
.grid-hover-tooltip {{
  border: 1px solid #0f766e;
  border-radius: 8px;
  box-shadow: 0 8px 22px rgba(15,23,42,.22);
  color: #0f172a;
}}
.grid-hover-title {{
  margin-bottom: 4px;
  font-weight: 800;
}}
.grid-hover-count {{
  margin: 4px 0;
  color: #0f766e;
  font-size: 18px;
  font-weight: 900;
  line-height: 1.15;
}}
.grid-hover-meta {{
  color: #475569;
  font-size: 12px;
  line-height: 1.35;
}}
.grid-hover-warning {{
  margin-top: 4px;
  color: #b45309;
  font-size: 12px;
  font-weight: 800;
  line-height: 1.35;
}}
@media (max-width: 760px) {{
  .title-panel {{
    left: 56px;
    right: 12px;
    width: auto;
    min-width: 0;
  }}
  .title-panel h1 {{ font-size: 17px; }}
  .legend-panel {{
    width: calc(100vw - 56px);
    left: 12px;
    bottom: 18px;
  }}
}}
</style>
</head>
<body>
<div id="map"></div>
<div class="title-panel">
  <h1>서울 10개구 상가+숙박 개수 밀도</h1>
  <div class="sub" id="subtitle">각 50,000㎡ 격자별 상가수와 숙박시설수를 합산한 뒤 ha 기준으로 밀도화했습니다. 호버와 팝업에서 면적·층수 기반 규모도 함께 확인할 수 있습니다.</div>
</div>
<div class="legend-panel">
  <strong id="legend-title">상가+숙박 개수/ha</strong>
  <div><span class="swatch" style="background:#f8fafc"></span>상가+숙박 0개</div>
  <div><span class="swatch" style="background:#deebf7"></span>낮은 밀도</div>
  <div><span class="swatch" style="background:#08519c"></span>높은 밀도</div>
  <div><span class="building-dot" style="background:#38bdf8"></span>실제 건물 위치</div>
  <div><span class="building-dot" style="background:#f97316"></span>실제 상가 건물</div>
  <div><span class="building-dot" style="background:#f43f5e"></span>표제부 보정 건물</div>
  <div id="legend-range" style="margin-top:6px;color:#475569"></div>
  <div style="margin-top:6px;color:#475569">격자 안 숫자는 상가+숙박 개수/ha입니다.</div>
  <div style="margin-top:6px;color:#475569">상가 {summary["상가수_합계"]:,}개 · 숙박 {summary["숙박시설수_합계"]:,}개 · 합계 {summary["상가숙박_개수_합계"]:,}개</div>
</div>
<script src="actual_buildings_10gu_0417.js"></script>
<script>
const GRID_DATA = {grid_json};
const CAPS = {caps_json};
const SUMMARY = {summary_json};
const BUILDING_POINTS = window.BUILDING_POINTS || [];

const METRICS = {{
  footprint_floor_density: {{
    label: "건축면적×층수 밀도",
    field: "건축면적x층수_밀도",
    decimals: 2,
    zeroDecimals: 0,
    method: "Σ(계산건축면적×총층수 기반 분자) / 50,000㎡",
    subtitle: "건축면적이 없고 연면적과 층수가 있으면 연면적/총층수로 건축면적을 역추산해 계산합니다."
  }},
  building_fire_density: {{
    label: "전체건물 보정밀도",
    field: "전체건물_입체화재하중밀도",
    decimals: 2,
    zeroDecimals: 0,
    method: "Σ(보정분자) / 50,000㎡",
    subtitle: "건축면적×층수가 이상하거나 0이면 연면적으로 대체한 보정밀도입니다."
  }},
  commercial_footprint_floor_density: {{
    label: "상가 건축면적×층수 밀도",
    field: "상가건축면적x층수_밀도",
    decimals: 2,
    zeroDecimals: 0,
    method: "Σ(상가 계산건축면적×총층수 기반 분자) / 50,000㎡",
    subtitle: "상가 건물만 대상으로, 건축면적이 없으면 연면적/총층수로 역추산해 계산합니다."
  }},
  commercial_fire_density: {{
    label: "상가 보정밀도",
    field: "상가_입체화재하중밀도",
    decimals: 2,
    zeroDecimals: 0,
    method: "Σ(상가 보정분자) / 50,000㎡",
    subtitle: "상가 건물만 대상으로, 건축면적×층수가 이상하거나 0이면 연면적으로 대체한 밀도입니다."
  }},
  building_count: {{
    label: "전체 건물수",
    field: "전체건물수",
    decimals: 0,
    zeroDecimals: 0,
    method: "격자 안 건물 수",
    subtitle: "격자 안 숫자는 전체 건물수입니다."
  }},
  commercial_count: {{
    label: "상가수",
    field: "상가수",
    decimals: 0,
    zeroDecimals: 0,
    method: "격자 안 상가 건물 수",
    subtitle: "격자 안 숫자는 상가수입니다."
  }},
  lodging_count: {{
    label: "숙박시설수",
    field: "숙박시설수",
    decimals: 0,
    zeroDecimals: 0,
    method: "격자 안 숙박시설 수",
    subtitle: "격자 안 숫자는 숙박시설수입니다."
  }},
  commercial_lodging_count: {{
    label: "상가+숙박 개수",
    field: "상가숙박_개수",
    decimals: 0,
    zeroDecimals: 0,
    method: "상가수 + 숙박시설수",
    subtitle: "격자 안 숫자는 상가수와 숙박시설수를 합친 개수입니다."
  }},
  building_per_ha: {{
    label: "전체 건물수/ha",
    field: "전체건물수_per_ha",
    decimals: 1,
    zeroDecimals: 0,
    method: "전체건물수 / 5ha",
    subtitle: "격자 안 숫자는 ha당 전체 건물수입니다."
  }},
  commercial_per_ha: {{
    label: "상가수/ha",
    field: "상가수_per_ha",
    decimals: 1,
    zeroDecimals: 0,
    method: "상가수 / 5ha",
    subtitle: "격자 안 숫자는 ha당 상가수입니다."
  }},
  lodging_per_ha: {{
    label: "숙박시설수/ha",
    field: "숙박시설수_per_ha",
    decimals: 1,
    zeroDecimals: 0,
    method: "숙박시설수 / 5ha",
    subtitle: "격자 안 숫자는 ha당 숙박시설수입니다."
  }},
  commercial_lodging_per_ha: {{
    label: "상가+숙박 개수/ha",
    field: "상가숙박_개수_per_ha",
    decimals: 1,
    zeroDecimals: 0,
    method: "(상가수 + 숙박시설수) / 5ha",
    subtitle: "상가와 숙박시설을 합친 개수를 ha 기준으로 밀도화한 최종값입니다. 호버와 팝업에서 면적·층수 기반 규모도 함께 확인할 수 있습니다."
  }}
}};

const BLUE_SCALE = [
  "#deebf7", "#c6dbef", "#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#08519c"
];
let currentMetric = "commercial_lodging_per_ha";
let labelsVisible = true;
let actualBuildingsVisible = false;
const LABEL_MIN_VALUE = 0.01;
const ACTUAL_BUILDING_MIN_ZOOM = 15;
const ACTUAL_BUILDING_RENDER_LIMIT = 8000;

const map = L.map("map", {{
  center: [37.56, 126.99],
  zoom: 11,
  preferCanvas: true
}});

const osm = L.tileLayer("https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png", {{
  maxZoom: 19,
  attribution: "&copy; OpenStreetMap contributors"
}});
const positron = L.tileLayer("https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png", {{
  maxZoom: 20,
  attribution: "&copy; OpenStreetMap contributors &copy; CARTO"
}}).addTo(map);

function getProp(feature, field) {{
  const value = Number(feature.properties[field] || 0);
  return Number.isFinite(value) ? value : 0;
}}

function getColor(feature) {{
  const value = getProp(feature, METRICS[currentMetric].field);
  if (value <= 0) return "#f8fafc";
  const cap = Math.max(CAPS[currentMetric].p98 || CAPS[currentMetric].max || 1, 1);
  const t = Math.max(0, Math.min(value / cap, 1));
  const idx = Math.min(BLUE_SCALE.length - 1, Math.floor(t * BLUE_SCALE.length));
  return BLUE_SCALE[idx];
}}

function gridStyle(feature) {{
  const value = getProp(feature, METRICS[currentMetric].field);
  return {{
    color: value <= 0 ? "#cbd5e1" : "#1f2937",
    weight: value <= 0 ? 0.9 : 1.25,
    fillColor: getColor(feature),
    fillOpacity: value <= 0 ? 0.38 : 0.62,
    opacity: 0.72
  }};
}}

function fmt(value, decimals) {{
  const numberValue = Number(value || 0);
  if (!Number.isFinite(numberValue) || numberValue === 0) return "0";
  return numberValue.toLocaleString("ko-KR", {{
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals
  }});
}}

function labelText(feature) {{
  const metric = METRICS[currentMetric];
  const value = getProp(feature, metric.field);
  return fmt(value, value === 0 ? metric.zeroDecimals : metric.decimals);
}}

function popupHtml(feature) {{
  const p = feature.properties;
  const rows = [
    ["그리드ID", p["그리드ID"]],
    ["구/동", `${{p["주요구"]}} ${{p["주요법정동명"]}}`],
    ["최종값: 상가+숙박 개수/ha", fmt(p["상가숙박_개수_per_ha"], 1)],
    ["상가+숙박 개수", fmt(p["상가숙박_개수"], 0)],
    ["상가수", fmt(p["상가수"], 0)],
    ["숙박시설수", fmt(p["숙박시설수"], 0)],
    ["상가 건축면적×층수 분자", fmt(p["상가건축면적x층수_분자"], 1)],
    ["숙박 연면적 기반 분자", fmt(p["숙박입체화재하중_분자"], 1)],
    ["상가+숙박 면적·층수 밀도", fmt(p["상가숙박_면적층수_밀도"], 3)],
    ["전체 건물수", fmt(p["전체건물수"], 0)],
    ["표제부 보정 건물수", fmt(p["표제부보정건물수"], 0)],
    ["분모", fmt(p["기준면적_m2"], 0)]
  ];
  return `<table class="popup-table">${{rows.map(row => `<tr><th>${{row[0]}}</th><td>${{row[1]}}</td></tr>`).join("")}}</table>`;
}}

function tooltipHtml(feature) {{
  const p = feature.properties;
  return `
    <div class="grid-hover-title">${{p["주요구"]}} ${{p["주요법정동명"]}}</div>
    <div class="grid-hover-count">상가+숙박 ${{fmt(p["상가숙박_개수"], 0)}}개</div>
    <div>개수 밀도: <strong>${{fmt(p["상가숙박_개수_per_ha"], 1)}}/ha</strong></div>
    <div class="grid-hover-meta">상가 ${{fmt(p["상가수"], 0)}}개 · 숙박 ${{fmt(p["숙박시설수"], 0)}}개</div>
    <div class="grid-hover-meta">면적·층수 밀도 ${{fmt(p["상가숙박_면적층수_밀도"], 3)}}</div>
    <div class="grid-hover-meta">${{p["그리드ID"]}}</div>
  `;
}}

const gridLayer = L.geoJSON(GRID_DATA, {{
  style: gridStyle,
  onEachFeature: function(feature, layer) {{
    layer.bindPopup(popupHtml(feature), {{maxWidth: 390}});
    layer.bindTooltip(tooltipHtml(feature), {{sticky: true, className: "grid-hover-tooltip"}});
    layer.on("mouseover", function() {{
      layer.setStyle({{weight: 3, color: "#0f766e", fillOpacity: 0.84}});
      layer.setTooltipContent(tooltipHtml(feature));
      layer.bringToFront();
    }});
    layer.on("mouseout", function() {{
      gridLayer.resetStyle(layer);
    }});
  }}
}}).addTo(map);

const labelLayer = L.layerGroup().addTo(map);
const buildingDotLayer = L.layerGroup();
const commercialDotLayer = L.layerGroup();
const actualBuildingLayer = L.layerGroup();
const actualBuildingRenderer = L.canvas({{padding: 0.5}});
let actualBuildingRenderTimer = null;

const actualBuildingStatus = L.control({{position: "bottomright"}});
actualBuildingStatus.onAdd = function() {{
  const div = L.DomUtil.create("div", "actual-building-status");
  div.style.display = "none";
  return div;
}};
actualBuildingStatus.addTo(map);

function setActualBuildingStatus(message) {{
  const el = actualBuildingStatus.getContainer();
  if (!el) return;
  el.textContent = message || "";
  el.style.display = message ? "block" : "none";
}}

function addLabels() {{
  labelLayer.clearLayers();
  if (!labelsVisible) return;
  GRID_DATA.features.forEach(feature => {{
    const p = feature.properties;
    const lat = Number(p["중심위도"]);
    const lng = Number(p["중심경도"]);
    if (!Number.isFinite(lat) || !Number.isFinite(lng)) return;
    const value = getProp(feature, METRICS[currentMetric].field);
    if (value < LABEL_MIN_VALUE) return;
    const html = `<span class="density-label ${{value === 0 ? "zero" : ""}}">${{labelText(feature)}}</span>`;
    L.marker([lat, lng], {{
      interactive: false,
      icon: L.divIcon({{
        className: "density-label-icon",
        html,
        iconSize: [50, 18],
        iconAnchor: [25, 9]
      }})
    }}).addTo(labelLayer);
  }});
}}

function addDots() {{
  buildingDotLayer.clearLayers();
  commercialDotLayer.clearLayers();
  const maxBuilding = Math.max(CAPS.building_count.max || 1, 1);
  const maxCommercial = Math.max(CAPS.commercial_count.max || 1, 1);
  GRID_DATA.features.forEach(feature => {{
    const p = feature.properties;
    const lat = Number(p["중심위도"]);
    const lng = Number(p["중심경도"]);
    if (!Number.isFinite(lat) || !Number.isFinite(lng)) return;
    const buildings = getProp(feature, "전체건물수");
    const shops = getProp(feature, "상가수");
    if (buildings > 0) {{
      L.circleMarker([lat, lng], {{
        radius: 2 + Math.sqrt(buildings / maxBuilding) * 8,
        color: "#111827",
        weight: 0,
        fillColor: "#111827",
        fillOpacity: 0.42
      }}).bindTooltip(`건물 ${{fmt(buildings, 0)}}개<br>${{p["주요구"]}} ${{p["주요법정동명"]}}`, {{sticky: true}}).addTo(buildingDotLayer);
    }}
    if (shops > 0) {{
      L.circleMarker([lat, lng], {{
        radius: 2 + Math.sqrt(shops / maxCommercial) * 8,
        color: "#92400e",
        weight: 1,
        opacity: 0.75,
        fillColor: "#f59e0b",
        fillOpacity: 0.62
      }}).bindTooltip(`상가 ${{fmt(shops, 0)}}개<br>${{p["주요구"]}} ${{p["주요법정동명"]}}`, {{sticky: true}}).addTo(commercialDotLayer);
    }}
  }});
}}

function actualBuildingTooltip(item) {{
  const typeLabel = item[2] ? "상가 건물" : "일반 건물";
  const useName = item[7] || "용도 미상";
  const buildingName = item[15] || "";
  const corrected = item[16] === 1;
  const registryPk = item[17] || "";
  const title = buildingName || `${{item[5]}} ${{item[6]}}`;
  const areaLabel = item[12]
    ? `계산건축면적 ${{fmt(item[11], 1)}}㎡ (연면적/층수 역추산)`
    : `건축면적 ${{fmt(item[10], 1)}}㎡`;
  const correctionLabel = corrected
    ? `<br>표제부 보정 적용${{registryPk ? ` · PK ${{registryPk}}` : ""}}`
    : "";
  return `<strong>${{title}}</strong><br>${{item[5]}} ${{item[6]}} · ${{typeLabel}} · ${{useName}}<br>${{areaLabel}} × ${{fmt(item[4], 0)}}층 = ${{fmt(item[14], 1)}}㎡<br>연면적 ${{fmt(item[3], 1)}}㎡<br>${{item[8]}}<br>ID ${{item[9]}}${{correctionLabel}}`;
}}

function renderActualBuildings() {{
  actualBuildingLayer.clearLayers();
  if (!actualBuildingsVisible) return;
  if (!BUILDING_POINTS.length) {{
    setActualBuildingStatus("실제 건물 데이터가 없습니다.");
    return;
  }}
  if (map.getZoom() < ACTUAL_BUILDING_MIN_ZOOM) {{
    setActualBuildingStatus(`실제 건물 위치 ${{BUILDING_POINTS.length.toLocaleString("ko-KR")}}개 · 확대 ${{ACTUAL_BUILDING_MIN_ZOOM}}+`);
    return;
  }}

  const bounds = map.getBounds().pad(0.08);
  const south = bounds.getSouth();
  const north = bounds.getNorth();
  const west = bounds.getWest();
  const east = bounds.getEast();
  let rendered = 0;
  let overflow = false;

  for (const item of BUILDING_POINTS) {{
    const lat = item[0];
    const lng = item[1];
    if (lat < south || lat > north || lng < west || lng > east) continue;
    if (rendered >= ACTUAL_BUILDING_RENDER_LIMIT) {{
      overflow = true;
      break;
    }}
    const isCommercial = item[2] === 1;
    const isCorrected = item[16] === 1;
    L.circleMarker([lat, lng], {{
      renderer: actualBuildingRenderer,
      radius: isCorrected ? 5.2 : (isCommercial ? 3.2 : 2.2),
      color: isCorrected ? "#be123c" : (isCommercial ? "#7c2d12" : "#075985"),
      weight: isCorrected ? 1.4 : (isCommercial ? 0.8 : 0.45),
      opacity: isCorrected ? 0.95 : 0.86,
      fillColor: isCorrected ? "#f43f5e" : (isCommercial ? "#f97316" : "#38bdf8"),
      fillOpacity: isCorrected ? 0.9 : (isCommercial ? 0.82 : 0.58)
    }}).bindTooltip(actualBuildingTooltip(item), {{sticky: true}}).addTo(actualBuildingLayer);
    rendered += 1;
  }}

  const suffix = overflow ? ` / ${{ACTUAL_BUILDING_RENDER_LIMIT.toLocaleString("ko-KR")}}개까지만 표시` : "";
  setActualBuildingStatus(`실제 건물 ${{rendered.toLocaleString("ko-KR")}}개 표시${{suffix}}`);
}}

function scheduleActualBuildingRender() {{
  if (!actualBuildingsVisible) return;
  window.clearTimeout(actualBuildingRenderTimer);
  actualBuildingRenderTimer = window.setTimeout(renderActualBuildings, 120);
}}

function updateLegend() {{
  const metric = METRICS[currentMetric];
  document.getElementById("legend-title").textContent = metric.label;
  document.getElementById("subtitle").textContent = metric.subtitle;
  document.getElementById("legend-range").textContent =
    `계산식: ${{metric.method}} · 라벨 0.01 이상`;
}}

function refreshMetric() {{
  gridLayer.setStyle(gridStyle);
  gridLayer.eachLayer(layer => layer.setTooltipContent(tooltipHtml(layer.feature)));
  addLabels();
  updateLegend();
}}

addLabels();
updateLegend();
map.fitBounds(gridLayer.getBounds(), {{padding: [18, 18]}});

const baseLayers = {{
  "서울 지도 OpenStreetMap": osm,
  "밝은 배경 지도": positron
}};
const overlays = {{
  "50,000㎡ 그리드별 밀도": gridLayer,
  "밀도값 라벨": labelLayer,
  "실제 건물 위치": actualBuildingLayer
}};
L.control.layers(baseLayers, overlays, {{position: "topright", collapsed: false}}).addTo(map);
L.control.scale().addTo(map);

map.on("overlayadd overlayremove", event => {{
  if (event.name === "밀도값 라벨") {{
    labelsVisible = map.hasLayer(labelLayer);
    if (labelsVisible) addLabels();
  }}
  if (event.name === "실제 건물 위치") {{
    actualBuildingsVisible = map.hasLayer(actualBuildingLayer);
    if (actualBuildingsVisible) {{
      renderActualBuildings();
    }} else {{
      actualBuildingLayer.clearLayers();
      setActualBuildingStatus("");
    }}
  }}
}});
map.on("moveend zoomend", scheduleActualBuildingRender);
</script>
</body>
</html>
"""


def main() -> None:
    base_geojson, base_df, origin_x, origin_y = load_base_grid()
    agg, source_summary, building_points = load_and_aggregate_buildings(set(base_df["그리드ID"]), origin_x, origin_y)
    geojson, csv_df, meta = build_grid_dataset(base_geojson, base_df, agg)
    summary = {**source_summary, **meta["summary"]}
    write_outputs(geojson, csv_df, summary, building_points)
    OUT_HTML.write_text(build_html(geojson, meta["caps"], summary), encoding="utf-8-sig")

    print(f"HTML: {OUT_HTML}")
    print(f"BUILDING_POINTS_JS: {OUT_BUILDING_POINTS_JS}")
    print(f"CSV: {OUT_CSV}")
    print(f"GEOJSON: {OUT_GEOJSON}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
