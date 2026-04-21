# -*- coding: utf-8 -*-
"""
숙박시설별 주변 인접 건물 노후도 변수를 50m/100m 버퍼로 산출한다.

입력:
  - data/통합숙박시설최종안0415.csv
  - data/분석변수_테이블.csv
  - data/AL_D010_11_20260409/AL_D010_11_20260409_filtered.shp

출력:
  - data/통합숙박시설최종안0415_주변노후도_50m100m.csv
  - data/분석변수_테이블_주변노후도_50m100m.csv
  - data/주변노후건물_50m100m_상관행렬.csv
  - data/주변노후건물_50m100m_vif_전체후보.csv
  - data/주변노후건물_50m100m_vif_핵심후보.csv
  - data/주변노후건물_50m100m_요약.json
  - data/주변노후건물_50m100m_컬럼정의.csv
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree
from shapely.geometry import Point
from statsmodels.stats.outliers_influence import variance_inflation_factor


sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
BUILDING_DIR = DATA_DIR / "AL_D010_11_20260409"

LODGING_PATH = DATA_DIR / "통합숙박시설최종안0415.csv"
ANALYSIS_PATH = DATA_DIR / "분석변수_테이블.csv"
BUILDING_SHP = BUILDING_DIR / "AL_D010_11_20260409_filtered.shp"

OUT_LODGING = DATA_DIR / "통합숙박시설최종안0415_주변노후도_50m100m.csv"
OUT_ANALYSIS = DATA_DIR / "분석변수_테이블_주변노후도_50m100m.csv"
OUT_CORR = DATA_DIR / "주변노후건물_50m100m_상관행렬.csv"
OUT_VIF_FULL = DATA_DIR / "주변노후건물_50m100m_vif_전체후보.csv"
OUT_VIF_CORE = DATA_DIR / "주변노후건물_50m100m_vif_핵심후보.csv"
OUT_VIF_COMBINED = DATA_DIR / "주변노후건물_50m100m_vif_기존분석변수결합.csv"
OUT_SUMMARY = DATA_DIR / "주변노후건물_50m100m_요약.json"
OUT_COLUMNS = DATA_DIR / "주변노후건물_50m100m_컬럼정의.csv"

REFERENCE_DATE = pd.Timestamp("2026-04-15")
RADII_M = (50, 100)
OLD_AGE_YEARS = 30.0
VERY_OLD_AGE_YEARS = 50.0
SELF_DISTANCE_EXCLUDE_M = 1.0


BUILDING_RENAME = {
    "A1": "건물ID",
    "A2": "건물관리번호",
    "A13": "사용승인일",
    "A14": "연면적",
    "A15": "건축면적",
}


def normalize_code(value: object, width: int) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    if not text or text.lower() == "nan":
        return ""
    return text.zfill(width)


def make_lot_key(row: pd.Series) -> str:
    """건축물대장 건물관리번호의 지번 앞 19자리 형식으로 필지키를 만든다."""
    sigungu = normalize_code(row.get("시군구코드"), 5)
    dong = normalize_code(row.get("법정동코드"), 5)
    lot_type_raw = normalize_code(row.get("대지구분코드"), 1)
    lot_type = "2" if lot_type_raw == "1" else "1"
    main_no = normalize_code(row.get("번"), 4)
    sub_no = normalize_code(row.get("지"), 4)
    if not (sigungu and dong and lot_type and main_no and sub_no):
        return ""
    return f"{sigungu}{dong}{lot_type}{main_no}{sub_no}"


def age_years_from_approval(values: pd.Series) -> pd.Series:
    dates = pd.to_datetime(values, errors="coerce")
    dates = dates.where(dates.le(REFERENCE_DATE))
    return (REFERENCE_DATE - dates).dt.days / 365.25


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return np.nan
    return float(numerator) / float(denominator)


def summarize_neighbors(candidates: gpd.GeoDataFrame, radius: int) -> dict[str, float]:
    prefix = f"{radius}m"
    total_count = int(len(candidates))
    valid = candidates[candidates["건물연한"].notna()].copy()
    valid_count = int(len(valid))

    old = valid[valid["건물연한"] >= OLD_AGE_YEARS]
    very_old = valid[valid["건물연한"] >= VERY_OLD_AGE_YEARS]

    floor_area = pd.to_numeric(valid["연면적"], errors="coerce").fillna(0.0).clip(lower=0.0)
    old_floor_area = pd.to_numeric(old["연면적"], errors="coerce").fillna(0.0).clip(lower=0.0)
    very_old_floor_area = pd.to_numeric(very_old["연면적"], errors="coerce").fillna(0.0).clip(lower=0.0)

    total_floor_area = float(floor_area.sum())
    old_floor_area_sum = float(old_floor_area.sum())
    very_old_floor_area_sum = float(very_old_floor_area.sum())

    if valid_count:
        mean_age = float(valid["건물연한"].mean())
        median_age = float(valid["건물연한"].median())
        max_age = float(valid["건물연한"].max())
        mean_year = float(valid["사용승인연도"].mean())
    else:
        mean_age = median_age = max_age = mean_year = np.nan

    nearest_old_distance = float(old["_distance_m"].min()) if len(old) else np.nan
    nearest_very_old_distance = float(very_old["_distance_m"].min()) if len(very_old) else np.nan

    return {
        f"주변건물수_{prefix}": total_count,
        f"주변_사용승인일유효건물수_{prefix}": valid_count,
        f"주변_사용승인일유효률_{prefix}": safe_ratio(valid_count, total_count),
        f"주변_노후건물수_30년이상_{prefix}": int(len(old)),
        f"주변_노후건물비율_30년이상_{prefix}": safe_ratio(len(old), valid_count),
        f"주변_초노후건물수_50년이상_{prefix}": int(len(very_old)),
        f"주변_초노후건물비율_50년이상_{prefix}": safe_ratio(len(very_old), valid_count),
        f"주변_평균건물연한_{prefix}": mean_age,
        f"주변_중위건물연한_{prefix}": median_age,
        f"주변_최대건물연한_{prefix}": max_age,
        f"주변_평균사용승인연도_{prefix}": mean_year,
        f"주변_연면적합계_m2_{prefix}": total_floor_area,
        f"주변_노후연면적비율_30년이상_{prefix}": safe_ratio(old_floor_area_sum, total_floor_area),
        f"주변_초노후연면적비율_50년이상_{prefix}": safe_ratio(very_old_floor_area_sum, total_floor_area),
        f"최근접_노후건물거리_30년이상_m_{prefix}": nearest_old_distance,
        f"최근접_초노후건물거리_50년이상_m_{prefix}": nearest_very_old_distance,
    }


def build_neighbor_features(lodging: pd.DataFrame, buildings: gpd.GeoDataFrame) -> pd.DataFrame:
    lodging_gdf = gpd.GeoDataFrame(
        lodging.copy(),
        geometry=gpd.points_from_xy(lodging["X좌표"], lodging["Y좌표"]),
        crs="EPSG:5181",
    ).to_crs(buildings.crs)
    lodging_gdf["필지키"] = lodging_gdf.apply(make_lot_key, axis=1)

    spatial_index = buildings.sindex
    max_radius = max(RADII_M)
    rows: list[dict[str, float]] = []

    for idx, row in lodging_gdf.iterrows():
        point: Point = row.geometry
        candidate_idx = list(spatial_index.intersection(point.buffer(max_radius).bounds))
        candidates = buildings.iloc[candidate_idx].copy()
        if not candidates.empty:
            distances = candidates.geometry.distance(point)
            candidates["_distance_m"] = distances.to_numpy()
            candidates = candidates[candidates["_distance_m"] <= max_radius].copy()

            same_lot = candidates["건물관리번호"].astype(str).eq(str(row["필지키"]))
            same_point = candidates["_distance_m"] <= SELF_DISTANCE_EXCLUDE_M
            candidates = candidates[~(same_lot | same_point)].copy()
            candidates = candidates.sort_values("_distance_m")

        result: dict[str, float] = {"원본행번호": int(idx), "필지키": row["필지키"]}
        for radius in RADII_M:
            within = candidates[candidates["_distance_m"] <= radius] if not candidates.empty else candidates
            result.update(summarize_neighbors(within, radius))
        rows.append(result)

        if (idx + 1) % 500 == 0 or idx + 1 == len(lodging_gdf):
            print(f"버퍼 계산: {idx + 1:,}/{len(lodging_gdf):,}")

    return pd.DataFrame(rows)


def vif_judgement(value: float) -> str:
    if not np.isfinite(value):
        return "무한대/완전공선성"
    if value > 10:
        return "높음(>10)"
    if value > 5:
        return "주의(5~10)"
    return "양호"


def calculate_vif(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    columns = [col for col in columns if col in df.columns]
    data = df[columns].replace([np.inf, -np.inf], np.nan).copy()
    data = data.dropna(axis=0, how="any")

    usable_columns = []
    dropped = []
    for col in data.columns:
        numeric = pd.to_numeric(data[col], errors="coerce")
        if numeric.notna().sum() < 3 or numeric.nunique(dropna=True) <= 1:
            dropped.append((col, "분산 없음/유효값 부족"))
            continue
        data[col] = numeric
        usable_columns.append(col)

    if len(usable_columns) < 2:
        return pd.DataFrame(
            {
                "변수": usable_columns + [item[0] for item in dropped],
                "VIF": [np.nan] * (len(usable_columns) + len(dropped)),
                "판정": ["계산불가"] * len(usable_columns) + [item[1] for item in dropped],
                "사용행수": [len(data)] * (len(usable_columns) + len(dropped)),
            }
        )

    x = data[usable_columns].astype(float)
    x = (x - x.mean()) / x.std(ddof=0)
    x.insert(0, "const", 1.0)

    records = []
    for offset, col in enumerate(usable_columns, start=1):
        try:
            vif_value = float(variance_inflation_factor(x.values, offset))
        except Exception:
            vif_value = np.inf
        records.append({"변수": col, "VIF": vif_value, "판정": vif_judgement(vif_value), "사용행수": len(x)})

    for col, reason in dropped:
        records.append({"변수": col, "VIF": np.nan, "판정": reason, "사용행수": len(data)})

    return pd.DataFrame(records).sort_values(["VIF", "변수"], ascending=[False, True], na_position="last")


def feature_columns() -> list[str]:
    cols = []
    for radius in RADII_M:
        prefix = f"{radius}m"
        cols.extend(
            [
                f"주변건물수_{prefix}",
                f"주변_사용승인일유효건물수_{prefix}",
                f"주변_사용승인일유효률_{prefix}",
                f"주변_노후건물수_30년이상_{prefix}",
                f"주변_노후건물비율_30년이상_{prefix}",
                f"주변_초노후건물수_50년이상_{prefix}",
                f"주변_초노후건물비율_50년이상_{prefix}",
                f"주변_평균건물연한_{prefix}",
                f"주변_중위건물연한_{prefix}",
                f"주변_최대건물연한_{prefix}",
                f"주변_평균사용승인연도_{prefix}",
                f"주변_연면적합계_m2_{prefix}",
                f"주변_노후연면적비율_30년이상_{prefix}",
                f"주변_초노후연면적비율_50년이상_{prefix}",
                f"최근접_노후건물거리_30년이상_m_{prefix}",
                f"최근접_초노후건물거리_50년이상_m_{prefix}",
            ]
        )
    return cols


def core_feature_columns() -> list[str]:
    return [
        "주변건물수_50m",
        "주변건물수_100m",
        "주변_노후건물비율_30년이상_50m",
        "주변_노후건물비율_30년이상_100m",
        "주변_초노후건물비율_50년이상_50m",
        "주변_초노후건물비율_50년이상_100m",
        "주변_평균건물연한_50m",
        "주변_평균건물연한_100m",
        "주변_노후연면적비율_30년이상_50m",
        "주변_노후연면적비율_30년이상_100m",
    ]


def write_column_dictionary(path: Path) -> None:
    descriptions = []
    for radius in RADII_M:
        prefix = f"{radius}m"
        descriptions.extend(
            [
                (f"주변건물수_{prefix}", f"{radius}m 버퍼 안 전체 인접 건물 수"),
                (f"주변_사용승인일유효건물수_{prefix}", f"{radius}m 버퍼 안 사용승인일이 유효한 인접 건물 수"),
                (f"주변_사용승인일유효률_{prefix}", f"사용승인일 유효 건물 수 / 전체 인접 건물 수"),
                (f"주변_노후건물수_30년이상_{prefix}", f"{radius}m 버퍼 안 30년 이상 인접 건물 수"),
                (f"주변_노후건물비율_30년이상_{prefix}", f"30년 이상 인접 건물 수 / 사용승인일 유효 건물 수"),
                (f"주변_초노후건물수_50년이상_{prefix}", f"{radius}m 버퍼 안 50년 이상 인접 건물 수"),
                (f"주변_초노후건물비율_50년이상_{prefix}", f"50년 이상 인접 건물 수 / 사용승인일 유효 건물 수"),
                (f"주변_평균건물연한_{prefix}", f"{radius}m 버퍼 안 사용승인일 유효 인접 건물의 평균 연한"),
                (f"주변_중위건물연한_{prefix}", f"{radius}m 버퍼 안 사용승인일 유효 인접 건물의 중위 연한"),
                (f"주변_최대건물연한_{prefix}", f"{radius}m 버퍼 안 가장 오래된 인접 건물 연한"),
                (f"주변_평균사용승인연도_{prefix}", f"{radius}m 버퍼 안 사용승인일 유효 인접 건물의 평균 승인연도"),
                (f"주변_연면적합계_m2_{prefix}", f"{radius}m 버퍼 안 사용승인일 유효 인접 건물의 연면적 합계"),
                (f"주변_노후연면적비율_30년이상_{prefix}", f"30년 이상 인접 건물 연면적 / 사용승인일 유효 인접 건물 연면적"),
                (f"주변_초노후연면적비율_50년이상_{prefix}", f"50년 이상 인접 건물 연면적 / 사용승인일 유효 인접 건물 연면적"),
                (f"최근접_노후건물거리_30년이상_m_{prefix}", f"{radius}m 안 최근접 30년 이상 인접 건물까지의 거리"),
                (f"최근접_초노후건물거리_50년이상_m_{prefix}", f"{radius}m 안 최근접 50년 이상 인접 건물까지의 거리"),
            ]
        )
    pd.DataFrame(descriptions, columns=["변수명", "정의"]).to_csv(path, index=False, encoding="utf-8-sig")


def analysis_rows_match_lodging(lodging: pd.DataFrame, analysis: pd.DataFrame) -> bool:
    if len(lodging) != len(analysis) or not {"위도", "경도"}.issubset(analysis.columns):
        return False

    transformer = Transformer.from_crs("EPSG:5181", "EPSG:4326", always_xy=True)
    sample_idx = list(range(min(20, len(lodging))))
    lon, lat = transformer.transform(
        lodging.loc[sample_idx, "X좌표"].to_numpy(),
        lodging.loc[sample_idx, "Y좌표"].to_numpy(),
    )
    lat_diff = np.abs(np.asarray(lat) - analysis.loc[sample_idx, "위도"].to_numpy(dtype=float))
    lon_diff = np.abs(np.asarray(lon) - analysis.loc[sample_idx, "경도"].to_numpy(dtype=float))
    return bool(np.nanmax(lat_diff) < 1e-4 and np.nanmax(lon_diff) < 1e-4)


def append_features_to_analysis(
    lodging: pd.DataFrame,
    analysis: pd.DataFrame,
    features: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame | None, str]:
    if analysis_rows_match_lodging(lodging, analysis):
        enriched = pd.concat([analysis.reset_index(drop=True), features[feature_cols].reset_index(drop=True)], axis=1)
        return enriched, "행순서 직접 결합"

    if not {"위도", "경도"}.issubset(analysis.columns):
        return None, "분석변수_테이블.csv에 위도/경도 컬럼이 없어 결합 생략"

    lodging_points = gpd.GeoDataFrame(
        lodging[["원본행번호", "X좌표", "Y좌표"]].copy(),
        geometry=gpd.points_from_xy(lodging["X좌표"], lodging["Y좌표"]),
        crs="EPSG:5181",
    ).to_crs("EPSG:5186")
    lodging_coords = np.column_stack([lodging_points.geometry.x.to_numpy(), lodging_points.geometry.y.to_numpy()])

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:5186", always_xy=True)
    analysis_x, analysis_y = transformer.transform(
        analysis["경도"].to_numpy(dtype=float),
        analysis["위도"].to_numpy(dtype=float),
    )
    analysis_coords = np.column_stack([analysis_x, analysis_y])

    tree = cKDTree(lodging_coords)
    distances, positions = tree.query(analysis_coords, k=1)
    matched_source_rows = lodging_points.iloc[positions]["원본행번호"].to_numpy(dtype=int)

    matched_features = (
        features.set_index("원본행번호")
        .loc[matched_source_rows, feature_cols]
        .reset_index(drop=True)
    )
    enriched = pd.concat([analysis.reset_index(drop=True), matched_features], axis=1)
    enriched["주변노후도_매칭원본행번호"] = matched_source_rows
    enriched["주변노후도_매칭거리_m"] = distances

    max_distance = float(np.nanmax(distances)) if len(distances) else np.nan
    mean_distance = float(np.nanmean(distances)) if len(distances) else np.nan
    if max_distance > 10:
        message = f"좌표 최근접 결합, 최대 매칭거리 {max_distance:.3f}m 확인 필요"
    else:
        message = f"좌표 최근접 결합, 평균 {mean_distance:.3f}m / 최대 {max_distance:.3f}m"
    return enriched, message


def combined_vif_columns(df: pd.DataFrame) -> list[str]:
    existing = [
        "소방접근성_점수",
        "노후도_점수",
        "반경_50m_건물수",
        "집중도(%)",
        "로그_주변대비_상대위험도_고유단속지점_50m",
        "공식도로폭m",
    ]
    new_candidates = core_feature_columns()
    return [col for col in existing + new_candidates if col in df.columns]


def main() -> None:
    print("숙박시설 로드...")
    lodging = pd.read_csv(LODGING_PATH, encoding="utf-8-sig")
    lodging["원본행번호"] = np.arange(len(lodging), dtype=int)

    print("건물 포인트 로드...")
    buildings = gpd.read_file(BUILDING_SHP)
    buildings = buildings.rename(columns=BUILDING_RENAME)
    keep_cols = ["건물ID", "건물관리번호", "사용승인일", "연면적", "건축면적", "geometry"]
    buildings = buildings[keep_cols].copy()
    buildings["건물관리번호"] = buildings["건물관리번호"].astype(str)
    buildings["건물연한"] = age_years_from_approval(buildings["사용승인일"])
    buildings["사용승인연도"] = pd.to_datetime(buildings["사용승인일"], errors="coerce").dt.year
    buildings["연면적"] = pd.to_numeric(buildings["연면적"], errors="coerce").fillna(0.0)
    buildings = buildings[buildings.geometry.notna() & ~buildings.geometry.is_empty].copy()
    print(f"건물 수: {len(buildings):,}, 사용승인일 유효: {buildings['건물연한'].notna().sum():,}")

    features = build_neighbor_features(lodging, buildings)
    feature_cols = feature_columns()

    enriched_lodging = lodging.merge(features, on="원본행번호", how="left", suffixes=("", "_주변노후도"))
    enriched_lodging.to_csv(OUT_LODGING, index=False, encoding="utf-8-sig")
    print(f"저장: {OUT_LODGING}")

    combined_vif_written = False
    analysis_join_message = "분석변수_테이블.csv 없음"
    if ANALYSIS_PATH.exists():
        analysis = pd.read_csv(ANALYSIS_PATH, encoding="utf-8-sig")
        enriched_analysis, analysis_join_message = append_features_to_analysis(lodging, analysis, features, feature_cols)
        if enriched_analysis is not None:
            enriched_analysis.to_csv(OUT_ANALYSIS, index=False, encoding="utf-8-sig")
            print(f"저장: {OUT_ANALYSIS}")
            print(f"분석변수 결합 방식: {analysis_join_message}")

            vif_combined = calculate_vif(enriched_analysis, combined_vif_columns(enriched_analysis))
            vif_combined.to_csv(OUT_VIF_COMBINED, index=False, encoding="utf-8-sig")
            combined_vif_written = True
            print(f"저장: {OUT_VIF_COMBINED}")
        else:
            print(analysis_join_message)

    corr = features[feature_cols].corr(numeric_only=True)
    corr.to_csv(OUT_CORR, encoding="utf-8-sig")
    print(f"저장: {OUT_CORR}")

    vif_full = calculate_vif(features, feature_cols)
    vif_full.to_csv(OUT_VIF_FULL, index=False, encoding="utf-8-sig")
    print(f"저장: {OUT_VIF_FULL}")

    vif_core = calculate_vif(features, core_feature_columns())
    vif_core.to_csv(OUT_VIF_CORE, index=False, encoding="utf-8-sig")
    print(f"저장: {OUT_VIF_CORE}")

    write_column_dictionary(OUT_COLUMNS)
    print(f"저장: {OUT_COLUMNS}")

    summary = {
        "기준일": REFERENCE_DATE.date().isoformat(),
        "노후기준년수": OLD_AGE_YEARS,
        "초노후기준년수": VERY_OLD_AGE_YEARS,
        "버퍼반경_m": list(RADII_M),
        "숙박시설수": int(len(lodging)),
        "건물수": int(len(buildings)),
        "건물_사용승인일유효수": int(buildings["건물연한"].notna().sum()),
        "산출파일": {
            "숙박시설상세": str(OUT_LODGING.relative_to(BASE_DIR)),
            "분석변수결합": str(OUT_ANALYSIS.relative_to(BASE_DIR)),
            "상관행렬": str(OUT_CORR.relative_to(BASE_DIR)),
            "vif_전체후보": str(OUT_VIF_FULL.relative_to(BASE_DIR)),
            "vif_핵심후보": str(OUT_VIF_CORE.relative_to(BASE_DIR)),
            "vif_기존분석변수결합": str(OUT_VIF_COMBINED.relative_to(BASE_DIR)) if combined_vif_written else "",
            "컬럼정의": str(OUT_COLUMNS.relative_to(BASE_DIR)),
        },
        "분석변수결합메시지": analysis_join_message,
    }
    for radius in RADII_M:
        prefix = f"{radius}m"
        summary[f"{prefix}_주변건물수_평균"] = float(features[f"주변건물수_{prefix}"].mean())
        summary[f"{prefix}_노후건물비율_평균"] = float(features[f"주변_노후건물비율_30년이상_{prefix}"].mean())
        summary[f"{prefix}_초노후건물비율_평균"] = float(features[f"주변_초노후건물비율_50년이상_{prefix}"].mean())
        summary[f"{prefix}_평균건물연한_평균"] = float(features[f"주변_평균건물연한_{prefix}"].mean())

    with OUT_SUMMARY.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"저장: {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
