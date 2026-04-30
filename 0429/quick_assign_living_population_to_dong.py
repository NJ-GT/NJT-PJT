from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely import wkt


BASE_DIR = Path(__file__).resolve().parent
ROOT = BASE_DIR.parents[0]
INPUT_PATH = BASE_DIR / "새 폴더" / "날짜별_concat" / "생활인구수_한글컬럼_방문4시간그룹.csv"
DONG_GEOJSON = ROOT / "data" / "법정동별_사용승인구간_공간정보0415.geojson"
OUTPUT_PATH = BASE_DIR / "새 폴더" / "날짜별_concat" / "생활인구수_한글컬럼_방문4시간그룹_동추정.csv"
SUMMARY_PATH = BASE_DIR / "새 폴더" / "날짜별_concat" / "동별_25개월평균_방문생활인구수_빠른추정.csv"

TIME_COLS = [
    "방문생활인구수_00_03시",
    "방문생활인구수_04_07시",
    "방문생활인구수_08_11시",
    "방문생활인구수_12_15시",
    "방문생활인구수_16_19시",
    "방문생활인구수_20_23시",
]


def main() -> None:
    df = pd.read_csv(INPUT_PATH, encoding="utf-8-sig", dtype=str)
    if "상권좌표내용" not in df.columns:
        raise KeyError("상권좌표내용 column not found")

    geometry = df["상권좌표내용"].map(wkt.loads)
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    # Use a metric CRS for stable centroids, then return to WGS84 for joining.
    centroids = gdf.to_crs("EPSG:5179").geometry.centroid
    point_gdf = gdf.drop(columns=["geometry"]).copy()
    point_gdf = gpd.GeoDataFrame(point_gdf, geometry=centroids, crs="EPSG:5179").to_crs("EPSG:4326")

    dong = gpd.read_file(DONG_GEOJSON)
    dong = dong[["구", "법정동명", "geometry"]].to_crs("EPSG:4326")

    joined = gpd.sjoin(point_gdf, dong, how="left", predicate="within").drop(columns=["index_right"])
    joined = joined.rename(columns={"구": "추정_구", "법정동명": "추정_동"})
    joined["추정방식"] = "상권폴리곤_중심점_법정동매칭"
    joined = pd.DataFrame(joined.drop(columns=["geometry"]))
    joined.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    numeric = joined.copy()
    for col in TIME_COLS:
        numeric[col] = pd.to_numeric(numeric[col], errors="coerce").fillna(0)

    monthly_dong = (
        numeric.dropna(subset=["추정_구", "추정_동"])
        .groupby(["파일기준년월", "추정_구", "추정_동"], as_index=False)[TIME_COLS]
        .sum()
    )
    summary = monthly_dong.groupby(["추정_구", "추정_동"], as_index=False)[TIME_COLS].mean()
    summary["25개월평균_방문생활인구수"] = summary[TIME_COLS].sum(axis=1)
    summary = summary.sort_values("25개월평균_방문생활인구수", ascending=False)
    summary.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")

    print(OUTPUT_PATH)
    print(SUMMARY_PATH)
    print(f"rows={len(joined)} matched={joined['추정_동'].notna().sum()} unmatched={joined['추정_동'].isna().sum()}")
    print(summary.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
