# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = BASE_DIR.parent

LODGING_CSV = BASE_DIR / "data" / "서울10구_숙소_소방거리_유클리드.csv"
ROAD_GEOJSON = REPO_DIR / "data" / "seoul_road_width_10gu_rw_polygons.geojson"

OUT_FEATURES = BASE_DIR / "data" / "서울10구_숙소_도로폭_공간매칭.csv"
OUT_JOINED = BASE_DIR / "data" / "서울10구_숙소_소방거리_유클리드_도로폭추가.csv"
OUT_SUMMARY = BASE_DIR / "data" / "서울10구_숙소_도로폭_공간매칭_요약.csv"

WGS84 = "EPSG:4326"
METRIC_CRS = "EPSG:5179"

WIDTH_RISK = {
    "6m미만": 1.00,
    "폭6-8m": 0.85,
    "폭8-10m": 0.70,
    "폭10-12m": 0.55,
    "폭12-15m": 0.45,
    "폭15-20m": 0.35,
    "폭20-25m": 0.25,
    "폭25-30m": 0.18,
    "폭30-35m": 0.12,
    "폭35-40m": 0.08,
    "폭40-50m": 0.04,
    "폭50-70m": 0.00,
    "불가": np.nan,
}

WIDTH_SPEED_FACTOR = {
    "6m미만": 1.30,
    "폭6-8m": 1.22,
    "폭8-10m": 1.14,
    "폭10-12m": 1.08,
    "폭12-15m": 1.04,
    "폭15-20m": 1.02,
    "폭20-25m": 1.00,
    "폭25-30m": 0.98,
    "폭30-35m": 0.96,
    "폭35-40m": 0.95,
    "폭40-50m": 0.94,
    "폭50-70m": 0.93,
    "불가": 1.00,
}


def width_bucket_from_m(value: object) -> str:
    width = pd.to_numeric(value, errors="coerce")
    if pd.isna(width):
        return "불가"
    width = float(width)
    if width < 6:
        return "6m미만"
    if width < 8:
        return "폭6-8m"
    if width < 10:
        return "폭8-10m"
    if width < 12:
        return "폭10-12m"
    if width < 15:
        return "폭12-15m"
    if width < 20:
        return "폭15-20m"
    if width < 25:
        return "폭20-25m"
    if width < 30:
        return "폭25-30m"
    if width < 35:
        return "폭30-35m"
    if width < 40:
        return "폭35-40m"
    if width < 50:
        return "폭40-50m"
    return "폭50-70m"


def read_lodging() -> pd.DataFrame:
    # skipinitialspace handles quoted names that contain commas after padded delimiters.
    df = pd.read_csv(LODGING_CSV, encoding="utf-8-sig", skipinitialspace=True)
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()
        df.loc[df[col].isin(["nan", "None"]), col] = ""
    df["숙소ID"] = np.arange(1, len(df) + 1)
    df["위도"] = pd.to_numeric(df["위도"], errors="coerce")
    df["경도"] = pd.to_numeric(df["경도"], errors="coerce")
    return df


def load_roads() -> gpd.GeoDataFrame:
    roads = gpd.read_file(ROAD_GEOJSON)
    if roads.crs is None:
        roads = roads.set_crs(WGS84)
    roads = roads.to_crs(METRIC_CRS)

    usable = roads[
        (roads["width"].fillna("") != "불가")
        & (roads["widthSource"].fillna("") != "실폭만")
        & roads.geometry.notna()
        & (~roads.geometry.is_empty)
    ].copy()
    usable["공식도로폭등급"] = usable["officialWidthM"].map(width_bucket_from_m)
    usable.loc[usable["공식도로폭등급"] == "불가", "공식도로폭등급"] = usable.loc[
        usable["공식도로폭등급"] == "불가", "width"
    ]
    usable["도로폭_위험도"] = usable["공식도로폭등급"].map(WIDTH_RISK)
    usable["도로폭_속도계수"] = usable["공식도로폭등급"].map(WIDTH_SPEED_FACTOR).fillna(1.0)
    return usable


def build_points(df: pd.DataFrame) -> gpd.GeoDataFrame:
    valid = df["위도"].notna() & df["경도"].notna()
    points = gpd.GeoDataFrame(
        df.loc[valid, ["숙소ID", "구", "동", "업소명", "주소", "위도", "경도"]].copy(),
        geometry=gpd.points_from_xy(df.loc[valid, "경도"], df.loc[valid, "위도"]),
        crs=WGS84,
    )
    return points.to_crs(METRIC_CRS)


def attach_features(df: pd.DataFrame, roads: gpd.GeoDataFrame) -> pd.DataFrame:
    points = build_points(df)
    road_cols = [
        "id",
        "rwSn",
        "gu",
        "road",
        "width",
        "widthSource",
        "matchMethod",
        "officialWidthM",
        "공식도로폭등급",
        "matchDistanceM",
        "intersectLengthM",
        "rdsManNo",
        "rnCd",
        "도로폭_위험도",
        "도로폭_속도계수",
        "geometry",
    ]

    matched = gpd.sjoin_nearest(
        points,
        roads[road_cols],
        how="left",
        distance_col="도로폭_거리m",
    ).drop(columns=["index_right"], errors="ignore")
    matched = (
        matched.sort_values(["숙소ID", "도로폭_거리m", "도로폭_위험도"], ascending=[True, True, False])
        .drop_duplicates("숙소ID", keep="first")
        .copy()
    )

    features = pd.DataFrame(
        {
            "숙소ID": matched["숙소ID"].astype(int),
            "구": matched["구"],
            "동": matched["동"],
            "업소명": matched["업소명"],
            "주소": matched["주소"],
            "위도": matched["위도"],
            "경도": matched["경도"],
            "인접도로명": matched["road"],
            "인접도로폭": matched["공식도로폭등급"],
            "지도표시도로폭": matched["width"],
            "도로폭_출처": matched["widthSource"],
            "도로폭_도로구간매칭": matched["matchMethod"],
            "도로폭_거리m": matched["도로폭_거리m"].round(2),
            "공식도로폭m": pd.to_numeric(matched["officialWidthM"], errors="coerce").round(2),
            "도로폭_위험도": pd.to_numeric(matched["도로폭_위험도"], errors="coerce").round(4),
            "도로폭_속도계수": pd.to_numeric(matched["도로폭_속도계수"], errors="coerce").round(4),
            "도로폭_보정이동시간초": np.nan,
            "도로폭_보정예상도착초": np.nan,
            "실폭도로ID": matched["id"],
            "실폭도로일련번호": matched["rwSn"],
            "도로구간관리번호": matched["rdsManNo"],
            "도로명코드": matched["rnCd"],
        }
    )

    indexed = df.set_index("숙소ID")
    if "이동시간초" in df.columns:
        move_sec = pd.to_numeric(indexed.loc[features["숙소ID"], "이동시간초"].to_numpy(), errors="coerce")
        features["도로폭_보정이동시간초"] = np.rint(move_sec * features["도로폭_속도계수"]).astype("Int64")
    if "이동시간초" in df.columns:
        # 기존 예상도착분은 출동준비 60초 + 이동시간 기준이므로 같은 기준으로 초 단위를 산출한다.
        move_sec = pd.to_numeric(indexed.loc[features["숙소ID"], "이동시간초"].to_numpy(), errors="coerce")
        features["도로폭_보정예상도착초"] = np.rint(60 + move_sec * features["도로폭_속도계수"]).astype("Int64")

    joined = df.merge(
        features[
            [
                "숙소ID",
                "인접도로명",
                "인접도로폭",
                "지도표시도로폭",
                "도로폭_출처",
                "도로폭_도로구간매칭",
                "도로폭_거리m",
                "공식도로폭m",
                "도로폭_위험도",
                "도로폭_속도계수",
                "도로폭_보정이동시간초",
                "도로폭_보정예상도착초",
                "실폭도로ID",
                "실폭도로일련번호",
                "도로구간관리번호",
                "도로명코드",
            ]
        ],
        on="숙소ID",
        how="left",
    )
    return features, joined


def save_summary(features: pd.DataFrame) -> None:
    rows = []
    rows.append({"구분": "전체 숙소", "값": len(features)})
    rows.append({"구분": "도로폭 매칭 숙소", "값": int(features["인접도로폭"].notna().sum())})
    rows.append({"구분": "평균 도로까지 거리m", "값": round(float(features["도로폭_거리m"].mean()), 2)})
    rows.append({"구분": "중앙 도로까지 거리m", "값": round(float(features["도로폭_거리m"].median()), 2)})
    rows.append({"구분": "50m 초과 숙소", "값": int((features["도로폭_거리m"] > 50).sum())})
    rows.append({"구분": "100m 초과 숙소", "값": int((features["도로폭_거리m"] > 100).sum())})

    width_counts = features["인접도로폭"].value_counts(dropna=False).reset_index()
    width_counts.columns = ["구분", "값"]
    width_counts["구분"] = "인접도로폭: " + width_counts["구분"].astype(str)

    gu_width = (
        features.pivot_table(index="구", columns="인접도로폭", values="숙소ID", aggfunc="count", fill_value=0)
        .reset_index()
        .rename_axis(None, axis=1)
    )

    pd.concat([pd.DataFrame(rows), width_counts], ignore_index=True).to_csv(
        OUT_SUMMARY, index=False, encoding="utf-8-sig"
    )
    gu_width.to_csv(
        OUT_SUMMARY.with_name("서울10구_숙소_도로폭_구별분포.csv"),
        index=False,
        encoding="utf-8-sig",
    )


def main() -> int:
    lodging = read_lodging()
    roads = load_roads()
    features, joined = attach_features(lodging, roads)

    features.to_csv(OUT_FEATURES, index=False, encoding="utf-8-sig")
    joined.to_csv(OUT_JOINED, index=False, encoding="utf-8-sig")
    save_summary(features)

    print(f"숙소: {len(lodging):,}개")
    print(f"도로폭 후보 폴리곤: {len(roads):,}개")
    print(f"도로폭 매칭: {features['인접도로폭'].notna().sum():,}개")
    print(f"평균 도로까지 거리: {features['도로폭_거리m'].mean():.1f}m")
    print(f"50m 초과: {(features['도로폭_거리m'] > 50).sum():,}개")
    print(f"100m 초과: {(features['도로폭_거리m'] > 100).sum():,}개")
    print("\n인접도로폭 분포:")
    print(features["인접도로폭"].value_counts().to_string())
    print(f"\n저장: {OUT_FEATURES}")
    print(f"저장: {OUT_JOINED}")
    print(f"저장: {OUT_SUMMARY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
