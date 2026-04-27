# -*- coding: utf-8 -*-
"""분析변수_최종테이블0423_AHP3등급비교.csv 에 EPSG:5181 평면좌표 (x_5181, y_5181) 추가"""
import glob
import sys

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

sys.stdout.reconfigure(encoding="utf-8")

SRC = glob.glob("C:/Users/USER/Documents/GitHub/*/NJT-PJT/0424/*/tables/*AHP3*.csv")[0]

df = pd.read_csv(SRC, encoding="utf-8-sig")
print(f"로드: {SRC}  ({len(df)}행)")

gdf = gpd.GeoDataFrame(
    df.copy(),
    geometry=[Point(lon, lat) for lon, lat in zip(df["경도"], df["위도"])],
    crs="EPSG:4326",
)
gdf_proj = gdf.to_crs(epsg=5181)

df["x_5181"] = gdf_proj.geometry.x.round(2).values
df["y_5181"] = gdf_proj.geometry.y.round(2).values

df.to_csv(SRC, index=False, encoding="utf-8-sig")
print(f"저장 완료: x_5181, y_5181 컬럼 추가")
print(df[["위도", "경도", "x_5181", "y_5181"]].head(3).to_string(index=False))
