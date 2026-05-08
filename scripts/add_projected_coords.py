# -*- coding: utf-8 -*-
"""
분석변수_최종테이블0423_AHP3등급비교.csv 에 EPSG:5181 평면좌표(x_5181, y_5181) 컬럼을 추가하는 스크립트.

배경:
    위도/경도(WGS84, EPSG:4326)는 거리 계산에 부적합하므로,
    한국 중부원점(EPSG:5181) 기반 미터 좌표계로 변환해 두면
    이후 KNN/공간조인/거리 기반 분석에서 일관된 단위(m)를 쓸 수 있다.

처리:
    1) glob 패턴으로 입력 CSV를 자동 탐색
    2) (경도, 위도) 컬럼으로 GeoDataFrame 생성 (CRS=4326)
    3) EPSG:5181 로 재투영
    4) x_5181/y_5181 컬럼을 추가하고 원본 CSV에 덮어쓰기
"""

import glob
import sys

# 공간 데이터 처리
import geopandas as gpd
import pandas as pd
# 점 geometry 생성
from shapely.geometry import Point

# Windows 콘솔 한글 깨짐 방지
sys.stdout.reconfigure(encoding="utf-8")

# 입력 CSV 자동 탐색 (gh repo 어디에 있어도 대응) — 첫 번째 매칭 결과를 사용
SRC = glob.glob("C:/Users/USER/Documents/GitHub/*/NJT-PJT/0424/*/tables/*AHP3*.csv")[0]

# CSV 로드 (UTF-8 BOM 호환)
df = pd.read_csv(SRC, encoding="utf-8-sig")
print(f"로드: {SRC}  ({len(df)}행)")

# 위경도(EPSG:4326) Point geometry로 GeoDataFrame 구성
gdf = gpd.GeoDataFrame(
    df.copy(),
    geometry=[Point(lon, lat) for lon, lat in zip(df["경도"], df["위도"])],
    crs="EPSG:4326",
)
# 한국 중부원점 평면좌표(미터 단위)로 재투영
gdf_proj = gdf.to_crs(epsg=5181)

# 변환된 좌표를 원본 df에 평면좌표 컬럼으로 부착 (소수 둘째자리 반올림)
df["x_5181"] = gdf_proj.geometry.x.round(2).values
df["y_5181"] = gdf_proj.geometry.y.round(2).values

# 같은 경로에 덮어쓰기
df.to_csv(SRC, index=False, encoding="utf-8-sig")
print("저장 완료: x_5181, y_5181 컬럼 추가")
# 결과 미리보기 — 변환이 제대로 됐는지 위경도와 함께 확인
print(df[["위도", "경도", "x_5181", "y_5181"]].head(3).to_string(index=False))
