# -*- coding: utf-8 -*-
"""
빠른 동(법정동) 매칭 스크립트.

목적:
    상권 폴리곤(WKT)으로 표현된 생활인구수 데이터에 대해,
    각 상권의 중심점을 계산하고 그 중심점이 속하는 법정동을 부여한다.
    이후 동 단위로 25개월 평균 방문생활인구수를 집계해 요약 CSV를 생성한다.

처리 흐름:
    1) 원본 CSV(WKT 문자열 포함) 로드
    2) WKT -> Shapely geometry 변환 후 GeoDataFrame 구성 (WGS84 기준)
    3) 안정적인 centroid 계산을 위해 EPSG:5179(미터 단위 투영)로 변환 후
       중심점을 구하고, 다시 EPSG:4326으로 되돌린 다음 동 폴리곤과 sjoin
    4) 매칭 결과 CSV 저장 + 시간대별 합계 -> 동별 25개월 평균 요약 CSV 저장
"""

# 향후 호환성을 위해 typing 어노테이션을 문자열로 평가 (Python 3.7+)
from __future__ import annotations

# 경로 객체 사용을 위해 pathlib 사용 (OS 독립적인 경로 처리)
from pathlib import Path

# 공간 데이터(GeoDataFrame, sjoin 등) 처리를 위한 GeoPandas
import geopandas as gpd
# 일반 표 형식 데이터 처리를 위한 pandas
import pandas as pd
# WKT(Well-Known Text) 문자열을 Shapely geometry 객체로 파싱하기 위한 모듈
from shapely import wkt


# 현재 스크립트가 위치한 폴더 (예: NJT-PJT/0429/)
BASE_DIR = Path(__file__).resolve().parent
# 한 단계 위 디렉터리(NJT-PJT/) — data 폴더 접근 등에 사용
ROOT = BASE_DIR.parents[0]
# 입력 CSV: 방문 4시간 그룹별 생활인구수가 한글 컬럼명으로 정리된 파일
INPUT_PATH = (
    BASE_DIR / "새 폴더" / "날짜별_concat" / "생활인구수_한글컬럼_방문4시간그룹.csv"
)
# 법정동 경계가 포함된 GeoJSON (구/법정동명/geometry 컬럼 보유)
DONG_GEOJSON = ROOT / "data" / "법정동별_사용승인구간_공간정보0415.geojson"
# 출력 1: 동 매칭이 부여된 원본 데이터 (행 단위)
OUTPUT_PATH = (
    BASE_DIR
    / "새 폴더"
    / "날짜별_concat"
    / "생활인구수_한글컬럼_방문4시간그룹_동추정.csv"
)
# 출력 2: 동별 25개월 평균 방문생활인구수 요약
SUMMARY_PATH = (
    BASE_DIR
    / "새 폴더"
    / "날짜별_concat"
    / "동별_25개월평균_방문생활인구수_빠른추정.csv"
)

# 시간대(4시간 단위) 그룹 컬럼 목록 — 합산/평균에 사용
TIME_COLS = [
    "방문생활인구수_00_03시",
    "방문생활인구수_04_07시",
    "방문생활인구수_08_11시",
    "방문생활인구수_12_15시",
    "방문생활인구수_16_19시",
    "방문생활인구수_20_23시",
]


def main() -> None:
    """엔트리 포인트: 동 매칭 + 요약 통계까지 일괄 수행."""
    # 모든 컬럼을 문자열로 일단 읽어 들임 (WKT 파싱 안전성과 코드 컬럼 보존을 위해)
    df = pd.read_csv(INPUT_PATH, encoding="utf-8-sig", dtype=str)
    # 필수 컬럼이 없으면 즉시 실패 (데이터 스키마 가드)
    if "상권좌표내용" not in df.columns:
        raise KeyError("상권좌표내용 column not found")

    # WKT 문자열 -> Shapely geometry 객체로 변환
    geometry = df["상권좌표내용"].map(wkt.loads)
    # GeoDataFrame 구성: 좌표계는 WGS84 (위경도)
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    # 위경도 상에서의 centroid는 왜곡되므로,
    # 한국 중부원점 기반 미터 좌표계(EPSG:5179)로 변환 후 centroid를 계산하고,
    # 이후 동 폴리곤과 join하기 위해 다시 WGS84로 되돌린다.
    centroids = gdf.to_crs("EPSG:5179").geometry.centroid
    # 원본 폴리곤 geometry는 제거하고 속성만 보존
    point_gdf = gdf.drop(columns=["geometry"]).copy()
    # 중심점(geometry: Point) 기반 GeoDataFrame을 다시 만들고 WGS84로 변환
    point_gdf = gpd.GeoDataFrame(point_gdf, geometry=centroids, crs="EPSG:5179").to_crs(
        "EPSG:4326"
    )

    # 법정동 경계 로드 후 필요한 컬럼만 유지하고 좌표계를 WGS84로 통일
    dong = gpd.read_file(DONG_GEOJSON)
    dong = dong[["구", "법정동명", "geometry"]].to_crs("EPSG:4326")

    # 점이 어떤 동 폴리곤 내부에 있는지 공간 조인 (within 술어)
    # how="left" 로 매칭 실패한 점도 NaN으로 보존
    joined = gpd.sjoin(point_gdf, dong, how="left", predicate="within").drop(
        columns=["index_right"]
    )
    # 매칭된 동/구 컬럼명을 명확히 "추정_" 접두사로 표시
    joined = joined.rename(columns={"구": "추정_구", "법정동명": "추정_동"})
    # 어떤 방식으로 동을 부여했는지 기록 (감사 추적용)
    joined["추정방식"] = "상권폴리곤_중심점_법정동매칭"
    # geometry 컬럼은 CSV에 저장하지 않으므로 일반 DataFrame으로 변환
    joined = pd.DataFrame(joined.drop(columns=["geometry"]))
    # 동 매칭 결과 저장 (BOM 포함 UTF-8 — 엑셀 호환)
    joined.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    # 요약 통계용 사본 생성 — 시간대 컬럼을 숫자형으로 강제 변환
    numeric = joined.copy()
    for col in TIME_COLS:
        # 변환 실패 값은 NaN -> 0 으로 처리 (합산 안전)
        numeric[col] = pd.to_numeric(numeric[col], errors="coerce").fillna(0)

    # 1단계: (월 × 구 × 동) 단위로 시간대별 합계
    monthly_dong = (
        numeric.dropna(subset=["추정_구", "추정_동"])
        .groupby(["파일기준년월", "추정_구", "추정_동"], as_index=False)[TIME_COLS]
        .sum()
    )
    # 2단계: 월 차원 제거 후 동별 평균 — "월별 합계의 25개월 평균"이 됨
    summary = monthly_dong.groupby(["추정_구", "추정_동"], as_index=False)[
        TIME_COLS
    ].mean()
    # 시간대 평균을 모두 합산 -> 하루 단위 평균 방문생활인구수 (대표 지표)
    summary["25개월평균_방문생활인구수"] = summary[TIME_COLS].sum(axis=1)
    # 큰 값부터 정렬 (top 동을 빠르게 식별)
    summary = summary.sort_values("25개월평균_방문생활인구수", ascending=False)
    # 동별 요약 저장
    summary.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")

    # 산출물 경로와 매칭 통계 출력 (콘솔 확인용)
    print(OUTPUT_PATH)
    print(SUMMARY_PATH)
    print(
        f"rows={len(joined)} matched={joined['추정_동'].notna().sum()} unmatched={joined['추정_동'].isna().sum()}"
    )
    # 상위 20개 동 미리보기 출력
    print(summary.head(20).to_string(index=False))


# 모듈을 직접 실행했을 때만 main() 호출 (import 부작용 방지)
if __name__ == "__main__":
    main()
