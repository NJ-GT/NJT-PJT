"""
[파일 설명]
숙박시설 좌표(XY.csv)를 기준으로 서울시 건물 shapefile에서
반경 50m 내의 건물 현황을 분석하는 GIS 공간 분석 스크립트.

주요 역할:
  1. 숙박시설 좌표를 EPSG:5181 → EPSG:5186으로 변환한다.
  2. 건물 shapefile의 모든 건물 중심점에 공간 인덱스(STRtree)를 생성한다.
  3. 각 숙박시설 반경 50m 안의 건물들을 용도별(주택/상업/숙박/사무/기타)로 집계한다.
  4. 요약(Summary)과 상세(Details) 두 가지 결과를 CSV로 저장한다.

입력: data/XY.csv                      (숙박시설 좌표 목록)
      gis/AL_D010_11_20260409.shp       (서울시 건물 shapefile)
출력: data/XY_GIS_Analysis_Summary.csv  (숙박시설별 주변 건물 요약)
      data/XY_GIS_Building_Details.csv  (반경 내 건물 상세 목록)
"""

import pandas as pd
import shapefile
from shapely.geometry import shape, Point
from shapely.strtree import STRtree
from pyproj import Transformer
import os
import tqdm

# 입출력 경로 상수
XY_PATH = 'data/XY.csv'                            # 숙박시설 좌표 파일
SHP_PATH = 'gis/AL_D010_11_20260409.shp'            # 서울시 건물 shapefile
SUMMARY_PATH = 'data/XY_GIS_Analysis_Summary.csv'  # 분석 요약 결과
DETAILS_PATH = 'data/XY_GIS_Building_Details.csv'  # 반경 내 건물 상세 결과
RADIUS = 50  # 분석 반경 (미터 단위)

# 건물 주용도코드(A8 필드) → 카테고리 분류
RESIDENTIAL_CODES = {'01000', '02000'}       # 주택류
COMMERCIAL_CODES = {'03000', '04000', '07000'}  # 근린생활시설, 판매시설
ACCOMMODATION_CODES = {'15000'}              # 숙박시설
OFFICE_CODES = {'14000'}                     # 업무시설

def analyze():
    """숙박시설별 반경 50m 건물 현황을 분석하여 요약·상세 CSV를 생성한다."""
    print("Loading target points...")
    # XY.csv에서 숙박시설 좌표 목록 로드
    xy_df = pd.read_csv(XY_PATH)

    # 좌표계 변환기 생성: EPSG:5181(서울 TM) → EPSG:5186(중부원점 TM)
    # always_xy=True 로 항상 (경도, 위도) 순서를 유지
    transformer = Transformer.from_crs("EPSG:5181", "EPSG:5186", always_xy=True)

    print("Transforming coordinates...")
    target_points = []
    for idx, row in xy_df.iterrows():
        # 원본 좌표를 shapefile과 같은 좌표계(5186)로 변환
        new_x, new_y = transformer.transform(row['X좌표'], row['Y좌표'])
        # Shapely Point 객체로 만들어 나중에 거리 계산에 사용
        target_points.append({
            'original_index': row['index'],
            'point': Point(new_x, new_y),
            'X': new_x,
            'Y': new_y
        })

    print("Loading GIS buildings (loading attributes)...")
    # 서울시 건물 shapefile 읽기 (인코딩: cp949 = 한글 Windows 기본 인코딩)
    sf = shapefile.Reader(SHP_PATH, encoding='cp949')
    buildings = []

    for i in tqdm.tqdm(range(len(sf)), desc="Reading Buildings"):
        rec = sf.record(i)
        usage_code = rec[8]  # A8 필드: 주용도코드 (예: '01000' = 단독주택)
        usage_name = rec[9]  # A9 필드: 주용도명 (한글 용도명)
        above_floor = rec[26]  # A26 필드: 지상층수
        below_floor = rec[27]  # A27 필드: 지하층수

        # 용도코드를 5개 카테고리로 분류
        category = '기타'
        if usage_code in RESIDENTIAL_CODES:
            category = '주택'
        elif usage_code in COMMERCIAL_CODES:
            category = '상업'
        elif usage_code in ACCOMMODATION_CODES:
            category = '숙박'
        elif usage_code in OFFICE_CODES:
            category = '사무'

        # shapefile의 도형을 Shapely 객체로 변환 후 중심점(centroid) 계산
        sh = sf.shape(i)
        geom = shape(sh.__geo_interface__)
        centroid = geom.centroid

        buildings.append({
            'centroid': centroid,
            'category': category,
            '용도': usage_name,
            '지상층수': above_floor,
            '지하층수': below_floor,
            '건물ID': rec[1]  # A1 필드: 건물 고유번호
        })

    print("Building spatial index...")
    # 모든 건물 중심점으로 R-Tree 공간 인덱스 생성
    # STRtree를 쓰면 '반경 50m 내 건물 찾기'를 전수 검색 대신 빠르게 처리할 수 있음
    building_geoms = [b['centroid'] for b in buildings]
    tree = STRtree(building_geoms)

    print("Starting detailed analysis...")
    summary_results = []
    detailed_results = []

    for tp in tqdm.tqdm(target_points, desc="Analyzing Points"):
        p = tp['point']
        # p.buffer(RADIUS): 반경 50m 원을 만들고, tree.query()로 그 원과 겹치는 건물 후보 인덱스 반환
        indices = tree.query(p.buffer(RADIUS))
        
        counts = {
            'total': 0,
            '주택': 0,
            '상업': 0,
            '숙박': 0,
            '사무': 0,
            '기타': 0
        }
        
        found_buildings = []
        for idx in indices:
            if p.distance(building_geoms[idx]) <= RADIUS:
                b = buildings[idx]
                counts['total'] += 1
                counts[b['category']] += 1
                
                # Add to detailed results
                detailed_results.append({
                    '좌표인덱스': tp['original_index'],
                    '대상_X': tp['X'],
                    '대상_Y': tp['Y'],
                    '건물_고유번호': b['건물ID'],
                    '주용도명': b['용도'],
                    '지상층수': b['지상층수'],
                    '지하층수': b['지하층수'],
                    '거리(m)': round(p.distance(b['centroid']), 2)
                })
        
        # Summary mapping
        total = counts['total']
        dominant = "없음"
        concentration = 0.0
        if total > 0:
            cats = ['주택', '상업', '숙박', '사무', '기타']
            dominant = max(cats, key=lambda c: counts[c])
            concentration = round((counts[dominant] / total) * 100, 1)

        summary_results.append({
            '인덱스': tp['original_index'],
            '보정_X': tp['X'],
            '보정_Y': tp['Y'],
            f'반경_{RADIUS}m_건물수': counts['total'],
            '주택_수': counts['주택'],
            '상업_수': counts['상업'],
            '숙박_수': counts['숙박'],
            '사무_수': counts['사무'],
            '기타_수': counts['기타'],
            '밀집도': f"{counts['total']}개", # Simple density representation
            '주요_시설군': dominant,
            '집중도(%)': concentration
        })

    print("Saving results...")
    summary_df = pd.DataFrame(summary_results)
    summary_df.to_csv(SUMMARY_PATH, index=False, encoding='utf-8-sig')
    
    details_df = pd.DataFrame(detailed_results)
    details_df.to_csv(DETAILS_PATH, index=False, encoding='utf-8-sig')
    
    print(f"Summary saved to {SUMMARY_PATH}")
    print(f"Details saved to {DETAILS_PATH}")

if __name__ == "__main__":
    analyze()
