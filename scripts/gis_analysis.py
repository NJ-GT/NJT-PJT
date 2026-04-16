import pandas as pd
import shapefile
from shapely.geometry import shape, Point
from shapely.strtree import STRtree
from pyproj import Transformer
import os
import tqdm

# Paths
XY_PATH = 'data/XY.csv'
SHP_PATH = 'gis/AL_D010_11_20260409.shp'
SUMMARY_PATH = 'data/XY_GIS_Analysis_Summary.csv'
DETAILS_PATH = 'data/XY_GIS_Building_Details.csv'
RADIUS = 50 # Meters

# Building Category Mapping (A8 field)
RESIDENTIAL_CODES = {'01000', '02000'}
COMMERCIAL_CODES = {'03000', '04000', '07000'}
ACCOMMODATION_CODES = {'15000'}
OFFICE_CODES = {'14000'}

def analyze():
    print("Loading target points...")
    xy_df = pd.read_csv(XY_PATH)
    
    # Coordinate Transformation: EPSG:5181 to EPSG:5186
    transformer = Transformer.from_crs("EPSG:5181", "EPSG:5186", always_xy=True)
    
    print("Transforming coordinates...")
    target_points = []
    for idx, row in xy_df.iterrows():
        new_x, new_y = transformer.transform(row['X좌표'], row['Y좌표'])
        target_points.append({
            'original_index': row['index'],
            'point': Point(new_x, new_y),
            'X': new_x,
            'Y': new_y
        })

    print("Loading GIS buildings (loading attributes)...")
    sf = shapefile.Reader(SHP_PATH, encoding='cp949')
    buildings = []
    
    for i in tqdm.tqdm(range(len(sf)), desc="Reading Buildings"):
        rec = sf.record(i)
        usage_code = rec[8] # A8: 주용도코드
        usage_name = rec[9] # A9: 주용도명
        above_floor = rec[26] # A26: 지상층수
        below_floor = rec[27] # A27: 지하층수
        
        # Determine category
        category = '기타'
        if usage_code in RESIDENTIAL_CODES:
            category = '주택'
        elif usage_code in COMMERCIAL_CODES:
            category = '상업'
        elif usage_code in ACCOMMODATION_CODES:
            category = '숙박'
        elif usage_code in OFFICE_CODES:
            category = '사무'
            
        sh = sf.shape(i)
        geom = shape(sh.__geo_interface__)
        centroid = geom.centroid
        
        buildings.append({
            'centroid': centroid,
            'category': category,
            '용도': usage_name,
            '지상층수': above_floor,
            '지하층수': below_floor,
            '건물ID': rec[1] # A1: 고유번호
        })

    print("Building spatial index...")
    building_geoms = [b['centroid'] for b in buildings]
    tree = STRtree(building_geoms)

    print("Starting detailed analysis...")
    summary_results = []
    detailed_results = []
    
    for tp in tqdm.tqdm(target_points, desc="Analyzing Points"):
        p = tp['point']
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
