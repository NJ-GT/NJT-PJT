"""
[파일 설명]
oa_density.json(집계구별 숙박시설 분석 결과)을 3D 지도용으로 경량화하는 스크립트.

주요 역할:
  1. 집계구 shapefile의 경계 좌표를 단순화(simplify)하여 파일 크기를 줄인다.
  2. oa_density.json에서 3D 맵에 필요한 속성만 추출한다 (slim 함수).
  3. 좌표를 소수점 4자리로 반올림하여 파일 크기를 더욱 줄인다.
  4. 최종 결과를 oa_3d.json으로 저장한다.

입력: data/bnd_oa_11_2025_2Q/bnd_oa_11_2025_2Q.shp  (집계구 경계)
      data/oa_density.json                           (집계구별 분석 데이터)
출력: data/oa_3d.json                                (경량화된 3D 맵 데이터)
"""

import sys, json, geopandas as gpd, os
sys.stdout.reconfigure(encoding='utf-8')  # 한글 출력을 위해 UTF-8 설정

# ─── 1. 집계구 경계 shapefile 로드 및 단순화 ─────────────────────────────
# to_crs('EPSG:4326') : 좌표계를 WGS84(위도/경도)로 변환 (웹 지도 표준)
oa = gpd.read_file('data/bnd_oa_11_2025_2Q/bnd_oa_11_2025_2Q.shp').to_crs('EPSG:4326')
# simplify: 폴리곤의 꼭짓점 수를 줄여 용량을 절약. 0.0003 = 약 30m 오차 허용
# preserve_topology=True : 도형이 서로 겹치거나 빈틈이 생기지 않도록 유지
oa['geometry'] = oa['geometry'].simplify(0.0003, preserve_topology=True)

# ─── 2. 집계구별 분석 데이터 로드 ─────────────────────────────────────────
with open('data/oa_density.json', encoding='utf-8') as f:
    raw = json.load(f)

# 집계구 ID를 키로, 속성 정보를 값으로 하는 딕셔너리 생성 (빠른 조회용)
prop_map = {f['properties']['id']: f['properties'] for f in raw['features']}

# ─── 3. 속성 경량화 함수 ──────────────────────────────────────────────────
def slim(p):
    """
    3D 지도에서 실제로 사용되는 속성만 남기고 나머지를 버린다.
    키 이름도 짧게 줄여 JSON 파일 크기를 최소화한다.
    p : 원본 집계구 속성 딕셔너리
    """
    return {
        'id':    p.get('id', ''),          # 집계구 고유 ID (14자리 코드)
        'gu':    p.get('gu_name', ''),     # 자치구 이름 (예: '강남구')
        'no':    p.get('oa_no', ''),       # 집계구 번호
        'cnt':   p.get('count', 0),        # 숙박시설 수
        'fl':    round(p.get('avg_floors') or 0, 1),  # 평균 층수 (None이면 0)
        'age':   round(p.get('avg_age') or 0, 1),     # 평균 건축연령 (년)
        'rat':   round(p.get('ratio', 0), 1),          # 숙박 건물 비율 (%)
        'fire':  round(p.get('fire_score', 0), 1),     # 화재위험점수 (0~100)
        'area':  round(p.get('area_ha', 0), 2),        # 집계구 면적 (ha)
    }

# ─── 4. 좌표 반올림 함수 ─────────────────────────────────────────────────
def round_geom(geom):
    """
    GeoJSON 도형의 좌표를 소수점 4자리로 반올림한다.
    소수점 4자리 = 약 11m 정밀도 (지도 시각화에 충분한 수준).
    Polygon과 MultiPolygon 두 가지 도형 타입을 모두 처리한다.
    """
    t = geom['type']
    if t == 'Polygon':
        # 외곽선(outer ring)과 구멍(inner ring)을 모두 반올림
        coords = [[[round(x,4), round(y,4)] for x,y in ring]
                  for ring in geom['coordinates']]
        return {'type': 'Polygon', 'coordinates': coords}
    elif t == 'MultiPolygon':
        # 여러 개의 폴리곤으로 구성된 경우 (예: 섬이 있는 지역)
        mp = []
        for poly in geom['coordinates']:
            rings = [[[round(x,4), round(y,4)] for x,y in ring] for ring in poly]
            mp.append(rings)
        return {'type': 'MultiPolygon', 'coordinates': mp}
    return geom  # Point 등 다른 타입은 그대로 반환

# ─── 5. GeoJSON Feature 생성 ─────────────────────────────────────────────
features = []
for _, row in oa.iterrows():
    oid = row['TOT_OA_CD']                         # 집계구 코드 (14자리)
    props = prop_map.get(oid, {'id': oid, 'count': 0})  # 분석 데이터 조회 (없으면 빈 데이터)
    geom = round_geom(row['geometry'].__geo_interface__)  # shapely 도형 → GeoJSON 딕셔너리 변환
    features.append({'type': 'Feature', 'geometry': geom, 'properties': slim(props)})

# ─── 6. 결과 저장 ────────────────────────────────────────────────────────
out = {'type': 'FeatureCollection', 'features': features}
# separators=(',',':') : 공백 없이 저장하여 파일 크기 최소화
with open('data/oa_3d.json', 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, separators=(',', ':'))

sz = os.path.getsize('data/oa_3d.json')
print(f'저장 완료: {sz//1024} KB  집계구 {len(features)}개')
