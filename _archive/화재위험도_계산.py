"""
[파일 설명]
집계구별 화재위험도 점수를 계산하고, 소방서/안전센터 위치를 추출하는 핵심 분석 스크립트.

주요 역할:
  1. 통합숙박시설 CSV에서 건축 노후도, 건폐율, 용적률, 층수를 읽어온다.
  2. 숙박시설 포인트를 집계구 경계 안으로 공간결합(spatial join)한다.
  3. 집계구별로 건축 지표를 평균 내고, 화재위험점수(0~100)를 계산한다.
     화재위험점수 = 노후도(30%) + 건폐율(25%) + 용적률(25%) + 층수(20%)
  4. 화재출동 이력 CSV에서 소방서/안전센터 위치를 추출한다.
  5. oa_density.json에 결과를 덮어써 업데이트하고, firestation_data.json을 새로 만든다.

입력: data/통합숙박시설최종안0415.csv   (숙박시설 4,246개)
      data/화재출동/화재출동_2021_2024.csv (화재출동 이력)
      data/bnd_oa_11_2025_2Q/bnd_oa_11_2025_2Q.shp (집계구 경계)
      data/oa_density.json              (기존 집계구 데이터, 덮어씀)
출력: data/oa_density.json             (화재위험점수 추가된 버전)
      data/firestation_data.json        (소방서·안전센터 위치 리스트)
"""

import sys, json, pandas as pd, numpy as np, geopandas as gpd
from pyproj import Transformer
sys.stdout.reconfigure(encoding='utf-8')  # 한글 출력 설정

# ─── 1. 숙박시설 CSV 로드 ───────────────────────────────────────
print("1. 숙박시설 데이터 로드...")
df = pd.read_csv('data/통합숙박시설최종안0415.csv', encoding='utf-8-sig')
cols = df.columns.tolist()  # 열 이름 목록 (인덱스로 접근하기 위해 저장)

# 좌표계 변환기 생성: EPSG:5181(한국 중부원점 TM) → EPSG:4326(WGS84 위도/경도)
tf = Transformer.from_crs('EPSG:5181', 'EPSG:4326', always_xy=True)
# CSV의 0번(X), 1번(Y) 열에 있는 좌표를 위도/경도로 변환
xs, ys = tf.transform(df[cols[0]].values, df[cols[1]].values)
df['lng'] = xs  # 경도 (동서 방향)
df['lat'] = ys  # 위도 (남북 방향)

# 분석에 필요한 수치형 컬럼 추출 (errors='coerce' : 변환 실패 시 NaN 처리)
df['연면적']  = pd.to_numeric(df[cols[11]], errors='coerce')  # 11번 열: 연면적(㎡)
df['층수']    = pd.to_numeric(df[cols[16]], errors='coerce')  # 16번 열: 지상층수
df['사용승인일'] = pd.to_numeric(df[cols[18]], errors='coerce')  # 18번 열: 사용승인일자(YYYYMMDD)
df['용적률']  = pd.to_numeric(df[cols[31]], errors='coerce')  # 31번 열: 용적률(%)
df['건폐율']  = pd.to_numeric(df[cols[32]], errors='coerce')  # 32번 열: 건폐율(%)

# 사용승인일(YYYYMMDD)에서 건축연도(YYYY) 추출
# 예: 19870512 → 1987, 값이 10000 미만이거나 NaN이면 결측값 처리
df['건축연도'] = df['사용승인일'].apply(
    lambda x: int(str(int(x))[:4]) if pd.notna(x) and x >= 10000 else np.nan)

# 노후연수 = 2025년 기준 경과 연수 (최소 0년, 최대 120년으로 제한)
df['노후연수'] = (2025 - df['건축연도']).clip(0, 120)

# ─── 2. 집계구 경계 로드 ──────────────────────────────────────────
print("2. 집계구 경계 로드...")
# shapefile을 WGS84 좌표계로 변환
oa = gpd.read_file('data/bnd_oa_11_2025_2Q/bnd_oa_11_2025_2Q.shp').to_crs('EPSG:4326')
# EPSG:5179(UTM-K)로 재투영하여 면적 계산 (미터 단위)
oa_m = oa.to_crs('EPSG:5179')
oa['area_ha'] = oa_m.geometry.area / 10000  # m² → ha 변환 (1ha = 10,000m²)

# 집계구 코드 앞 5자리 = 자치구 코드
gu_map = {
    '11010':'종로구','11020':'중구','11030':'용산구','11040':'성동구','11050':'광진구',
    '11060':'동대문구','11070':'중랑구','11080':'노원구','11090':'강북구','11100':'도봉구',
    '11110':'은평구','11120':'서대문구','11130':'마포구','11140':'양천구','11150':'강서구',
    '11160':'구로구','11170':'금천구','11180':'영등포구','11190':'동작구','11200':'관악구',
    '11210':'서초구','11220':'강남구','11230':'송파구','11240':'강동구','11250':'도봉구',
}
oa['gu_name'] = oa['TOT_OA_CD'].str[:5].map(gu_map).fillna('알수없음')  # 코드→구 이름 매핑

# 나중에 ID→구이름, ID→행정동코드 조회가 빠르도록 딕셔너리로 만들어 둠
oa_gu  = dict(zip(oa['TOT_OA_CD'], oa['gu_name']))
oa_adm = dict(zip(oa['TOT_OA_CD'], oa['ADM_CD']))

# ─── 3. 숙박시설 → 집계구 공간결합 ───────────────────────────────
print("3. 숙박시설 공간결합...")
# 숙박시설 데이터프레임을 좌표 기반 GeoDataFrame으로 변환
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['lng'], df['lat']), crs='EPSG:4326')
# 각 숙박시설 포인트가 어느 집계구 폴리곤 안에 있는지 매핑 (within = 내부 포함)
joined = gpd.sjoin(gdf, oa[['TOT_OA_CD', 'geometry']], how='left', predicate='within')

# 집계구별로 건축 지표를 평균 낸다
grp = joined.groupby('TOT_OA_CD').agg(
    avg_floors  =('층수',    'mean'),    # 평균 층수
    avg_yongjuk =('용적률',  'mean'),    # 평균 용적률(%)
    avg_geonpye =('건폐율',  'mean'),    # 평균 건폐율(%)
    avg_age     =('노후연수','mean'),    # 평균 건축연령
    max_age     =('노후연수','max'),     # 가장 오래된 건물의 연령
    min_age     =('노후연수','min'),     # 가장 새 건물의 연령
).round(1)
print(f"   집계 완료: {len(grp)}개 집계구")

# ─── 4. 소방안전센터 위치 추출 (출동 중심점 추산) ─────────────────
print("4. 소방안전센터 위치 추출...")
fire = pd.read_csv('data/화재출동/화재출동_2021_2024.csv',
                   encoding='utf-8-sig', low_memory=False)
fc = fire.columns.tolist()

# 경도(33번 열), 위도(34번 열)를 숫자로 변환하고 결측값 제거
fire['lng'] = pd.to_numeric(fire[fc[33]], errors='coerce')
fire['lat'] = pd.to_numeric(fire[fc[34]], errors='coerce')
fire = fire.dropna(subset=['lng','lat'])  # 좌표 없는 행 제거

# 서울 범위 밖의 이상값 제거 (위도 37.3~37.8, 경도 126.5~127.3)
fire = fire[(fire['lng']>126.5)&(fire['lng']<127.3)&
            (fire['lat']>37.3) &(fire['lat']<37.8)]

# 관할소방서(본서) : 25번 열의 소방서 이름으로 그룹화 → 출동 위치 중앙값이 소방서 위치
hs_grp = fire.groupby(fc[25]).agg(
    lat=('lat','median'), lng=('lng','median'),  # median = 중앙값 (극단값 영향 적음)
    count=('lat','count')                         # 출동 건수
).reset_index().rename(columns={fc[25]:'name'})
hs_grp['type'] = '소방서'

# 출동안전센터 : 26번 열의 안전센터 이름으로 그룹화
sc_grp = fire.groupby(fc[26]).agg(
    lat=('lat','median'), lng=('lng','median'),
    count=('lat','count')
).reset_index().rename(columns={fc[26]:'name'})
sc_grp['type'] = '안전센터'

# 두 데이터를 합치고, 출동건수 10건 이상인 곳만 신뢰성 있는 위치로 채택
stations = pd.concat([hs_grp, sc_grp], ignore_index=True)
stations = stations[stations['count'] >= 10]
stations_list = stations[['name','type','lat','lng','count']].to_dict('records')
print(f"   소방서: {len(hs_grp)}개, 안전센터: {len(sc_grp)}개 → 총 {len(stations_list)}개")

with open('data/firestation_data.json', 'w', encoding='utf-8') as f:
    json.dump(stations_list, f, ensure_ascii=False, separators=(',',':'))
print("   → data/firestation_data.json 저장")

# ─── 5. 화재위험점수 계산 ─────────────────────────────────────────
print("5. 화재위험점수 계산...")
# 각 지표의 최대값 기준 (이 값 이상이면 만점으로 처리)
MAX_AGE=80   # 건축연령 80년 이상 = 노후도 만점
MAX_FLOOR=20 # 20층 이상 = 층수 만점
MAX_YONG=600 # 용적률 600% 이상 = 만점
MAX_GPYE=80  # 건폐율 80% 이상 = 만점

def fire_score(p):
    """
    화재위험점수를 계산하여 0~100 사이의 값으로 반환한다.
    각 지표를 최대값으로 나누어 0~1로 정규화한 뒤 가중치를 곱한다.
      - 노후도 30점 : 건물이 오래될수록 화재 위험 증가
      - 건폐율 25점 : 건물이 빽빽할수록 화재 확산 위험 증가
      - 용적률 25점 : 건물이 입체적으로 클수록 화재 규모 증가
      - 층수   20점 : 높을수록 피난·소방 접근 어려움
    p : 집계구 속성 딕셔너리
    """
    age   = min(p.get('avg_age')   or 0, MAX_AGE)   / MAX_AGE   * 30
    gpye  = min(p.get('avg_geonpye') or 0, MAX_GPYE)  / MAX_GPYE  * 25
    yong  = min(p.get('avg_yongjuk') or 0, MAX_YONG)  / MAX_YONG  * 25
    floor = min(p.get('avg_floors') or 0, MAX_FLOOR) / MAX_FLOOR * 20
    return round(age + gpye + yong + floor, 1)

# ─── 6. oa_density.json 업데이트 ─────────────────────────────────
print("6. oa_density.json 업데이트...")
with open('data/oa_density.json', encoding='utf-8') as f:
    geo = json.load(f)

for feat in geo['features']:
    oid = feat['properties']['id']  # 집계구 고유 ID
    p   = feat['properties']

    # 구 이름, 법정동 코드, 집계구 번호를 ID에서 파싱하여 추가
    p['gu_name']  = oa_gu.get(oid, '알수없음')
    p['dong_code']= oa_adm.get(oid, '00000000')[5:]  # ADM_CD의 뒤 8자리가 동 코드
    p['oa_no']    = oid[8:]  # ID 앞 8자리는 구/동 코드, 나머지가 집계구 번호

    if oid in grp.index:
        # 해당 집계구에 숙박시설이 있는 경우: 집계된 건축 지표를 기입
        r = grp.loc[oid]
        def sv(v): return round(float(v), 1) if pd.notna(v) else None  # NaN이면 None 반환
        p['avg_floors']  = sv(r['avg_floors'])
        p['avg_yongjuk'] = sv(r['avg_yongjuk'])
        p['avg_geonpye'] = sv(r['avg_geonpye'])
        p['avg_age']     = sv(r['avg_age'])
        p['max_age']     = int(r['max_age'])  if pd.notna(r['max_age'])  else None
        p['min_age']     = int(r['min_age'])  if pd.notna(r['min_age'])  else None
    else:
        # 숙박시설이 없는 집계구: 모든 건축 지표를 None으로 채움
        for k in ['avg_floors','avg_yongjuk','avg_geonpye','avg_age','max_age','min_age']:
            p[k] = None

    p['fire_score'] = fire_score(p)  # 화재위험점수 계산 및 저장

with open('data/oa_density.json', 'w', encoding='utf-8') as f:
    json.dump(geo, f, ensure_ascii=False, separators=(',',':'))

# 숙박시설이 있는 집계구의 화재위험점수 통계 출력
scores = [ft['properties']['fire_score']
          for ft in geo['features'] if ft['properties']['count'] > 0]
arr = np.array(scores)
print(f"   저장 완료 | 점수범위: {arr.min():.1f}~{arr.max():.1f} 평균:{arr.mean():.1f}")
print("Done.")
