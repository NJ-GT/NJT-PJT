"""
집계구별 화재위험도 계산 + 소방안전센터 위치 추출
- 숙박시설 CSV: 노후도, 건폐율, 용적률, 층수
- 화재출동 CSV: 소방안전센터 위치(출동 중심점 추산), 소방서 위치
→ oa_density.json 업데이트 + firestation_data.json 생성
"""
import sys, json, pandas as pd, numpy as np, geopandas as gpd
from pyproj import Transformer
sys.stdout.reconfigure(encoding='utf-8')

# ─── 1. 숙박시설 CSV 로드 ───────────────────────────────────────
print("1. 숙박시설 데이터 로드...")
df = pd.read_csv('data/통합숙박시설최종안0415.csv', encoding='utf-8-sig')
cols = df.columns.tolist()

tf = Transformer.from_crs('EPSG:5181', 'EPSG:4326', always_xy=True)
xs, ys = tf.transform(df[cols[0]].values, df[cols[1]].values)
df['lng'] = xs
df['lat'] = ys
df['연면적']  = pd.to_numeric(df[cols[11]], errors='coerce')
df['층수']    = pd.to_numeric(df[cols[16]], errors='coerce')
df['사용승인일'] = pd.to_numeric(df[cols[18]], errors='coerce')
df['용적률']  = pd.to_numeric(df[cols[31]], errors='coerce')
df['건폐율']  = pd.to_numeric(df[cols[32]], errors='coerce')
df['건축연도'] = df['사용승인일'].apply(
    lambda x: int(str(int(x))[:4]) if pd.notna(x) and x >= 10000 else np.nan)
df['노후연수'] = (2025 - df['건축연도']).clip(0, 120)

# ─── 2. 집계구 경계 로드 ──────────────────────────────────────────
print("2. 집계구 경계 로드...")
oa = gpd.read_file('data/bnd_oa_11_2025_2Q/bnd_oa_11_2025_2Q.shp').to_crs('EPSG:4326')
oa_m = oa.to_crs('EPSG:5179')
oa['area_ha'] = oa_m.geometry.area / 10000
gu_map = {
    '11010':'종로구','11020':'중구','11030':'용산구','11040':'성동구','11050':'광진구',
    '11060':'동대문구','11070':'중랑구','11080':'노원구','11090':'강북구','11100':'도봉구',
    '11110':'은평구','11120':'서대문구','11130':'마포구','11140':'양천구','11150':'강서구',
    '11160':'구로구','11170':'금천구','11180':'영등포구','11190':'동작구','11200':'관악구',
    '11210':'서초구','11220':'강남구','11230':'송파구','11240':'강동구','11250':'도봉구',
}
oa['gu_name'] = oa['TOT_OA_CD'].str[:5].map(gu_map).fillna('알수없음')
oa_gu  = dict(zip(oa['TOT_OA_CD'], oa['gu_name']))
oa_adm = dict(zip(oa['TOT_OA_CD'], oa['ADM_CD']))

# ─── 3. 숙박시설 → 집계구 공간결합 ───────────────────────────────
print("3. 숙박시설 공간결합...")
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['lng'], df['lat']), crs='EPSG:4326')
joined = gpd.sjoin(gdf, oa[['TOT_OA_CD', 'geometry']], how='left', predicate='within')
grp = joined.groupby('TOT_OA_CD').agg(
    avg_floors  =('층수',    'mean'),
    avg_yongjuk =('용적률',  'mean'),
    avg_geonpye =('건폐율',  'mean'),
    avg_age     =('노후연수','mean'),
    max_age     =('노후연수','max'),
    min_age     =('노후연수','min'),
).round(1)
print(f"   집계 완료: {len(grp)}개 집계구")

# ─── 4. 소방안전센터 위치 추출 (출동 중심점 추산) ─────────────────
print("4. 소방안전센터 위치 추출...")
fire = pd.read_csv('data/화재출동/화재출동_2021_2024.csv',
                   encoding='utf-8-sig', low_memory=False)
fc = fire.columns.tolist()
# col25=관할소방서, col26=출동안전센터, col33=경도, col34=위도
fire['lng'] = pd.to_numeric(fire[fc[33]], errors='coerce')
fire['lat'] = pd.to_numeric(fire[fc[34]], errors='coerce')
fire = fire.dropna(subset=['lng','lat'])
fire = fire[(fire['lng']>126.5)&(fire['lng']<127.3)&
            (fire['lat']>37.3) &(fire['lat']<37.8)]   # 서울 범위

# 관할소방서 (본서) - 출동건 중심점
hs_grp = fire.groupby(fc[25]).agg(
    lat=('lat','median'), lng=('lng','median'),
    count=('lat','count')
).reset_index().rename(columns={fc[25]:'name'})
hs_grp['type'] = '소방서'

# 출동안전센터 - 중심점
sc_grp = fire.groupby(fc[26]).agg(
    lat=('lat','median'), lng=('lng','median'),
    count=('lat','count')
).reset_index().rename(columns={fc[26]:'name'})
sc_grp['type'] = '안전센터'

stations = pd.concat([hs_grp, sc_grp], ignore_index=True)
stations = stations[stations['count'] >= 10]   # 10건 이상만 신뢰
stations_list = stations[['name','type','lat','lng','count']].to_dict('records')
print(f"   소방서: {len(hs_grp)}개, 안전센터: {len(sc_grp)}개 → 총 {len(stations_list)}개")

with open('data/firestation_data.json', 'w', encoding='utf-8') as f:
    json.dump(stations_list, f, ensure_ascii=False, separators=(',',':'))
print("   → data/firestation_data.json 저장")

# ─── 5. 화재위험점수 계산 ─────────────────────────────────────────
print("5. 화재위험점수 계산...")
MAX_AGE=80; MAX_FLOOR=20; MAX_YONG=600; MAX_GPYE=80

def fire_score(p):
    """
    화재위험점수 (0~100)
    노후도 30 + 건폐율 25 + 용적률 25 + 층수 20
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
    oid = feat['properties']['id']
    p   = feat['properties']
    p['gu_name']  = oa_gu.get(oid, '알수없음')
    p['dong_code']= oa_adm.get(oid, '00000000')[5:]
    p['oa_no']    = oid[8:]

    if oid in grp.index:
        r = grp.loc[oid]
        def sv(v): return round(float(v), 1) if pd.notna(v) else None
        p['avg_floors']  = sv(r['avg_floors'])
        p['avg_yongjuk'] = sv(r['avg_yongjuk'])
        p['avg_geonpye'] = sv(r['avg_geonpye'])
        p['avg_age']     = sv(r['avg_age'])
        p['max_age']     = int(r['max_age'])  if pd.notna(r['max_age'])  else None
        p['min_age']     = int(r['min_age'])  if pd.notna(r['min_age'])  else None
    else:
        for k in ['avg_floors','avg_yongjuk','avg_geonpye','avg_age','max_age','min_age']:
            p[k] = None

    p['fire_score'] = fire_score(p)

with open('data/oa_density.json', 'w', encoding='utf-8') as f:
    json.dump(geo, f, ensure_ascii=False, separators=(',',':'))

scores = [ft['properties']['fire_score']
          for ft in geo['features'] if ft['properties']['count'] > 0]
arr = np.array(scores)
print(f"   저장 완료 | 점수범위: {arr.min():.1f}~{arr.max():.1f} 평균:{arr.mean():.1f}")
print("Done.")
