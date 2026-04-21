# -*- coding: utf-8 -*-
"""
=============================================================
[목적] 외부 API로 등기부등본_숙박업_핵심피처.csv 의 누락값 채우기

[입력]  data/등기부등본_숙박업_핵심피처.csv

[출력]  data/등기부등본_숙박업_핵심피처.csv  (덮어쓰기)

[채우는 컬럼]
  건축면적(m2), 지상층수, 지하층수, 연면적(m2), 승용승강기수, 사용승인일  ← 건축HUB API
  좌표정보X/Y(EPSG:5174)                                               ← Kakao 지오코딩

[사용 API]
  - 건축HUB: apis.data.go.kr/1613000/BldRgstHubService/getBrTitleInfo
    (시군구코드+법정동코드+본번+부번 → 건축물대장 표제부)
  - Kakao Local: dapi.kakao.com/v2/local/search/address.json
    (도로명주소 → WGS84 좌표 → EPSG:5174 변환)

[사용 라이브러리]  pandas, requests, pyproj

[실행 순서]
  1. fill_from_api.py             ← 이 스크립트
  2. fill_missing_from_registry.py
  3. build_legal_limits_csv.py
=============================================================
"""
import pandas as pd
import requests
import time
import os

BASE = os.path.join(os.path.dirname(__file__), '..')
BLDG_KEY  = '1c1ea0b782ec251d390c4d34426e6ac87281041591d929dec42b641d51098eff'
KAKAO_KEY = '96172db4c3b086f76853ed89242acefa'
BLDG_URL  = 'https://apis.data.go.kr/1613000/BldRgstHubService/getBrTitleInfo'

# ── 1. 핵심피처 로드 ──────────────────────────────────────────────────
feat = pd.read_csv(os.path.join(BASE, 'data', '등기부등본_숙박업_핵심피처.csv'),
                   encoding='utf-8-sig', low_memory=False)
print(f'핵심피처: {len(feat)}행')

num_cols = ['대지면적(㎡)', '건축면적(㎡)', '연면적(㎡)', '지상층수', '지하층수', '승용승강기수']
for col in num_cols:
    feat[col] = pd.to_numeric(feat[col], errors='coerce').fillna(0)

print('채우기 전 0/null 현황:')
for col in num_cols + ['사용승인일']:
    if col in ['사용승인일']:
        n = feat[col].isna().sum()
        print(f'  {col}: null={n}개')
    else:
        print(f'  {col}: 0={(feat[col]==0).sum()}개')
coord_null = feat['좌표정보X(EPSG5174)'].isna().sum()
print(f'  좌표: null={coord_null}개')

# ── 2. 건축물대장 API 호출 ────────────────────────────────────────────
# 건물 면적·층수·승강기수 등이 0이거나 사용승인일이 없는 행을 대상으로 API 조회
target_mask = (
    (feat['건축면적(㎡)'] == 0) | (feat['지상층수'] == 0) |
    (feat['지하층수'] == 0) | (feat['연면적(㎡)'] == 0) |
    (feat['승용승강기수'] == 0) | feat['사용승인일'].isna()
)
targets = feat[target_mask][['관리건축물대장PK','대지위치','시군구코드','법정동코드','번','지']].drop_duplicates()
print(f'\n건축물대장 API 조회 대상: {len(targets)}건')

def fetch_bldg(sigungu_cd, bjdong_cd, bun, ji):
    """건축물대장 API에서 주건축물 표제부 1건을 반환한다."""
    try:
        # 시군구코드·법정동코드·본번·부번을 각각 5·5·4·4자리 0패딩
        url = (f'{BLDG_URL}?serviceKey={BLDG_KEY}'
               f'&sigunguCd={str(sigungu_cd).zfill(5)}'
               f'&bjdongCd={str(bjdong_cd).zfill(5)}'
               f'&bun={str(int(float(bun))).zfill(4) if pd.notna(bun) else "0000"}'
               f'&ji={str(int(float(ji))).zfill(4) if pd.notna(ji) else "0000"}'
               f'&numOfRows=10&pageNo=1&_type=json')
        r = requests.get(url, timeout=10)
        items = r.json().get('response', {}).get('body', {}).get('items', {})
        if not items: return None
        item_list = items.get('item', [])
        if isinstance(item_list, dict): item_list = [item_list]  # 단일 항목이면 리스트로 감쌈
        for item in item_list:
            if item.get('mainAtchGbCdNm') == '주건축물':  # 주건축물 우선 반환
                return item
        return item_list[0] if item_list else None
    except:
        return None

api_results = {}
for i, (_, row) in enumerate(targets.iterrows()):
    pk = str(row['관리건축물대장PK'])
    item = fetch_bldg(row['시군구코드'], row['법정동코드'], row['번'], row['지'])
    if item:
        api_results[pk] = item
    if (i+1) % 100 == 0:
        print(f'  진행: {i+1}/{len(targets)}')
    time.sleep(0.05)

print(f'API 응답: {len(api_results)}건')

# ── 3. 건축물대장 값 채우기 ───────────────────────────────────────────
filled = feat.copy()
filled['관리건축물대장PK'] = filled['관리건축물대장PK'].astype(str)

# API 응답 필드명 → DataFrame 컬럼명 매핑표 (필드명, 타입, 기본값 0)
col_map = {
    '건축면적(㎡)':  ('archArea',      float, 0),
    '지상층수':      ('grndFlrCnt',    int,   0),
    '지하층수':      ('ugrndFlrCnt',   int,   0),
    '연면적(㎡)':    ('totArea',       float, 0),
    '승용승강기수':  ('rideUseElvtCnt',int,   0),
}

n_filled = {col: 0 for col in col_map}
n_승인일 = 0

for idx, row in filled.iterrows():
    pk = str(row['관리건축물대장PK'])
    if pk not in api_results:
        continue
    item = api_results[pk]

    for col, (api_field, dtype, zero_val) in col_map.items():
        if row[col] == zero_val:
            val = item.get(api_field)
            if val is not None and val != '' and float(val) > 0:
                filled.at[idx, col] = dtype(val)
                n_filled[col] += 1

    # 사용승인일
    if pd.isna(row['사용승인일']):
        val = str(item.get('useAprDay', '') or '').strip()
        if len(val) == 8 and val.isdigit():
            filled.at[idx, '사용승인일'] = float(val)
            n_승인일 += 1

print('\n[건축물대장 API] 채운 결과:')
for col, n in n_filled.items():
    print(f'  {col}: {n}개')
print(f'  사용승인일: {n_승인일}개')

# ── 4. Kakao API → 좌표 채우기 ───────────────────────────────────────
coord_null_mask = filled['좌표정보X(EPSG5174)'].isna() | filled['좌표정보Y(EPSG5174)'].isna()
coord_targets = filled[coord_null_mask][['도로명대지위치','대지위치']].drop_duplicates()
print(f'\nKakao 좌표 조회 대상: {len(coord_targets)}건')

from pyproj import Transformer
transformer = Transformer.from_crs('EPSG:4326', 'EPSG:5174', always_xy=True)

def kakao_geocode(addr):
    try:
        r = requests.get('https://dapi.kakao.com/v2/local/search/address.json',
            headers={'Authorization': f'KakaoAK {KAKAO_KEY}'},
            params={'query': addr, 'size': 1}, timeout=5)
        docs = r.json().get('documents', [])
        if docs:
            lat = float(docs[0]['y'])
            lon = float(docs[0]['x'])
            x5174, y5174 = transformer.transform(lon, lat)
            return x5174, y5174
    except:
        pass
    return None, None

kakao_results = {}
for _, row in coord_targets.iterrows():
    addr = row['도로명대지위치'] if pd.notna(row['도로명대지위치']) else row['대지위치']
    x, y = kakao_geocode(str(addr))
    if x:
        kakao_results[str(addr)] = (x, y)
    time.sleep(0.1)

print(f'Kakao 좌표 획득: {len(kakao_results)}건')

n_coord = 0
for idx, row in filled.iterrows():
    if not (pd.isna(row['좌표정보X(EPSG5174)']) or pd.isna(row['좌표정보Y(EPSG5174)'])):
        continue
    addr = row['도로명대지위치'] if pd.notna(row['도로명대지위치']) else row['대지위치']
    if str(addr) in kakao_results:
        x, y = kakao_results[str(addr)]
        filled.at[idx, '좌표정보X(EPSG5174)'] = x
        filled.at[idx, '좌표정보Y(EPSG5174)'] = y
        n_coord += 1

print(f'좌표 채운 행: {n_coord}개')

# ── 5. 저장 ───────────────────────────────────────────────────────────
out_path = os.path.join(BASE, 'data', '등기부등본_숙박업_핵심피처.csv')
filled.to_csv(out_path, index=False, encoding='utf-8-sig')
print(f'\n저장: {out_path}')

print('\n최종 현황:')
for col in num_cols:
    print(f'  {col}: 0={(filled[col]==0).sum()}개')
print(f'  사용승인일: null={filled["사용승인일"].isna().sum()}개')
print(f'  좌표: null={filled["좌표정보X(EPSG5174)"].isna().sum()}개')
