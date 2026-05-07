# -*- coding: utf-8 -*-
"""
소방청 특정소방대상물 CSV 2개에 X좌표/Y좌표(EPSG:5174) 추가
  1. 소방청_특정소방대상물정보서비스.csv    - X좌표/Y좌표 컬럼 있음, 빈값만 채우기
  2. 소방청_특정소방대상물소방시설정보서비스.csv - 좌표 컬럼 없음, 신규 추가

Kakao 주소검색 -> 실패시 키워드검색(대상물명+구명) -> WGS84 -> EPSG:5174 변환
"""
import pandas as pd
import requests
import time
import os
from pyproj import Transformer

BASE = os.path.join(os.path.dirname(__file__), '..')
KAKAO_KEY = '96172db4c3b086f76853ed89242acefa'
# WGS84(위경도) → EPSG:5174(한국 TM 서부) 좌표계 변환기
transformer = Transformer.from_crs('EPSG:4326', 'EPSG:5174', always_xy=True)

coord_cache = {}  # 동일 주소를 두 번 API 조회하지 않도록 결과 캐시

def kakao_address(addr):
    """도로명/지번 주소로 좌표 검색"""
    try:
        r = requests.get('https://dapi.kakao.com/v2/local/search/address.json',
            headers={'Authorization': f'KakaoAK {KAKAO_KEY}'},
            params={'query': addr.strip(), 'size': 1}, timeout=5)
        docs = r.json().get('documents', [])
        if docs:
            lon, lat = float(docs[0]['x']), float(docs[0]['y'])
            x, y = transformer.transform(lon, lat)
            return round(x, 4), round(y, 4)
    except Exception:
        pass
    return None, None

def kakao_keyword(query):
    """키워드(건물명+주소)로 좌표 검색"""
    try:
        r = requests.get('https://dapi.kakao.com/v2/local/search/keyword.json',
            headers={'Authorization': f'KakaoAK {KAKAO_KEY}'},
            params={'query': query.strip(), 'size': 1}, timeout=5)
        docs = r.json().get('documents', [])
        if docs:
            lon, lat = float(docs[0]['x']), float(docs[0]['y'])
            x, y = transformer.transform(lon, lat)
            return round(x, 4), round(y, 4)
    except Exception:
        pass
    return None, None

def get_coords(addr, name, gu):
    """주소·이름·구명 조합으로 좌표를 반환한다. 캐시 우선, 주소검색 → 키워드검색 순으로 시도."""
    addr = str(addr).strip() if pd.notna(addr) else ''
    name = str(name).strip() if pd.notna(name) else ''
    gu   = str(gu).strip()   if pd.notna(gu)   else ''

    # 캐시 키: 주소가 있으면 주소, 없으면 이름|구명 조합
    cache_key = addr or f'{name}|{gu}'
    if cache_key in coord_cache:
        return coord_cache[cache_key]

    x, y = None, None

    # 1차: 정확한 주소로 검색 (가장 신뢰도 높음)
    if addr:
        x, y = kakao_address(addr)

    # 2차: 대상물명 + 구명 키워드로 검색 (주소 검색 실패 시 폴백)
    if (not x) and name:
        query = f'{name} {gu}' if gu else name
        x, y = kakao_keyword(query)

    coord_cache[cache_key] = (x, y)
    time.sleep(0.06)  # API 과호출 방지를 위한 딜레이
    return x, y


# ══════════════════════════════════════════════════════════
# [1] 소방청_특정소방대상물정보서비스.csv
#     X좌표/Y좌표 컬럼 존재 → 빈값만 채우기
# ══════════════════════════════════════════════════════════
print('=== [1] 특정소방대상물정보서비스 ===')
path1 = os.path.join(BASE, '\uc18c\ubc29\uccad_\ud2b9\uc815\uc18c\ubc29\ub300\uc0c1\ubb3c\uc815\ubcf4\uc11c\ube44\uc2a4.csv')
df1 = pd.read_csv(path1, encoding='utf-8-sig', skipinitialspace=True, dtype=str)
df1.columns = df1.columns.str.strip()
df1 = df1.applymap(lambda x: x.strip() if isinstance(x, str) else x)
print(f'로드: {len(df1)}행')

# 빈값 파악
need1 = df1['X\uc88c\ud45c'].isin(['', 'nan']) | df1['X\uc88c\ud45c'].isna()
print(f'좌표 없는 행: {need1.sum()}개')

filled1 = 0
for idx in df1[need1].index:
    addr = df1.at[idx, '\uae30\ubcf8\uc8fc\uc18c']
    name = df1.at[idx, '\ub300\uc0c1\ubb3c\uba85']
    gu   = df1.at[idx, '\uc2dc\uad70\uad6c\uba85']
    x, y = get_coords(addr, name, gu)
    if x:
        df1.at[idx, 'X\uc88c\ud45c'] = str(x)
        df1.at[idx, 'Y\uc88c\ud45c'] = str(y)
        filled1 += 1
    if (filled1) % 100 == 0 and filled1 > 0:
        print(f'  진행: {filled1}/{need1.sum()}')

print(f'채운 행: {filled1}개 / {need1.sum()}개')
df1.to_csv(path1, index=False, encoding='utf-8-sig')
print(f'저장: {path1}')


# ══════════════════════════════════════════════════════════
# [2] 소방청_특정소방대상물소방시설정보서비스.csv
#     좌표 컬럼 없음 → X좌표/Y좌표 신규 추가
# ══════════════════════════════════════════════════════════
print('\n=== [2] 특정소방대상물소방시설정보서비스 ===')
path2 = os.path.join(BASE, '\uc18c\ubc29\uccad_\ud2b9\uc815\uc18c\ubc29\ub300\uc0c1\ubb3c\uc18c\ubc29\uc2dc\uc124\uc815\ubcf4\uc11c\ube44\uc2a4.csv')
df2 = pd.read_csv(path2, encoding='utf-8-sig', skipinitialspace=True, dtype=str)
df2.columns = df2.columns.str.strip()
df2 = df2.applymap(lambda x: x.strip() if isinstance(x, str) else x)
print(f'로드: {len(df2)}행')

x_vals, y_vals = [], []
filled2 = 0
for i, (idx, row) in enumerate(df2.iterrows()):
    addr = row.get('\uae30\ubcf8\uc8fc\uc18c', '')
    name = row.get('\ub300\uc0c1\ubb3c\uba85', '')
    gu   = row.get('\uc2dc\uad70\uad6c\uba85', '')
    x, y = get_coords(addr, name, gu)
    x_vals.append(str(x) if x else '')
    y_vals.append(str(y) if y else '')
    if x:
        filled2 += 1
    if (i + 1) % 200 == 0:
        print(f'  진행: {i+1}/{len(df2)} (성공: {filled2})')

df2['X\uc88c\ud45c'] = x_vals
df2['Y\uc88c\ud45c'] = y_vals
print(f'좌표 획득: {filled2}개 / {len(df2)}개')

df2.to_csv(path2, index=False, encoding='utf-8-sig')
print(f'저장: {path2}')
print('\n완료')
