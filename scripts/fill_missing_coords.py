# -*- coding: utf-8 -*-
"""
소방청_특정소방대상물정보서비스.csv 미조회 148개 좌표 추가 채우기 +
두 CSV 모두 X/Y좌표 소수점 2자리로 정리
"""
import pandas as pd
import requests
import time
import re
import os
from pyproj import Transformer

BASE = os.path.join(os.path.dirname(__file__), '..')
KAKAO_KEY = '96172db4c3b086f76853ed89242acefa'
transformer = Transformer.from_crs('EPSG:4326', 'EPSG:5174', always_xy=True)

def kakao_address(query):
    try:
        r = requests.get('https://dapi.kakao.com/v2/local/search/address.json',
            headers={'Authorization': f'KakaoAK {KAKAO_KEY}'},
            params={'query': query.strip(), 'size': 1}, timeout=5)
        docs = r.json().get('documents', [])
        if docs:
            lon, lat = float(docs[0]['x']), float(docs[0]['y'])
            x, y = transformer.transform(lon, lat)
            return round(x, 2), round(y, 2)
    except Exception:
        pass
    return None, None

def kakao_keyword(query):
    try:
        r = requests.get('https://dapi.kakao.com/v2/local/search/keyword.json',
            headers={'Authorization': f'KakaoAK {KAKAO_KEY}'},
            params={'query': query.strip(), 'size': 1}, timeout=5)
        docs = r.json().get('documents', [])
        if docs:
            lon, lat = float(docs[0]['x']), float(docs[0]['y'])
            x, y = transformer.transform(lon, lat)
            return round(x, 2), round(y, 2)
    except Exception:
        pass
    return None, None

def extract_jibun(text, gu):
    """대상물명에서 '동/가 번지' 패턴 추출 → '서울특별시 구 동 번지' 형태로 반환"""
    if pd.isna(text): return ''
    # 예: "상암동 429-35빌라", "홍은동48-203빌라", "신사동 204-4빌라"
    m = re.search(r'([가-힣]+(?:동|가|읍|면)\d*)\s*(\d+(?:-\d+)?)', str(text))
    if m:
        dong = m.group(1)
        beonji = m.group(2)
        return f'서울특별시 {gu} {dong} {beonji}'
    return ''

def get_coords(name, bldg, gu, sido):
    """여러 전략으로 좌표 탐색"""
    gu   = str(gu).strip()   if pd.notna(gu)   else ''
    name = str(name).strip() if pd.notna(name) else ''
    bldg = str(bldg).strip() if pd.notna(bldg) else ''
    sido = str(sido).strip() if pd.notna(sido) else '서울특별시'

    # 전략 1: 대상물명에서 지번 추출 → 주소검색
    addr_from_name = extract_jibun(name, gu)
    if addr_from_name:
        x, y = kakao_address(addr_from_name)
        if x: return x, y

    # 전략 2: 건물명에서 지번 추출 → 주소검색
    addr_from_bldg = extract_jibun(bldg, gu)
    if addr_from_bldg:
        x, y = kakao_address(addr_from_bldg)
        if x: return x, y

    # 전략 3: 건물명 키워드 검색 (서울 구명 포함)
    if bldg and bldg not in ['NaN', '']:
        x, y = kakao_keyword(f'{bldg} {sido} {gu}')
        if x: return x, y

    # 전략 4: 대상물명 키워드 검색
    if name:
        # 괄호 안 주소 힌트 추출 (예: "아파트명(상암동 123)")
        m = re.search(r'[（(]([^)）]+)[)）]', name)
        if m:
            hint = m.group(1).strip()
            addr_hint = f'서울특별시 {gu} {hint}'
            x, y = kakao_address(addr_hint)
            if x: return x, y
            x, y = kakao_keyword(f'{hint} {gu}')
            if x: return x, y

        # 대상물명 자체로 키워드 검색
        x, y = kakao_keyword(f'{name} {sido} {gu}')
        if x: return x, y

    return None, None


# ══════════════════════════════════════════════════════════
# [1] 특정소방대상물정보서비스 - 미조회 148개 채우기
# ══════════════════════════════════════════════════════════
print('=== [1] 미조회 148개 좌표 채우기 ===')
path1 = os.path.join(BASE, '\uc18c\ubc29\uccad_\ud2b9\uc815\uc18c\ubc29\ub300\uc0c1\ubb3c\uc815\ubcf4\uc11c\ube44\uc2a4.csv')
df1 = pd.read_csv(path1, encoding='utf-8-sig', skipinitialspace=True, dtype=str)
df1.columns = df1.columns.str.strip()
df1 = df1.map(lambda x: x.strip() if isinstance(x, str) else x)

need_mask = df1['X\uc88c\ud45c'].isin(['', 'nan']) | df1['X\uc88c\ud45c'].isna()
print(f'채워야 할 행: {need_mask.sum()}개')

filled = 0
for idx in df1[need_mask].index:
    name = df1.at[idx, '\ub300\uc0c1\ubb3c\uba85']
    bldg = df1.at[idx, '\uac74\ubb3c\uba85']
    gu   = df1.at[idx, '\uc2dc\uad70\uad6c\uba85']
    sido = df1.at[idx, '\uc2dc\ub3c4\uba85']
    x, y = get_coords(name, bldg, gu, sido)
    if x:
        df1.at[idx, 'X\uc88c\ud45c'] = str(x)
        df1.at[idx, 'Y\uc88c\ud45c'] = str(y)
        filled += 1
    time.sleep(0.06)

print(f'추가 채운 행: {filled}개')

# 소수점 2자리 정리
for col in ['X\uc88c\ud45c', 'Y\uc88c\ud45c']:
    df1[col] = pd.to_numeric(df1[col], errors='coerce').round(2)

still_empty = df1['X\uc88c\ud45c'].isna().sum()
print(f'최종 미조회: {still_empty}개')
df1.to_csv(path1, index=False, encoding='utf-8-sig')
print(f'저장: {path1}')


# ══════════════════════════════════════════════════════════
# [2] 소방시설정보서비스 - 좌표 소수점 2자리로 정리
# ══════════════════════════════════════════════════════════
print('\n=== [2] 소방시설정보서비스 소수점 2자리 정리 ===')
path2 = os.path.join(BASE, '\uc18c\ubc29\uccad_\ud2b9\uc815\uc18c\ubc29\ub300\uc0c1\ubb3c\uc18c\ubc29\uc2dc\uc124\uc815\ubcf4\uc11c\ube44\uc2a4.csv')
df2 = pd.read_csv(path2, encoding='utf-8-sig', skipinitialspace=True, dtype=str)
df2.columns = df2.columns.str.strip()
df2 = df2.map(lambda x: x.strip() if isinstance(x, str) else x)

for col in ['X\uc88c\ud45c', 'Y\uc88c\ud45c']:
    df2[col] = pd.to_numeric(df2[col], errors='coerce').round(2)

df2.to_csv(path2, index=False, encoding='utf-8-sig')
print(f'저장: {path2}')
print('완료')
