# -*- coding: utf-8 -*-
"""
[파일 설명]
등기부등본_숙박업_핵심피처_법정상한.csv에서 누락된 값을 보정하는 스크립트.

주요 역할:
  1. '도로명대지위치' 컬럼이 비어 있는 행을 카카오 주소검색 API로 보정한다.
     (지번주소 → 도로명주소 변환)
  2. '기타용동' 컬럼의 null 현황을 출력한다.

입력/출력: data/등기부등본_숙박업_핵심피처_법정상한.csv (동일 파일 덮어씀)
"""

import pandas as pd, requests, os
BASE = os.path.join(os.path.dirname(__file__), '..')  # 프로젝트 루트
KAKAO_KEY = '96172db4c3b086f76853ed89242acefa'         # 카카오 REST API 키
path = os.path.join(BASE, 'data', '등기부등본_숙박업_핵심피처_법정상한.csv')
df = pd.read_csv(path, encoding='utf-8-sig', skipinitialspace=True)
df.columns = df.columns.str.strip()

# ── 1. 도로명대지위치 null 7개 채우기 ──────────────────────────────────
def get_road_addr(jibun):
    """지번주소 → 도로명주소 (Kakao 주소검색)"""
    try:
        r = requests.get('https://dapi.kakao.com/v2/local/search/address.json',
            headers={'Authorization': f'KakaoAK {KAKAO_KEY}'},
            params={'query': jibun.strip(), 'size': 1}, timeout=5)
        docs = r.json().get('documents', [])
        if docs:
            road = docs[0].get('road_address')
            if road:
                return road.get('address_name', '')
    except Exception:
        pass
    return ''

null_mask = df['도로명대지위치'].isna() | (df['도로명대지위치'].astype(str).str.strip() == '')
print(f'도로명대지위치 null: {null_mask.sum()}개')
for idx in df[null_mask].index:
    jibun = str(df.at[idx, '대지위치']).strip()
    road = get_road_addr(jibun)
    print(f'  {jibun} → {road if road else "(못 찾음)"}')
    if road:
        df.at[idx, '도로명대지위치'] = road

still_null = df['도로명대지위치'].isna().sum()
print(f'채운 후 남은 null: {still_null}개')

# ── 2. 기타용동 null 확인 ─────────────────────────────────────────────
print()
print('=== 기타용동 null 현황 ===')
null_기타 = df['기타용동'].isna() | (df['기타용동'].astype(str).str.strip().isin(['', 'nan']))
print(f'null/빈값: {null_기타.sum()}개')
if null_기타.sum() > 0:
    print(df[null_기타][['대지위치','도로명대지위치','주용도코드명','기타용동']].to_string())

df.to_csv(path, index=False, encoding='utf-8-sig')
print(f'\n저장: {path}')
