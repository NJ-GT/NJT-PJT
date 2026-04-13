# -*- coding: utf-8 -*-
import pandas as pd, requests, os
BASE = os.path.join(os.path.dirname(__file__), '..')
KAKAO_KEY = '96172db4c3b086f76853ed89242acefa'
path = os.path.join(BASE, 'data', '\ub4f1\uae30\ubd80\ub4f1\ubcf8_\uc219\ubc15\uc5c5_\ud575\uc2ec\ud53c\ucc98_\ubc95\uc815\uc0c1\ud55c.csv')
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
