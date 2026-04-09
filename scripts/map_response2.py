# -*- coding: utf-8 -*-
"""
response_1775722813174.csv × 등기부등본 표제부 7개구
매핑 전략:
  1단계: 구 + 동 + 건물명  (가장 정확)
  2단계: 구 + 건물명       (동이 다르게 기재된 경우 커버)
"""
import pandas as pd
import re
import os

BASE = os.path.join(os.path.dirname(__file__), '..')

# ── 1. response 로드 ──────────────────────────────────────────────────
resp = pd.read_csv(os.path.join(BASE, 'response_1775722813174.csv'),
                   encoding='utf-8-sig', dtype=str)
print(f'response: {len(resp)}행')

# ── 2. 등기부등본 7개구 합치기 ────────────────────────────────────────
reg_files = [
    '등기부등본_표제부_강남.csv',
    '등기부등본_표제부_마포구.csv',
    '등기부등본_표제부_서초구.csv',
    '등기부등본_표제부_송파구.csv',
    '등기부등본_표제부_용산구.csv',
    '등기부등본_표제부_종로구.csv',
    '등기부등본_표제부_중구.csv',
]
dfs = []
for fname in reg_files:
    fpath = os.path.join(BASE, fname)
    for enc in ['utf-8-sig', 'cp949', 'euc-kr']:
        try:
            df = pd.read_csv(fpath, encoding=enc, low_memory=False, dtype=str)
            dfs.append(df)
            print(f'  로드: {fname} ({len(df)}행)')
            break
        except:
            continue

reg = pd.concat(dfs, ignore_index=True)
print(f'등기부등본 합계: {len(reg)}행')

# ── 3. 정규화 함수 ────────────────────────────────────────────────────
def norm_name(name):
    if pd.isna(name) or str(name).strip() == '':
        return ''
    s = str(name).strip()
    s = re.sub(r'[（(（][^)）)]*[)）)]', '', s)   # 괄호 제거
    s = re.sub(r'[^\uAC00-\uD7A3a-zA-Z0-9]', '', s)  # 특수문자 제거
    return s.lower()

def extract_gu(addr):
    if pd.isna(addr): return ''
    m = re.search(r'([가-힣]+구)', str(addr))
    return m.group(1) if m else ''

def extract_dong(addr):
    if pd.isna(addr): return ''
    # "XX동", "XX가", "XX로" 등 동 단위 추출
    m = re.search(r'([가-힣]+(?:동|가|읍|면)\d*)', str(addr))
    return m.group(1) if m else ''

# ── 4. 키 생성 ────────────────────────────────────────────────────────
# response
resp['_gu']   = resp['sggNm'].fillna('').str.strip()
resp['_dong'] = resp['bassAdres'].apply(extract_dong)
resp['_name'] = resp['objNm'].apply(norm_name)

# 등기부등본 (건물명 있는 행만)
reg['_gu']   = reg['대지위치'].apply(extract_gu)
reg['_dong'] = reg['대지위치'].apply(extract_dong)
reg['_name'] = reg['건물명'].apply(norm_name)
reg_named = reg[reg['_name'] != ''].copy()

print(f'\n등기부등본 건물명 있는 행: {len(reg_named)}개 / {len(reg)}개')

# 1단계 키: 구+동+건물명
resp['_key1'] = resp['_gu'] + '|' + resp['_dong'] + '|' + resp['_name']
reg_named['_key1'] = reg_named['_gu'] + '|' + reg_named['_dong'] + '|' + reg_named['_name']

# 2단계 키: 구+건물명
resp['_key2'] = resp['_gu'] + '|' + resp['_name']
reg_named['_key2'] = reg_named['_gu'] + '|' + reg_named['_name']

# ── 5. 1단계: 구+동+건물명 ───────────────────────────────────────────
m1 = pd.merge(resp, reg_named, on='_key1', how='inner', suffixes=('_resp','_reg'))
m1['매핑방법'] = '구+동+건물명'
matched_ids = set(m1['bildSn'])
print(f'\n[1단계] 구+동+건물명: {len(m1)}행 ({m1["bildSn"].nunique()}개 시설)')

# ── 6. 2단계: 미매칭에 대해 구+건물명 ────────────────────────────────
resp2 = resp[~resp['bildSn'].isin(matched_ids)].copy()
m2 = pd.merge(resp2, reg_named, on='_key2', how='inner', suffixes=('_resp','_reg'))
m2['매핑방법'] = '구+건물명'
print(f'[2단계] 구+건물명:    {len(m2)}행 ({m2["bildSn"].nunique()}개 시설)')

# ── 7. 결합 및 정리 ───────────────────────────────────────────────────
drop_cols = [c for c in ['_gu_resp','_dong_resp','_name_resp','_key1_resp','_key2_resp',
                          '_gu_reg', '_dong_reg', '_name_reg', '_key1_reg', '_key2_reg',
                          '_gu','_dong','_name','_key1','_key2'] ]

def clean(df):
    return df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

result = pd.concat([clean(m1), clean(m2)], ignore_index=True)
print(f'\n최종 매핑: {len(result)}행 ({result["bildSn"].nunique()}개 시설 / {len(resp)}개 중)')

# ── 8. 미매칭 ─────────────────────────────────────────────────────────
final_unmatched = resp[~resp['bildSn'].isin(set(result['bildSn']))][
    ['objNm','bassAdres','sggNm']].drop_duplicates()
print(f'미매칭: {len(final_unmatched)}개')
if len(final_unmatched) > 0:
    print(final_unmatched.to_string(index=False))

# ── 9. 저장 ───────────────────────────────────────────────────────────
out_path = os.path.join(BASE, 'data', 'response2_등기부등본_매핑.csv')
result.to_csv(out_path, index=False, encoding='utf-8-sig')
print(f'\n저장: {out_path}  ({len(result.columns)}컬럼)')
