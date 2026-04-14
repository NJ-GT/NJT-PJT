# -*- coding: utf-8 -*-
"""
=============================================================
[목적] 공공데이터포털 response CSV 2개를 등기부등본 표제부와 매핑
       건물명 기반으로 숙박업 인허가 데이터와 등기부등본을 연결

[입력]  response_1775722530131.csv   (숙박업 인허가 데이터 #1)
        response_1775722813174.csv   (숙박업 인허가 데이터 #2)
        data/등기부등본_표제부_*.csv  (7개구 원본)

[출력]  data/response_등기부등본_매핑.csv    (#1 매핑 결과, 35행)
        data/response2_등기부등본_매핑.csv   (#2 매핑 결과, 237행)

[매핑 전략]
  #1: 건물명(objNm) 정규화 후 단순 inner join
  #2: 1단계 구+동+건물명, 2단계 구+건물명 (동 불일치 커버)

[사용 라이브러리]  pandas, re
=============================================================
"""
import pandas as pd
import re
import os
from registry_title_loader import load_registry as load_registry_title

BASE = os.path.join(os.path.dirname(__file__), '..')

def norm_name(name):
    """건물명 정규화: 공백·괄호·특수문자 제거, 소문자"""
    if pd.isna(name) or str(name).strip() == '':
        return ''
    s = str(name).strip()
    s = re.sub(r'[（(（][^)）)]*[)）)]', '', s)
    s = re.sub(r'[^\uAC00-\uD7A3a-zA-Z0-9]', '', s)
    return s.lower()

def extract_gu(addr):
    if pd.isna(addr): return ''
    m = re.search(r'([가-힣]+구)', str(addr))
    return m.group(1) if m else ''

def extract_dong(addr):
    if pd.isna(addr): return ''
    m = re.search(r'([가-힣]+(?:동|가|읍|면)\d*)', str(addr))
    return m.group(1) if m else ''


# ══════════════════════════════════════════════════════════════════════
# [1] response_1775722530131.csv  ×  등기부등본  (건물명 단순 매핑)
# ══════════════════════════════════════════════════════════════════════
print('\n=== [1] response_1775722530131 매핑 시작 ===')
resp1 = pd.read_csv(os.path.join(BASE, 'response_1775722530131.csv'),
                    encoding='utf-8-sig', dtype=str)
print(f'response #1: {len(resp1)}행')

reg = load_registry_title(prefer_merged=True, low_memory=False, dtype=str)
print(f'등기부등본 합계: {len(reg)}행')

resp1['_key'] = resp1['objNm'].apply(norm_name)
reg['_key']   = reg['건물명'].apply(norm_name)

resp1_valid = resp1[resp1['_key'] != '']
reg_valid   = reg[reg['_key'] != '']

merged1 = pd.merge(resp1_valid, reg_valid, on='_key', how='inner',
                   suffixes=('_resp', '_reg'))
merged1.drop(columns=['_key'], inplace=True)
print(f'매핑 결과: {len(merged1)}행')

unmatched1 = resp1_valid[~resp1_valid['_key'].isin(reg_valid['_key'])]
print(f'미매핑: {len(unmatched1)}개')

out1 = os.path.join(BASE, 'data', 'response_\ub4f1\uae30\ubd80\ub4f1\ubcf8_\ub9e4\ud551.csv')
merged1.to_csv(out1, index=False, encoding='utf-8-sig')
print(f'저장: {out1}')


# ══════════════════════════════════════════════════════════════════════
# [2] response_1775722813174.csv  ×  등기부등본  (구+동+건물명 2단계)
# ══════════════════════════════════════════════════════════════════════
print('\n=== [2] response_1775722813174 매핑 시작 ===')
resp2 = pd.read_csv(os.path.join(BASE, 'response_1775722813174.csv'),
                    encoding='utf-8-sig', dtype=str)
print(f'response #2: {len(resp2)}행')

# response 키 생성
resp2['_gu']   = resp2['sggNm'].fillna('').str.strip()
resp2['_dong'] = resp2['bassAdres'].apply(extract_dong)
resp2['_name'] = resp2['objNm'].apply(norm_name)

# 등기부등본 키 생성 (건물명 있는 행만)
reg['_gu']   = reg['대지위치'].apply(extract_gu)
reg['_dong'] = reg['대지위치'].apply(extract_dong)
reg['_name'] = reg['건물명'].apply(norm_name)
reg_named = reg[reg['_name'] != ''].copy()
print(f'등기부등본 건물명 있는 행: {len(reg_named)}개')

# 1단계: 구+동+건물명
resp2['_key1']     = resp2['_gu'] + '|' + resp2['_dong'] + '|' + resp2['_name']
reg_named['_key1'] = reg_named['_gu'] + '|' + reg_named['_dong'] + '|' + reg_named['_name']

m1 = pd.merge(resp2, reg_named, on='_key1', how='inner', suffixes=('_resp', '_reg'))
m1['매핑방법'] = '구+동+건물명'
matched_ids = set(m1['bildSn'])
print(f'[1단계] 구+동+건물명: {len(m1)}행 ({m1["bildSn"].nunique()}개)')

# 2단계: 구+건물명 (미매핑 대상)
resp2['_key2']     = resp2['_gu'] + '|' + resp2['_name']
reg_named['_key2'] = reg_named['_gu'] + '|' + reg_named['_name']

resp2_remaining = resp2[~resp2['bildSn'].isin(matched_ids)].copy()
m2 = pd.merge(resp2_remaining, reg_named, on='_key2', how='inner', suffixes=('_resp', '_reg'))
m2['매핑방법'] = '구+건물명'
print(f'[2단계] 구+건물명:   {len(m2)}행 ({m2["bildSn"].nunique()}개)')

# 결합 정리
DROP_COLS = ['_gu_resp','_dong_resp','_name_resp','_key1_resp','_key2_resp',
             '_gu_reg','_dong_reg','_name_reg','_key1_reg','_key2_reg',
             '_gu','_dong','_name','_key1','_key2']

def clean_df(df):
    return df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors='ignore')

result2 = pd.concat([clean_df(m1), clean_df(m2)], ignore_index=True)
print(f'최종 매핑: {len(result2)}행 ({result2["bildSn"].nunique()}개 시설 / {len(resp2)}개 중)')

unmatched2 = resp2[~resp2['bildSn'].isin(set(result2['bildSn']))][
    ['objNm','bassAdres','sggNm']].drop_duplicates()
print(f'미매핑: {len(unmatched2)}개')
if len(unmatched2) > 0:
    print(unmatched2.to_string(index=False))

out2 = os.path.join(BASE, 'data', 'response2_\ub4f1\uae30\ubd80\ub4f1\ubcf8_\ub9e4\ud551.csv')
result2.to_csv(out2, index=False, encoding='utf-8-sig')
print(f'저장: {out2}  ({len(result2.columns)}컬럼)')
