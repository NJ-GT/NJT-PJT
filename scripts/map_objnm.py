# -*- coding: utf-8 -*-
import pandas as pd
import os
import re

BASE = os.path.join(os.path.dirname(__file__), '..')

# ── 1. response CSV 로드 ─────────────────────────────────────────────
resp = pd.read_csv(os.path.join(BASE, 'response_1775722530131.csv'),
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

# ── 3. 건물명 정규화 ──────────────────────────────────────────────────
def normalize(name):
    if pd.isna(name) or str(name).strip() == '':
        return ''
    name = str(name).strip()
    name = re.sub(r'\s+', '', name)           # 공백 제거
    name = re.sub(r'[（(][^)）]*[)）]', '', name)  # 괄호 제거
    name = name.lower()
    return name

resp['_key'] = resp['objNm'].apply(normalize)
reg['_key']  = reg['건물명'].apply(normalize)

# 빈 키 제외
resp_valid = resp[resp['_key'] != '']
reg_valid  = reg[reg['_key'] != '']
print(f'response 유효 건물명: {len(resp_valid)}개')
print(f'등기부등본 유효 건물명: {len(reg_valid)}개')

# ── 4. 매핑 (inner join) ──────────────────────────────────────────────
merged = pd.merge(resp_valid, reg_valid, on='_key', how='inner', suffixes=('_resp', '_reg'))
merged.drop(columns=['_key'], inplace=True)
print(f'\n매칭 결과: {len(merged)}행')

# ── 5. 저장 ──────────────────────────────────────────────────────────
out_path = os.path.join(BASE, 'data', 'response_등기부등본_매핑.csv')
merged.to_csv(out_path, index=False, encoding='utf-8-sig')
print(f'저장: {out_path}')

# ── 6. 미매칭 확인 ───────────────────────────────────────────────────
matched_keys = set(merged['_key']) if '_key' in merged.columns else set(
    resp_valid[resp_valid['_key'].isin(reg_valid['_key'])]['_key'])
reg_keys = set(reg_valid['_key'])
unmatched = resp_valid[~resp_valid['_key'].isin(reg_keys)][['objNm','bassAdres','sggNm']].drop_duplicates()
print(f'\n미매칭 response 건물: {len(unmatched)}개')
if len(unmatched) > 0:
    print('샘플:')
    print(unmatched.head(10).to_string(index=False))
