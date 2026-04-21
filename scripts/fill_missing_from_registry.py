# -*- coding: utf-8 -*-
"""
=============================================================
[목적] 등기부등본_숙박업_핵심피처.csv 의 0값 채우기
       같은 대지위치(번지)에 있는 다른 등기부등본 레코드에서
       최댓값을 가져와 채움 (집합건물/아파트 세대 레코드 대응)

[입력]  data/등기부등본_숙박업_핵심피처.csv
        data/등기부등본_표제부_*.csv  (7개구 원본)

[출력]  data/등기부등본_숙박업_핵심피처.csv  (덮어쓰기)

[채우는 컬럼]
  대지면적(m2), 건축면적(m2), 지상층수, 지하층수, 승용승강기수

[사용 라이브러리]  pandas, re

[실행 순서]
  1. fill_from_api.py       (건축HUB API + Kakao 좌표)
  2. fill_missing_from_registry.py  ← 이 스크립트
  3. build_legal_limits_csv.py      (법정상한 컬럼 추가)
=============================================================
"""
import pandas as pd
import re
import os
from registry_title_loader import load_registry as load_registry_title

BASE = os.path.join(os.path.dirname(__file__), '..')

# ── 핵심피처 로드 ─────────────────────────────────────────────────────
feat_path = os.path.join(BASE, 'data', '등기부등본_숙박업_핵심피처.csv')
feat = pd.read_csv(feat_path, encoding='utf-8-sig', low_memory=False)
feat.columns = feat.columns.str.strip()
print(f'핵심피처: {len(feat)}행')

# 채울 대상 컬럼 목록: 집합건물(아파트 등) 세대 레코드는 0으로 저장된 경우가 많음
TARGET_COLS = ['대지면적(㎡)', '건축면적(㎡)', '지상층수', '지하층수', '승용승강기수']
for col in TARGET_COLS:
    feat[col] = pd.to_numeric(feat[col], errors='coerce').fillna(0)

print('채우기 전 0값:')
for col in TARGET_COLS:
    print(f'  {col}: {(feat[col]==0).sum()}개')

# ── 등기부등본 원본 전체 로드 ─────────────────────────────────────────
# prefer_merged=True: 7개구 표제부 CSV를 하나로 합친 버전 우선 사용
reg = load_registry_title(prefer_merged=True, low_memory=False)
reg.columns = reg.columns.str.strip()
print(f'등기부등본 원본: {len(reg)}행')

for col in TARGET_COLS:
    if col in reg.columns:
        reg[col] = pd.to_numeric(reg[col], errors='coerce').fillna(0)

# ── 번지 정규화 (공백/번지 제거) ──────────────────────────────────────
def norm_jibun(addr):
    """'서울특별시 강남구 역삼동 123-4번지' → '서울특별시 강남구 역삼동 123-4'"""
    if pd.isna(addr): return ''
    return re.sub(r'\s*번지$', '', str(addr).strip()).strip()

# 같은 대지위치끼리 묶기 위해 정규화 키 생성
reg['_addr'] = reg['대지위치'].apply(norm_jibun)
feat['_addr'] = feat['대지위치'].apply(norm_jibun)

# ── 주소별 최댓값 테이블 ──────────────────────────────────────────────
# 같은 번지에 여러 레코드가 있을 때, 0이 아닌 값 중 최댓값을 기준값으로 사용
addr_max = {}
for col in TARGET_COLS:
    if col in reg.columns:
        addr_max[col] = reg[reg[col] > 0].groupby('_addr')[col].max()

# ── 채우기 ────────────────────────────────────────────────────────────
filled = feat.copy()
for col in TARGET_COLS:
    if col not in addr_max:
        continue
    zero_mask = filled[col] == 0                          # 채워야 할 행
    new_vals = filled.loc[zero_mask, '_addr'].map(addr_max[col])  # 주소로 최댓값 매핑
    can_fill = zero_mask & new_vals.notna()               # 실제로 채울 수 있는 행
    filled.loc[can_fill, col] = new_vals[can_fill]
    still = (filled[col] == 0).sum()
    print(f'  {col}: {can_fill.sum()}개 채움 → 0 남음: {still}개')

filled.drop(columns=['_addr'], inplace=True)
filled.to_csv(feat_path, index=False, encoding='utf-8-sig')
print(f'\n저장: {feat_path}')
