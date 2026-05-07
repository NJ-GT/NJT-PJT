# -*- coding: utf-8 -*-
"""
[파일 설명]
외국인관광도시민박업 인허가 데이터와 등기부등본 표제부를 도로명 주소로 매칭하는 스크립트.

주요 역할:
  - 등기부등본 표제부의 '도로명대지위치'와 민박업의 '도로명전체주소'를 정규화 후 정확 매칭한다.
  - 정규화 시 괄호 안 동명, 호수·층수 등 불필요한 부분을 제거한다.
  - 매칭된 등기부등본 행만 추출하여 저장한다.

입력: data/03_11_04_P_외국인관광도시민박업.csv  (민박업 인허가 정보)
      등기부등본_표제부_*.csv  (여러 구의 건물 등기 정보)
출력: data/등기부등본_민박업_매칭.csv           (매칭 결과)
"""
import pandas as pd
import re
import os
from registry_title_loader import load_registry as load_registry_title

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

# ─── 1. 등기부등본 전체 로드 ───────────────────────────────────────────
reg = load_registry_title(prefer_merged=True, low_memory=False)
print(f'등기부등본 합계: {len(reg)}행')

# ─── 2. 등기부등본 주소 정규화 ────────────────────────────────────────
def normalize_reg_addr(addr):
    if pd.isna(addr):
        return ''
    addr = str(addr).strip()
    addr = re.sub(r'\s*\([^)]+\)\s*$', '', addr)  # "(동명)" 제거
    return addr.strip()

reg['_addr_norm'] = reg['도로명대지위치'].apply(normalize_reg_addr)

# ─── 3. 민박업 로드 및 주소 정규화 ──────────────────────────────────
민박 = pd.read_csv(os.path.join(DATA_DIR, '03_11_04_P_외국인관광도시민박업.csv'),
                   encoding='utf-8-sig', low_memory=False)
print(f'민박업: {len(민박)}행')

def normalize_minbak_addr(addr):
    if pd.isna(addr):
        return ''
    addr = str(addr).strip()
    addr = re.sub(r'\s*\([^)]+\)\s*$', '', addr)   # "(동명)" 제거
    # ", XXX호 / XX층 / 지하X층 / 지층" 등 호수·층수 제거
    addr = re.sub(r',\s*(지하\s*\d+층|지층|[Bb]\d+층|\d+~\d+층|\d+층|\d+동\s*\d+호|\d+호|[가-힣]+\d+호).*$', '', addr)
    addr = re.sub(r',.*$', '', addr)  # 남은 쉼표 이후 제거
    return addr.strip()

# 민박업 정규화 주소 집합
민박_addrs = set(민박['도로명전체주소'].apply(normalize_minbak_addr).dropna())
민박_addrs.discard('')
print(f'민박업 고유 주소: {len(민박_addrs)}개')

# ─── 4. 등기부등본 필터링 ─────────────────────────────────────────────
mask = reg['_addr_norm'].isin(민박_addrs)
result = reg[mask].copy()
result.drop(columns=['_addr_norm'], inplace=True)
print(f'매칭된 등기부등본 행: {len(result)}개 / {len(reg)}개')

# ─── 5. 저장 ─────────────────────────────────────────────────────────
out_path = os.path.join(DATA_DIR, '등기부등본_민박업_매칭.csv')
result.to_csv(out_path, index=False, encoding='utf-8-sig')
print(f'저장 완료: {out_path}')

# ─── 6. 미매칭 주소 샘플 확인 ────────────────────────────────────────
unmatched = 민박['도로명전체주소'].apply(normalize_minbak_addr)
unmatched_set = set(unmatched) - 민박_addrs.intersection(reg['_addr_norm'].values if False else set())
# 등기부등본에 없는 민박업 주소
reg_addr_set = set(reg['_addr_norm'])
missing = [a for a in 민박_addrs if a not in reg_addr_set]
print(f'\n등기부등본에 없는 민박업 주소: {len(missing)}개')
if missing:
    print('샘플:')
    for a in missing[:5]:
        print(f'  {a}')
