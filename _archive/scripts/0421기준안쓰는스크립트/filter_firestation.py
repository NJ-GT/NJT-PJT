# -*- coding: utf-8 -*-
"""
[파일 설명]
소방청 특정소방대상물 CSV에서 '좌표 있음 + 자체점검대상 여부 값 있음' 조건을
동시에 만족하는 행만 추출하여 저장하는 스크립트.

입력: 소방청_특정소방대상물정보서비스.csv  (소방 관련 건물 전체 목록)
출력: data/소방청_자체점검대상_좌표포함.csv  (좌표 + 자체점검 정보가 있는 건물만)
"""

import pandas as pd, os
BASE = os.path.join(os.path.dirname(__file__), '..')  # 스크립트 기준 상위 폴더(프로젝트 루트)
path = os.path.join(BASE, '소방청_특정소방대상물정보서비스.csv')
df = pd.read_csv(path, encoding='utf-8-sig', skipinitialspace=True, dtype=str)  # dtype=str: 모든 값을 문자열로 읽기
df.columns = df.columns.str.strip()                        # 열 이름의 앞뒤 공백 제거
df = df.map(lambda x: x.strip() if isinstance(x, str) else x)  # 모든 문자열 값 공백 제거
print(f'전체: {len(df)}행')

# 조건 1: X좌표가 비어있지 않은 행 (좌표 정보 있음)
has_coord = ~(df['X좌표'].isin(['', 'nan']) | df['X좌표'].isna())
# 조건 2: 자체점검대상여부 컬럼에 값이 있는 행
has_check = ~(df['자체점검대상여부'].isin(['', 'nan']) | df['자체점검대상여부'].isna())

# 두 조건을 동시에 만족하는 행만 추출
result = df[has_coord & has_check].copy()
print(f'좌표 있음: {has_coord.sum()}행')
print(f'자체점검대상여부 값 있음: {has_check.sum()}행')
print(f'둘 다 해당: {len(result)}행')

out = os.path.join(BASE, 'data', '소방청_자체점검대상_좌표포함.csv')
result.to_csv(out, index=False, encoding='utf-8-sig')
print(f'저장: {out}')
