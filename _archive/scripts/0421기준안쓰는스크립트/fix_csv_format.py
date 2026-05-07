# -*- coding: utf-8 -*-
"""
[파일 설명]
등기부등본_숙박업_핵심피처_법정상한.csv의 열 이름과 값에 있는 앞뒤 공백을 제거하고
동일 경로에 덮어써 저장하는 정제 스크립트.

파싱 오류(열 이름에 공백이 섞여 있어 df['컬럼명'] 접근 실패) 해결 목적.

입력/출력: data/등기부등본_숙박업_핵심피처_법정상한.csv (동일 파일 덮어씀)
"""

import pandas as pd, os
BASE = os.path.join(os.path.dirname(__file__), '..')  # 프로젝트 루트
path = os.path.join(BASE, 'data',
    '등기부등본_숙박업_핵심피처_법정상한.csv')  # 정제할 대상 파일

df = pd.read_csv(path, encoding='utf-8-sig', skipinitialspace=True)
df.columns = df.columns.str.strip()

# 모든 문자열 컬럼 앞뒤 공백 제거
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].str.strip()

print(f'행수: {len(df)} / 컬럼수: {len(df.columns)}')
df.to_csv(path, index=False, encoding='utf-8-sig')
print(f'저장 완료: {path}')

# 검증: 기본 옵션으로 다시 읽어보기
try:
    test = pd.read_csv(path, encoding='utf-8-sig')
    print(f'검증 성공: {len(test)}행 x {len(test.columns)}컬럼')
except Exception as e:
    print(f'검증 실패: {e}')
