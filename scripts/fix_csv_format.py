# -*- coding: utf-8 -*-
"""CSV 공백 제거 후 깔끔하게 재저장 (파싱 오류 해결)"""
import pandas as pd, os
BASE = os.path.join(os.path.dirname(__file__), '..')
path = os.path.join(BASE, 'data',
    '\ub4f1\uae30\ubd80\ub4f1\ubcf8_\uc219\ubc15\uc5c5_\ud575\uc2ec\ud53c\ucc98_\ubc95\uc815\uc0c1\ud55c.csv')

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
