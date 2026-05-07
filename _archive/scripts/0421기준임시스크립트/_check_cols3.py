# -*- coding: utf-8 -*-
import pandas as pd, sys
sys.stdout.reconfigure(encoding='utf-8')
df  = pd.read_csv('data/서울10구_숙소_소방거리_유클리드.csv', encoding='utf-8-sig', on_bad_lines='skip')
src = pd.read_csv('data/통합숙박시설최종안0415.csv', encoding='utf-8-sig', on_bad_lines='skip')
print('메인 행수:', len(df), '| 통합 행수:', len(src))
print('메인 키 컬럼:', [c for c in df.columns if '업소' in c or '주소' in c or 'ID' in c or '인덱스' in c])
print('통합 키 컬럼:', [c for c in src.columns if '업소' in c or '주소' in c or 'ID' in c or '인덱스' in c])
print('\n메인 업소명 샘플:', df['업소명'].head(5).tolist())
print('통합 업소명 샘플:', src['업소명'].head(5).tolist())
