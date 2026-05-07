# -*- coding: utf-8 -*-
import pandas as pd, sys
sys.stdout.reconfigure(encoding='utf-8')
df = pd.read_csv('data/서울10구_숙소_소방거리_유클리드.csv', encoding='utf-8-sig', on_bad_lines='skip')
print('노후도_점수 describe:')
print(df['노후도_점수'].describe())
print('\n승인연도 결측:', df['승인연도'].isna().sum())
print('건물나이 샘플:')
print(df[['승인연도','건물나이','노후도_점수']].head(10).to_string())

src = pd.read_csv('data/통합숙박시설최종안0415.csv', encoding='utf-8-sig', on_bad_lines='skip')
print('\n통합CSV 컬럼:', [c for c in src.columns if '구조' in c or '업소' in c or '주소' in c])
print('기타구조 결측:', src['기타구조'].isna().sum(), '/', len(src))
