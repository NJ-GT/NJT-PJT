# -*- coding: utf-8 -*-
import pandas as pd, sys
sys.stdout.reconfigure(encoding='utf-8')
df = pd.read_csv('data/서울10구_숙소_소방거리_유클리드.csv', encoding='utf-8-sig', on_bad_lines='skip')
print(df.columns.tolist())
print(df[['기타구조','노후도_점수','승인연도']].head(10).to_string())
print('\n승인연도 결측:', df['승인연도'].isna().sum())
print('기타구조 결측:', df['기타구조'].isna().sum())
print(df['노후도_점수'].describe())
