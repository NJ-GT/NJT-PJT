# -*- coding: utf-8 -*-
import pandas as pd, numpy as np, re, sys
sys.stdout.reconfigure(encoding='utf-8')

BASE = 'c:/Users/USER/Documents/GitHub/기말공모전/NJT-PJT'
df  = pd.read_csv(f'{BASE}/data/서울10구_숙소_소방거리_유클리드.csv', encoding='utf-8-sig', on_bad_lines='skip')
src = pd.read_csv(f'{BASE}/data/통합숙박시설최종안0415.csv', encoding='utf-8-sig', on_bad_lines='skip')

# 도로명주소로 매칭
src_road = src[['소방청_도로명주소_매칭','기타구조']].dropna(subset=['소방청_도로명주소_매칭'])
src_road = src_road.drop_duplicates(subset='소방청_도로명주소_매칭', keep='first')

unmatched_idx = df[df['기타구조'].isna()].index
fixed = 0
for idx in unmatched_idx:
    addr = df.at[idx, '주소'].strip()
    hit = src_road[src_road['소방청_도로명주소_매칭'].str.strip() == addr]
    if not hit.empty:
        df.at[idx, '기타구조'] = hit.iloc[0]['기타구조']
        fixed += 1

print(f'도로명주소 매칭 성공: {fixed}/{len(unmatched_idx)}')

# 샘플 확인
print(df.loc[unmatched_idx, ['업소명','기타구조']].head(10).to_string())
