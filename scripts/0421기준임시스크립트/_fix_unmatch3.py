# -*- coding: utf-8 -*-
import pandas as pd, numpy as np, re, sys
sys.stdout.reconfigure(encoding='utf-8')

BASE = 'c:/Users/USER/Documents/GitHub/기말공모전/NJT-PJT'
df  = pd.read_csv(f'{BASE}/data/서울10구_숙소_소방거리_유클리드.csv', encoding='utf-8-sig', on_bad_lines='skip')
src = pd.read_csv(f'{BASE}/data/통합숙박시설최종안0415.csv', encoding='utf-8-sig', on_bad_lines='skip')

src_name = src[['사업장명','기타구조']].drop_duplicates(subset='사업장명', keep='first')

unmatched_idx = df[df['기타구조'].isna()].index
fixed = 0
for idx in unmatched_idx:
    name = df.at[idx, '업소명'].strip()
    hit = src_name[src_name['사업장명'].str.strip() == name]
    if not hit.empty:
        df.at[idx, '기타구조'] = hit.iloc[0]['기타구조']
        fixed += 1

print(f'사업장명 매칭 성공: {fixed}/{len(unmatched_idx)}')
print(f'남은 미매칭: {df["기타구조"].isna().sum()}개')
print()
print(df.loc[unmatched_idx, ['업소명','기타구조']].to_string())

# 구조_위험점수 재계산
def get_score(s):
    if pd.isna(s): return 3
    s = str(s).replace(' ', '').upper()
    if re.search(r'목조|목구조', s):                                   return 7
    if re.search(r'샌드위치|판넬|패널', s):                             return 6
    if re.search(r'경량철골', s):                                       return 5
    if re.search(r'연와|벽돌|조적|세멘|시멘트벽돌|부럭|부록', s):         return 4
    if re.search(r'일반철골|철골구조', s):                              return 3
    if re.search(r'철근콘크리트|R\.C|RC조|라멘|벽식|콘크리트', s):       return 2
    if re.search(r'SRC|철골철근콘크리트', s):                           return 1
    return 3

df['구조_위험점수'] = df['기타구조'].apply(get_score)
df['_raw'] = df['구조_위험점수'] * df['건물나이']
mn, mx = df['_raw'].min(), df['_raw'].max()
df['구조_노후_상호작용'] = ((df['_raw'] - mn) / (mx - mn)).round(4)
df = df.drop(columns=['_raw'])

df.to_csv(f'{BASE}/data/서울10구_숙소_소방거리_유클리드.csv', index=False, encoding='utf-8-sig')
print('\n[저장 완료]')
