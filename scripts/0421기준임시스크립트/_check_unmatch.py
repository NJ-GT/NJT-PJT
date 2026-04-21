# -*- coding: utf-8 -*-
import pandas as pd, sys
sys.stdout.reconfigure(encoding='utf-8')
df = pd.read_csv('data/서울10구_숙소_소방거리_유클리드.csv', encoding='utf-8-sig', on_bad_lines='skip')
unmatched = df[df['기타구조'].isna()]
print(f'미매칭: {len(unmatched)}개\n')
print(unmatched[['업소명','주소','건물나이','구조_위험점수']].to_string())
