# -*- coding: utf-8 -*-
import pandas as pd, sys
sys.stdout.reconfigure(encoding='utf-8')
df  = pd.read_csv('data/서울10구_숙소_소방거리_유클리드.csv', encoding='utf-8-sig', on_bad_lines='skip')
src = pd.read_csv('data/통합숙박시설최종안0415.csv', encoding='utf-8-sig', on_bad_lines='skip')

# 행 순서가 같은지: 사업장명 vs 업소명 인덱스별 비교
same = sum(df['업소명'].iloc[i] == src['사업장명'].iloc[i] for i in range(len(df)))
print(f'인덱스 순서 일치: {same}/{len(df)} ({same/len(df)*100:.1f}%)')

# 불일치 행 샘플
for i in range(len(df)):
    if df['업소명'].iloc[i] != src['사업장명'].iloc[i]:
        print(f'  [{i}] 메인={df["업소명"].iloc[i]} | 통합={src["사업장명"].iloc[i]}')
    if i > 50:
        break
