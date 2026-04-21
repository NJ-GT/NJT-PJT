# -*- coding: utf-8 -*-
import pandas as pd, sys
sys.stdout.reconfigure(encoding='utf-8')
df  = pd.read_csv('data/서울10구_숙소_소방거리_유클리드.csv', encoding='utf-8-sig', on_bad_lines='skip')
src = pd.read_csv('data/통합숙박시설최종안0415.csv', encoding='utf-8-sig', on_bad_lines='skip')

# 사업장명 vs 업소명 일치율 확인
match = (df['업소명'].values == src['사업장명'].values).sum()
print(f'업소명 == 사업장명 일치: {match}/{len(df)} ({match/len(df)*100:.1f}%)')
print('\n메인 업소명:', df['업소명'].head(5).tolist())
print('통합 사업장명:', src['사업장명'].head(5).tolist())
