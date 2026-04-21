# -*- coding: utf-8 -*-
import pandas as pd, sys
sys.stdout.reconfigure(encoding='utf-8')
df  = pd.read_csv('data/서울10구_숙소_소방거리_유클리드.csv', encoding='utf-8-sig', on_bad_lines='skip')
src = pd.read_csv('data/통합숙박시설최종안0415.csv', encoding='utf-8-sig', on_bad_lines='skip')

# 사업장명+대지위치 복합키 매칭
src['_key'] = src['사업장명'].str.strip() + '|' + src['대지위치'].str.strip()
df['_key']  = df['업소명'].str.strip()   + '|' + df['주소'].str.strip()

merged = df.merge(src[['_key','기타구조']], on='_key', how='left')
matched = merged['기타구조'].notna().sum()
print(f'복합키 매칭: {matched}/{len(df)} ({matched/len(df)*100:.1f}%)')
print('미매칭 샘플:')
print(merged[merged['기타구조'].isna()][['업소명','주소']].head(5).to_string())
