# -*- coding: utf-8 -*-
import pandas as pd, sys
sys.stdout.reconfigure(encoding='utf-8')
df  = pd.read_csv('data/서울10구_숙소_소방거리_유클리드.csv', encoding='utf-8-sig', on_bad_lines='skip')
src = pd.read_csv('data/통합숙박시설최종안0415.csv', encoding='utf-8-sig', on_bad_lines='skip')

# 미매칭 행의 주소 vs 통합CSV 주소 비교
unmatched = df[df['기타구조'].isna()]
print('미매칭 주소 샘플:')
for addr in unmatched['주소'].head(5):
    print(f'  메인: [{addr}]')
    # 유사한 통합CSV 주소 찾기
    hit = src[src['대지위치'].str.contains(addr[:10], na=False, regex=False)]
    if not hit.empty:
        print(f'  통합: [{hit.iloc[0]["대지위치"]}] → {hit.iloc[0]["기타구조"]}')
    else:
        print(f'  통합: 없음')
