# -*- coding: utf-8 -*-
import pandas as pd, sys
sys.stdout.reconfigure(encoding='utf-8')
df  = pd.read_csv('data/서울10구_숙소_소방거리_유클리드.csv', encoding='utf-8-sig', on_bad_lines='skip')
src = pd.read_csv('data/통합숙박시설최종안0415.csv', encoding='utf-8-sig', on_bad_lines='skip')

# 주소 기반 매칭 시도
match_addr = (df['주소'].values == src['대지위치'].values).sum()
print(f'주소 == 대지위치 일치: {match_addr}/{len(df)}')

# 메인CSV 숙소ID가 있으면 통합CSV 인덱스(행번호)와 같은지
print('\n메인 숙소ID 샘플:', df['숙소ID'].head(10).tolist())

# 통합CSV에 순번 컬럼 있는지
print('통합CSV 첫컬럼 샘플:', src.iloc[:5, 0].tolist())
