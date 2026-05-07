# -*- coding: utf-8 -*-
import pandas as pd, sys
sys.stdout.reconfigure(encoding='utf-8')
src = pd.read_csv('data/통합숙박시설최종안0415.csv', encoding='utf-8-sig', on_bad_lines='skip')
# 첫번째 미매칭: 서울특별시 중구 수표로 3-8
target = '수표로 3-8'
hit = src[src['소방청_도로명주소_매칭'].fillna('').str.contains(target, regex=False)]
print(f'[{target}] 검색결과:')
print(hit[['사업장명','소방청_도로명주소_매칭','기타구조']].to_string())

# 두번째: 서울특별시 중구 충무로 24
target2 = '충무로 24'
hit2 = src[src['소방청_도로명주소_매칭'].fillna('').str.contains(target2, regex=False)]
print(f'\n[{target2}] 검색결과:')
print(hit2[['사업장명','소방청_도로명주소_매칭','기타구조']].to_string())
