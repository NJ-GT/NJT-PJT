# -*- coding: utf-8 -*-
import pandas as pd, sys
sys.stdout.reconfigure(encoding='utf-8')
src = pd.read_csv('data/통합숙박시설최종안0415.csv', encoding='utf-8-sig', on_bad_lines='skip')
df  = pd.read_csv('data/서울10구_숙소_소방거리_유클리드.csv', encoding='utf-8-sig', on_bad_lines='skip')

# 통합CSV에서 이 26개 업소명(쉼표포함)이 어떻게 저장됐는지 확인
targets = ['빅박 스테이', '쉼,-IN', 'STAY, 쉼', '석촌애']
for t in targets:
    hit = src[src['사업장명'].str.contains(t, na=False, regex=False)]
    print(f'[{t}]:')
    print(hit[['사업장명','대지위치','기타구조']].to_string())
    print()
