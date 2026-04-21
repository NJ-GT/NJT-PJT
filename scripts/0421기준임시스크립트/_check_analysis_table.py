# -*- coding: utf-8 -*-
import pandas as pd, sys
sys.stdout.reconfigure(encoding='utf-8')
BASE = 'c:/Users/USER/Documents/GitHub/기말공모전/NJT-PJT'
an = pd.read_csv(f'{BASE}/data/분석변수_테이블.csv', encoding='utf-8-sig')
rw = pd.read_csv(f'{BASE}/data/서울10구_숙소_소방거리_유클리드_도로폭추가.csv', encoding='utf-8-sig', on_bad_lines='skip')
print('분석변수_테이블 컬럼:', an.columns.tolist())
print('도로폭추가 컬럼:', rw.columns.tolist())
print('\n분석변수 행수:', len(an), '| 도로폭추가 행수:', len(rw))
