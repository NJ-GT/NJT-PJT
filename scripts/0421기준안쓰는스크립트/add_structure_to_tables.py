# -*- coding: utf-8 -*-
import pandas as pd, sys
sys.stdout.reconfigure(encoding='utf-8')

BASE = 'c:/Users/USER/Documents/GitHub/기말공모전/NJT-PJT'
src = pd.read_csv(f'{BASE}/data/서울10구_숙소_소방거리_유클리드.csv',
                  encoding='utf-8-sig', on_bad_lines='skip')

# ── 1. 분석변수_테이블.csv: 위도/경도로 매칭 ─────────────────────────
an_path = f'{BASE}/data/분석변수_테이블.csv'
an = pd.read_csv(an_path, encoding='utf-8-sig')

src_ll = src[['위도','경도','구조_위험점수','구조_노후_상호작용']].copy()
src_ll['_key'] = src_ll['위도'].round(6).astype(str) + '|' + src_ll['경도'].round(6).astype(str)
src_ll = src_ll.drop_duplicates('_key')

an['_key'] = an['위도'].round(6).astype(str) + '|' + an['경도'].round(6).astype(str)
an = an.merge(src_ll[['_key','구조_위험점수','구조_노후_상호작용']], on='_key', how='left')
an = an.drop(columns=['_key'])
print(f'분석변수_테이블 매칭: {an["구조_위험점수"].notna().sum()}/{len(an)}')
an.to_csv(an_path, index=False, encoding='utf-8-sig')
print('  [저장 완료]')

# ── 2. 핵심서울10구_숙소_소방거리_유클리드_도로폭추가.csv ──────────────
rw_path = f'{BASE}/data/핵심서울10구_숙소_소방거리_유클리드_도로폭추가.csv'
rw = pd.read_csv(rw_path, encoding='utf-8-sig', on_bad_lines='skip')

src_key = src[['업소명','주소','구조_위험점수','구조_노후_상호작용']].copy()
src_key['_key'] = src_key['업소명'].str.strip() + '|' + src_key['주소'].str.strip()
src_key = src_key.drop_duplicates('_key')

rw['_key'] = rw['업소명'].str.strip() + '|' + rw['주소'].str.strip()
rw = rw.merge(src_key[['_key','구조_위험점수','구조_노후_상호작용']], on='_key', how='left')
rw = rw.drop(columns=['_key'])
print(f'도로폭추가 매칭: {rw["구조_위험점수"].notna().sum()}/{len(rw)}')
rw.to_csv(rw_path, index=False, encoding='utf-8-sig')
print('  [저장 완료]')
