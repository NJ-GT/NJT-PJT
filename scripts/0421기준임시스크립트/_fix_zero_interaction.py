# -*- coding: utf-8 -*-
import pandas as pd, numpy as np, glob, sys
sys.stdout.reconfigure(encoding='utf-8')

BASE = 'c:/Users/USER/Documents/GitHub/기말공모전/NJT-PJT'

# 메인 CSV 재계산
main_path = f'{BASE}/data/서울10구_숙소_소방거리_유클리드.csv'
df = pd.read_csv(main_path, encoding='utf-8-sig', on_bad_lines='skip')

age_floor = df['건물나이'].clip(lower=1)
raw = df['구조_위험점수'] * age_floor
mn, mx = raw.min(), raw.max()
normed = ((raw - mn) / (mx - mn)).round(4)
min_nonzero = normed[normed > 0].min()
normed = normed.replace(0.0, min_nonzero)
df['구조_노후_상호작용'] = normed

print(f'메인CSV 최솟값: {min_nonzero}, 0.0 남은 수: {(df["구조_노후_상호작용"]==0).sum()}')
df.to_csv(main_path, index=False, encoding='utf-8-sig')
print('  [저장] 서울10구_숙소_소방거리_유클리드.csv')

# 업데이트용 소스
src = df[['위도','경도','업소명','주소','구조_노후_상호작용']].copy()
src_ll = src[['위도','경도','구조_노후_상호작용']].drop_duplicates(['위도','경도'])
src_ll['_key'] = src_ll['위도'].round(6).astype(str) + '|' + src_ll['경도'].round(6).astype(str)

src_nm = src[['업소명','주소','구조_노후_상호작용']].copy()
src_nm['_key'] = src_nm['업소명'].str.strip() + '|' + src_nm['주소'].str.strip()
src_nm = src_nm.drop_duplicates('_key')

# glob으로 대상 파일 탐색
latlon_files = glob.glob(f'{BASE}/data/*분석변수_테이블.csv')
name_files   = glob.glob(f'{BASE}/data/*도로폭추가.csv')

print(f'\n위도경도 매칭 대상: {[p.split("/")[-1] for p in latlon_files]}')
print(f'업소명 매칭 대상: {[p.split("/")[-1] for p in name_files]}')

for path in latlon_files:
    t = pd.read_csv(path, encoding='utf-8-sig', on_bad_lines='skip')
    t['_key'] = t['위도'].round(6).astype(str) + '|' + t['경도'].round(6).astype(str)
    t = t.drop(columns=['구조_노후_상호작용'], errors='ignore')
    t = t.merge(src_ll[['_key','구조_노후_상호작용']], on='_key', how='left').drop(columns=['_key'])
    t.to_csv(path, index=False, encoding='utf-8-sig')
    print(f'  [저장] {path.split(chr(47))[-1]} | 0.0: {(t["구조_노후_상호작용"]==0).sum()}')

for path in name_files:
    t = pd.read_csv(path, encoding='utf-8-sig', on_bad_lines='skip')
    t['_key'] = t['업소명'].str.strip() + '|' + t['주소'].str.strip()
    t = t.drop(columns=['구조_노후_상호작용'], errors='ignore')
    t = t.merge(src_nm[['_key','구조_노후_상호작용']], on='_key', how='left').drop(columns=['_key'])
    t.to_csv(path, index=False, encoding='utf-8-sig')
    print(f'  [저장] {path.split(chr(47))[-1]} | 0.0: {(t["구조_노후_상호작용"]==0).sum()}')
