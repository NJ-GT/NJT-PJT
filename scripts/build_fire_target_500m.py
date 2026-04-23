# -*- coding: utf-8 -*-
"""
Y = log(1 + 반경500m 숙박화재수) 생성 스크립트

입력: data/핵심서울0424.csv, data/화재출동/화재출동_2021_2024.csv
출력: data/data_with_fire_targets.csv
"""
import pandas as pd
import numpy as np
import sys

sys.stdout.reconfigure(encoding='utf-8')

BASE = 'c:/Users/USER/Documents/GitHub/기말공모전/NJT-PJT/data'

# ── 1. 데이터 로드 ────────────────────────────────────────────────────
print('=== 데이터 로드 ===')
lodging = pd.read_csv(f'{BASE}/핵심서울0424.csv', encoding='utf-8-sig')
fire_raw = pd.read_csv(
    f'{BASE}/화재출동/화재출동_2021_2024.csv',
    encoding='utf-8-sig', low_memory=False
)
print(f'숙소: {len(lodging)}개')
print(f'화재전체: {len(fire_raw)}건')

# ── 2. 숙박화재 필터링 ────────────────────────────────────────────────
LODGING_TYPES = ['호텔', '모텔', '여관', '여인숙', '기타 숙박시설', '숙박공유업']
TEN_GU = ['종로구', '중구', '용산구', '성동구', '마포구',
           '강서구', '영등포구', '강남구', '서초구', '송파구']

fire = fire_raw[
    fire_raw['발화장소_소분류'].str.strip().isin(LODGING_TYPES) &
    fire_raw['발생시군구'].str.strip().isin(TEN_GU) &
    fire_raw['위도'].notna() &
    fire_raw['경도'].notna()
].copy()

print(f'숙박화재(10개구 필터): {len(fire)}건')
print(fire['발화장소_소분류'].value_counts().to_string())

# ── 3. Haversine 거리 함수 ─────────────────────────────────────────────
def haversine_matrix(lat1, lon1, lat2, lon2):
    """lat1/lon1: (N,) 숙소, lat2/lon2: (M,) 화재 → (N, M) 거리행렬(m)"""
    R = 6371000
    lat1 = np.radians(lat1)[:, None]   # (N, 1)
    lon1 = np.radians(lon1)[:, None]
    lat2 = np.radians(lat2)[None, :]   # (1, M)
    lon2 = np.radians(lon2)[None, :]

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    return R * 2 * np.arcsin(np.sqrt(a))   # (N, M)

# ── 4. 반경별 화재수 계산 ──────────────────────────────────────────────
print('\n=== 공간 매칭 중 (4246 × {}행렬) ==='.format(len(fire)))

lat_l = lodging['위도'].values
lon_l = lodging['경도'].values
lat_f = fire['위도'].values
lon_f = fire['경도'].values

dist = haversine_matrix(lat_l, lon_l, lat_f, lon_f)   # (N, M)

for r in [300, 500, 1000]:
    col = f'반경{r}m_화재수'
    lodging[col] = (dist <= r).sum(axis=1)
    lodging[f'log1p_반경{r}m'] = np.log1p(lodging[col])
    lodging[f'반경{r}m_화재발생여부'] = (lodging[col] >= 1).astype(int)
    print(f'  {r}m — 평균: {lodging[col].mean():.3f}, 0값비율: {(lodging[col]==0).mean()*100:.1f}%')

# 최근접 화재 거리
lodging['최근접화재_거리m'] = dist.min(axis=1)
lodging['최근접화재_거리_log'] = np.log1p(lodging['최근접화재_거리m'])

# ── 5. 저장 ───────────────────────────────────────────────────────────
out = f'{BASE}/data_with_fire_targets.csv'
lodging.to_csv(out, index=False, encoding='utf-8-sig')
print(f'\n[저장] {out}')
print(f'추가된 컬럼: 반경300/500/1000m 화재수, log1p, 발생여부, 최근접거리')

# ── 6. 요약 ───────────────────────────────────────────────────────────
print('\n=== 최종 Y 분포 (log1p_반경500m) ===')
y = lodging['log1p_반경500m']
print(f'  평균: {y.mean():.3f}  std: {y.std():.3f}  min: {y.min():.3f}  max: {y.max():.3f}')
print(f'  왜도: {y.skew():.3f}  (원본 왜도: {lodging["반경500m_화재수"].skew():.3f})')
