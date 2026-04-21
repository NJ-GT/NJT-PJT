# -*- coding: utf-8 -*-
"""
on_bad_lines='skip' 로 손실된 26개 행 복구 스크립트
- 소스: 통합숙박시설최종안0415.csv (4246행)
- 버퍼: XY_GIS_Analysis_Summary.csv (4246행, 동일 순서)
- 소방서: 소방서_안전센터_구조대_위치정보_2025_wgs84.csv
"""
import pandas as pd, numpy as np, sys
from pyproj import Transformer
sys.stdout.reconfigure(encoding='utf-8')

BASE = 'c:/Users/USER/Documents/GitHub/기말공모전/NJT-PJT/data'

# ── 파일 로드 ─────────────────────────────────────────────────────────
cur  = pd.read_csv(f'{BASE}/서울10구_숙소_소방거리_유클리드.csv', encoding='utf-8-sig')
src  = pd.read_csv(f'{BASE}/통합숙박시설최종안0415.csv',       encoding='utf-8-sig')
gis  = pd.read_csv(f'{BASE}/XY_GIS_Analysis_Summary.csv',     encoding='utf-8-sig')
fire = pd.read_csv(f'{BASE}/소방서_안전센터_구조대_위치정보_2025_wgs84.csv', encoding='utf-8-sig')

print(f'현재: {len(cur)}행 | 소스: {len(src)}행 | GIS: {len(gis)}행')

# ── 누락 행 위치 찾기 (소스 기준 인덱스) ─────────────────────────────
cur_names = set(cur['업소명'].str.strip())
missing_mask = ~src['사업장명'].isin(cur_names)
missing_src  = src[missing_mask].reset_index(drop=True)
missing_gis  = gis[missing_mask].reset_index(drop=True)
print(f'누락 행: {len(missing_src)}개')

# ── 좌표 변환 EPSG:5181 → WGS84 ──────────────────────────────────────
transformer = Transformer.from_crs('EPSG:5181', 'EPSG:4326', always_xy=True)
lons, lats  = transformer.transform(missing_src['X좌표'].values,
                                    missing_src['Y좌표'].values)
missing_src = missing_src.copy()
missing_src['위도'] = lats
missing_src['경도'] = lons

# ── 서울 10개구 필터 & 구/동 추출 ────────────────────────────────────
GUGUS = ['마포구','중구','종로구','용산구','강남구','영등포구','강서구','송파구','성동구','서초구']
missing_src['구'] = missing_src['도로명대지위치'].fillna('').str.extract(r'서울특별시\s+(\S+구)')[0]
missing_src['동'] = missing_src['도로명대지위치'].fillna('').str.extract(r'(\S+동)\s')[0]

# ── Haversine 거리 계산 ───────────────────────────────────────────────
R = 6371000
def nearest_euclidean(lats, lons, station_df):
    s_lats = np.radians(station_df['위도'].values)
    s_lons = np.radians(station_df['경도'].values)
    results = []
    for lat, lon in zip(lats, lons):
        dlat = s_lats - np.radians(lat)
        dlon = s_lons - np.radians(lon)
        a = np.sin(dlat/2)**2 + np.cos(np.radians(lat))*np.cos(s_lats)*np.sin(dlon/2)**2
        dists = R * 2 * np.arcsin(np.sqrt(a))
        idx = dists.argmin()
        results.append((idx, round(dists[idx])))
    return results

SPEED_KMH = 30
fire_sc = fire[fire['시설유형'] == '안전센터/구조대'].reset_index(drop=True)
fire_fs = fire[fire['시설유형'] == '소방서'].reset_index(drop=True)

nearest_sc = nearest_euclidean(missing_src['위도'], missing_src['경도'], fire_sc)
nearest_fs = nearest_euclidean(missing_src['위도'], missing_src['경도'], fire_fs)

def find_responsible(gu_kw, dong_kw, ftype):
    if not dong_kw or pd.isna(dong_kw): return None
    mask = (
        (fire['시설유형'] == ftype) &
        fire['관할구역'].fillna('').str.contains(str(gu_kw), regex=False) &
        fire['관할구역'].fillna('').str.contains(str(dong_kw), regex=False)
    )
    hits = fire[mask]
    return ' / '.join(hits['시설명'].tolist()) if not hits.empty else None

# ── 누락 행 DataFrame 생성 ────────────────────────────────────────────
rows = []
for i, row in missing_src.iterrows():
    sc_idx, sc_dist = nearest_sc[i]
    fs_idx, fs_dist = nearest_fs[i]
    sc   = fire_sc.iloc[sc_idx]
    fs   = fire_fs.iloc[fs_idx]
    best = min(sc_dist, fs_dist)
    move_sec  = best / (SPEED_KMH / 3.6)
    total_min = round((60 + move_sec) / 60, 1)

    # 승인연도 / 건물나이
    try:
        approval_yr = int(str(row['사용승인일'])[:4])
        if approval_yr < 1900 or approval_yr > 2026: approval_yr = np.nan
    except:
        approval_yr = np.nan
    building_age = (2025 - approval_yr) if not pd.isna(approval_yr) else np.nan

    # 소방접근성_점수
    fire_access = max(0, 1 - best / 2000)

    rows.append({
        '구':               row['구'],
        '동':               row['동'],
        '업소명':            row['사업장명'],
        '주소':              row['도로명대지위치'],
        '위도':              row['위도'],
        '경도':              row['경도'],
        '최근접_안전센터':   sc['시설명'],
        '안전센터_유클리드m': sc_dist,
        '최근접_소방서':     fs['시설명'],
        '소방서_유클리드m':  fs_dist,
        '최근접_거리m':      best,
        '이동시간초':        round(move_sec),
        '예상도착분':        total_min,
        '담당_안전센터':     find_responsible(row['구'], row['동'], '안전센터/구조대'),
        '담당_소방서':       find_responsible(row['구'], row['동'], '소방서'),
        '승인연도':          approval_yr,
        '건물나이':          building_age,
        '노후도_점수':       np.nan,   # 전체 Z-score 재계산 후 채움
        '소방접근성_점수':   round(fire_access, 4),
        '반경_50m_건물수':   missing_gis.loc[i, '반경_50m_건물수'],
        '집중도(%)':         missing_gis.loc[i, '집중도(%)'],
        '주요_시설군':       missing_gis.loc[i, '주요_시설군'],
        '위험점수_PCA':      np.nan,
        '위험점수_AHP':      np.nan,
        '군집':              np.nan,
        '노후도_zscore':     np.nan,
        '고유단속지점수_50m':                      row['고유단속지점수_50m'],
        '로그_주변대비_상대위험도_고유단속지점_50m': row['로그_주변대비_상대위험도_고유단속지점_50m'],
    })

new_rows = pd.DataFrame(rows)

# ── 현재 CSV 컬럼명 정리 후 병합 ─────────────────────────────────────
cur = cur.rename(columns={
    '불법주정차_단속수_50m':    '고유단속지점수_50m',
    '불법주정차_로그위험도_50m': '로그_주변대비_상대위험도_고유단속지점_50m',
})
# 컬럼 순서 맞추기
new_rows = new_rows[cur.columns]
full = pd.concat([cur, new_rows], ignore_index=True)
print(f'병합 후: {len(full)}행')

# ── 노후도_점수 전체 재계산 (Z-score) ────────────────────────────────
z = (full['건물나이'] - full['건물나이'].mean()) / full['건물나이'].std()
z_min, z_max = z.min(), z.max()
full['노후도_zscore'] = z
full['노후도_점수'] = (0.05 + 0.95 * (z - z_min) / (z_max - z_min)).clip(0.05, 1.0).round(4)

# ── 위험점수_AHP 재계산 ───────────────────────────────────────────────
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
full['소방_위험도']   = 1 - full['소방접근성_점수']
full['밀집도_정규화'] = scaler.fit_transform(full[['반경_50m_건물수']])
full['집중도_정규화'] = full['집중도(%)'] / 100

ahp_matrix = np.array([
    [1,   2,   3,   4],
    [1/2, 1,   2,   3],
    [1/3, 1/2, 1,   2],
    [1/4, 1/3, 1/2, 1],
], dtype=float)
eigenvalues, eigenvectors = np.linalg.eig(ahp_matrix)
max_idx = np.argmax(eigenvalues.real)
ahp_weights = eigenvectors[:, max_idx].real
ahp_weights = ahp_weights / ahp_weights.sum()

X = full[['소방_위험도','노후도_점수','밀집도_정규화','집중도_정규화']].values
score = (X * ahp_weights).sum(axis=1)
full['위험점수_AHP'] = ((score - score.min()) / (score.max() - score.min()) * 100).round(2)
full = full.drop(columns=['소방_위험도','밀집도_정규화','집중도_정규화'])

# ── 군집 재할당 ───────────────────────────────────────────────────────
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
VARS = ['소방접근성_점수','노후도_점수','반경_50m_건물수','집중도(%)']
X_cl = StandardScaler().fit_transform(full[VARS].fillna(0))
best_k = int(full['군집'].dropna().max()) + 1 if full['군집'].notna().any() else 2
km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
full['군집'] = km.fit_predict(X_cl)

# ── 저장 ─────────────────────────────────────────────────────────────
out = f'{BASE}/서울10구_숙소_소방거리_유클리드.csv'
full.to_csv(out, index=False, encoding='utf-8-sig')
print(f'[저장 완료] {out}')
print(f'최종 행 수: {len(full)}')
print(full[['구','업소명','노후도_점수','위험점수_AHP']].tail(5).to_string(index=False))
