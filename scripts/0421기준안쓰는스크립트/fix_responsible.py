# -*- coding: utf-8 -*-
import pandas as pd, sys
sys.stdout.reconfigure(encoding='utf-8')

BASE = 'c:/Users/USER/Documents/GitHub/기말공모전/NJT-PJT/data'
CSV  = f'{BASE}/서울10구_숙소_소방거리_유클리드.csv'
FIRE_CSV = f'{BASE}/소방서_안전센터_구조대_위치정보_2025_wgs84.csv'

df   = pd.read_csv(CSV,      encoding='utf-8-sig')
fire = pd.read_csv(FIRE_CSV, encoding='utf-8-sig')

def find_responsible(gu_kw, dong_kw, ftype):
    if not dong_kw or pd.isna(dong_kw):
        return None
    mask = (
        (fire['시설유형'] == ftype) &
        fire['관할구역'].fillna('').str.contains(str(gu_kw), regex=False) &
        fire['관할구역'].fillna('').str.contains(str(dong_kw), regex=False)
    )
    hits = fire[mask]
    return ' / '.join(hits['시설명'].tolist()) if not hits.empty else None

df['담당_안전센터'] = df.apply(lambda r: find_responsible(r['구'], r['동'], '안전센터/구조대'), axis=1)
df['담당_소방서']   = df.apply(lambda r: find_responsible(r['구'], r['동'], '소방서'), axis=1)

df.to_csv(CSV, index=False, encoding='utf-8-sig')
print('[저장 완료]')

print('\n=== 담당_소방서 매칭 결과 (구별 유니크 값) ===')
print(df.groupby('구')['담당_소방서'].apply(lambda x: x.dropna().unique().tolist()).to_string())
print('\n=== 담당_안전센터 매칭 결과 (구별 유니크 값 수) ===')
print(df.groupby('구')['담당_안전센터'].nunique())
