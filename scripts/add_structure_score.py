# -*- coding: utf-8 -*-
"""
기타구조 → 구조_위험점수(1~7) + 노후도_점수 + 상호작용항 합산
→ 구조_노후_통합점수 (MinMax 0~1) 1개 컬럼으로 최종 저장

처리 흐름:
  1. 통합숙박시설최종안0415.csv에서 기타구조 병합 (복합재는 가장 취약 소재 기준)
  2. 구조_위험점수 산출 (목조=7, 샌드위치판넬=6, 경량철골=5, 조적/연와=4, 일반철골=3, RC=2, SRC=1)
  3. 상호작용항 = 구조_위험점수 × max(건물나이, 1), MinMax 정규화
  4. 구조_노후_통합점수 = 구조_위험점수(원점수) + 노후도_점수(MinMax) + 상호작용항(MinMax), 재정규화
  5. 메인 CSV 및 핵심 테이블 갱신
"""
import pandas as pd, numpy as np, re, glob, sys
sys.stdout.reconfigure(encoding='utf-8')

BASE = 'c:/Users/USER/Documents/GitHub/기말공모전/NJT-PJT'
df  = pd.read_csv(f'{BASE}/data/서울10구_숙소_소방거리_유클리드.csv',
                  encoding='utf-8-sig', on_bad_lines='skip')
src = pd.read_csv(f'{BASE}/data/통합숙박시설최종안0415.csv',
                  encoding='utf-8-sig', on_bad_lines='skip')

# ── 1. 기타구조 병합 (복합재 → 가장 취약한 소재 기준) ─────────────────
src['_key'] = src['사업장명'].str.strip() + '|' + src['대지위치'].str.strip()
df['_key']  = df['업소명'].str.strip()   + '|' + df['주소'].str.strip()
src_dedup = src[['_key','기타구조']].drop_duplicates(subset='_key', keep='first')
df = df.drop(columns=['기타구조','구조_위험점수','구조_노후_상호작용','구조_노후_통합점수'], errors='ignore')
merged = df.merge(src_dedup, on='_key', how='left')

# 미매칭(업소명에 쉼표 포함) → 사업장명 단독 매칭
src_name = src[['사업장명','기타구조']].drop_duplicates('사업장명', keep='first')
for idx in merged[merged['기타구조'].isna()].index:
    hit = src_name[src_name['사업장명'].str.strip() == merged.at[idx,'업소명'].strip()]
    if not hit.empty:
        merged.at[idx,'기타구조'] = hit.iloc[0]['기타구조']
print(f'기타구조 매칭: {merged["기타구조"].notna().sum()}/{len(merged)}')

# ── 2. 구조_위험점수 (1~7, 높을수록 취약, 복합재는 취약 소재 우선) ──────
def get_score(s):
    if pd.isna(s): return 3
    s = str(s).replace(' ', '').upper()
    if re.search(r'목조|목구조', s):                                   return 7
    if re.search(r'샌드위치|판넬|패널', s):                             return 6
    if re.search(r'경량철골', s):                                       return 5
    if re.search(r'연와|벽돌|조적|세멘|시멘트벽돌|부럭|부록', s):         return 4
    if re.search(r'일반철골|철골구조', s):                              return 3
    if re.search(r'철근콘크리트|R\.C|RC조|라멘|벽식|콘크리트', s):       return 2
    if re.search(r'SRC|철골철근콘크리트', s):                           return 1
    return 3

merged['구조_위험점수'] = merged['기타구조'].apply(get_score)

# ── 3. 상호작용항: 구조_위험점수 × max(건물나이, 1), MinMax 정규화 ──────
interact_raw = merged['구조_위험점수'] * merged['건물나이'].clip(lower=1)
mn, mx = interact_raw.min(), interact_raw.max()
interact_norm = ((interact_raw - mn) / (mx - mn)).round(4)
min_nonzero = interact_norm[interact_norm > 0].min()
merged['구조_노후_상호작용'] = interact_norm.replace(0.0, min_nonzero)

# ── 4. 구조_노후_통합점수: 3개 합산 후 MinMax 재정규화 ───────────────────
raw_combined = merged['구조_위험점수'] + merged['노후도_점수'] + merged['구조_노후_상호작용']
mn2, mx2 = raw_combined.min(), raw_combined.max()
merged['구조_노후_통합점수'] = (0.1 + (raw_combined - mn2) / (mx2 - mn2) * 0.9).round(4)

merged = merged.drop(columns=['_key'])

print('\n구조_노후_통합점수 describe:')
print(merged['구조_노후_통합점수'].describe())
print('\n샘플:')
print(merged[['업소명','기타구조','건물나이','구조_위험점수','구조_노후_통합점수']].head(8).to_string())

# ── 5. 메인 CSV 저장 ──────────────────────────────────────────────────
merged.to_csv(f'{BASE}/data/서울10구_숙소_소방거리_유클리드.csv',
              index=False, encoding='utf-8-sig')
print('\n[저장] 서울10구_숙소_소방거리_유클리드.csv')

# ── 6. 핵심 테이블들 갱신 ─────────────────────────────────────────────
src_upd = merged[['위도','경도','구조_노후_통합점수']].drop_duplicates(['위도','경도'])
src_upd['_key'] = src_upd['위도'].round(6).astype(str) + '|' + src_upd['경도'].round(6).astype(str)

for f in glob.glob(f'{BASE}/data/*.csv'):
    fname = f.replace('\\','/').split('/')[-1]
    if '핵심분析변수' not in fname and '도로폭추가' not in fname:
        continue
    t = pd.read_csv(f, encoding='utf-8-sig', on_bad_lines='skip')
    t = t.drop(columns=['구조_노후_통합점수','구조_위험점수','노후도_점수','구조_노후_상호작용'], errors='ignore')
    if '위도' in t.columns:
        t['_key'] = t['위도'].round(6).astype(str) + '|' + t['경도'].round(6).astype(str)
        t = t.merge(src_upd[['_key','구조_노후_통합점수']], on='_key', how='left').drop(columns=['_key'])
    else:
        nm = merged[['업소명','주소','구조_노후_통합점수']].copy()
        nm['_key'] = nm['업소명'].str.strip() + '|' + nm['주소'].str.strip()
        t['_key'] = t['업소명'].str.strip() + '|' + t['주소'].str.strip()
        t = t.merge(nm[['_key','구조_노후_통합점수']].drop_duplicates('_key'), on='_key', how='left').drop(columns=['_key'])
    t.to_csv(f, index=False, encoding='utf-8-sig')
    print(f'[저장] {fname}')
