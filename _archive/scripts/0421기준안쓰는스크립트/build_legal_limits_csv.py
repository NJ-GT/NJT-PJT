# -*- coding: utf-8 -*-
"""
=============================================================
[목적] 등기부등본_숙박업_핵심피처_법정상한.csv 생성
       3단계 파이프라인:
         1) 등기부등본 원본에서 용적률/건폐율/용적률산정연면적 컬럼 추가
            + 역산으로 대지면적/건축면적 0값 채우기
         2) land.seoul.go.kr API로 용도지역 조회
            → 건폐율_법정상한(%), 용적률_법정상한(%) 컬럼 추가
            + 남은 대지면적/건축면적 0값 법정상한으로 역산
         3) 용적률(%)/건폐율(%) 0값 → 법정상한값으로 대체

[입력]  data/등기부등본_숙박업_핵심피처.csv
        data/등기부등본_표제부_*.csv  (7개구 원본, 용적률/건폐율 출처)

[출력]  data/등기부등본_숙박업_핵심피처_법정상한.csv
        컬럼: 기존 29개 + 용도지역 + 건폐율_법정상한(%) + 용적률_법정상한(%)  = 32개

[사용 API]
  - land.seoul.go.kr/land/wskras/getKoreps00047.do  (용도지역 조회)
    param: landCd = 시군구(5)+법정동(5)+대지구분(1=일반)+본번(4)+부번(4)

[사용 라이브러리]  pandas, requests, time

[실행 순서]
  1. fill_from_api.py             (건축HUB API + Kakao 좌표)
  2. fill_missing_from_registry.py (동일주소 최댓값 채우기)
  3. build_legal_limits_csv.py    ← 이 스크립트 (최종 CSV 생성)
=============================================================
"""
import pandas as pd
import requests
import time
import os
from collections import Counter
from registry_title_loader import load_registry as load_registry_title

BASE = os.path.join(os.path.dirname(__file__), '..')

# ── 서울시 용도지역별 법정 상한값 (국토계획법 기준) ───────────────────
# 용도지역명 → 건폐율/용적률 상한(%) 딕셔너리
ZONE_LIMITS = {
    '제1종전용주거지역': {'건폐율': 50,  '용적률': 100},
    '제2종전용주거지역': {'건폐율': 40,  '용적률': 120},
    '제1종일반주거지역': {'건폐율': 60,  '용적률': 200},
    '제2종일반주거지역': {'건폐율': 60,  '용적률': 250},
    '제3종일반주거지역': {'건폐율': 50,  '용적률': 300},
    '준주거지역':        {'건폐율': 70,  '용적률': 500},
    '중심상업지역':      {'건폐율': 80,  '용적률': 1000},
    '일반상업지역':      {'건폐율': 80,  '용적률': 800},
    '근린상업지역':      {'건폐율': 70,  '용적률': 600},
    '유통상업지역':      {'건폐율': 80,  '용적률': 1100},
    '전용공업지역':      {'건폐율': 70,  '용적률': 300},
    '일반공업지역':      {'건폐율': 70,  '용적률': 350},
    '준공업지역':        {'건폐율': 70,  '용적률': 400},
    '보전녹지지역':      {'건폐율': 20,  '용적률': 80},
    '생산녹지지역':      {'건폐율': 20,  '용적률': 100},
    '자연녹지지역':      {'건폐율': 20,  '용적률': 100},
    '보전관리지역':      {'건폐율': 20,  '용적률': 80},
    '생산관리지역':      {'건폐율': 20,  '용적률': 80},
    '계획관리지역':      {'건폐율': 40,  '용적률': 100},
    '농림지역':          {'건폐율': 20,  '용적률': 80},
    '자연환경보전지역':  {'건폐율': 20,  '용적률': 80},
}

# ── 1. 핵심피처 로드 ──────────────────────────────────────────────────
feat = pd.read_csv(os.path.join(BASE, 'data', '\ub4f1\uae30\ubd80\ub4f1\ubcf8_\uc219\ubc15\uc5c5_\ud575\uc2ec\ud53c\ucc98.csv'),
                   encoding='utf-8-sig', low_memory=False)
feat.columns = feat.columns.str.strip()
print(f'핵심피처: {len(feat)}행')

for col in ['대지면적(㎡)', '건축면적(㎡)', '연면적(㎡)', '지상층수', '용적률(%)', '건폐율(%)']:
    feat[col] = pd.to_numeric(feat[col], errors='coerce').fillna(0)

# ── 2. 등기부등본 원본에서 용적률/건폐율 컬럼 추가 ───────────────────
reg = load_registry_title(prefer_merged=True, low_memory=False)
reg.columns = reg.columns.str.strip()

for col in ['용적률(%)', '건폐율(%)', '용적률산정연면적(㎡)', '대지면적(㎡)', '건축면적(㎡)']:
    reg[col] = pd.to_numeric(reg[col], errors='coerce').fillna(0)

reg['관리건축물대장PK'] = reg['관리건축물대장PK'].astype(str).str.strip()
feat['관리건축물대장PK'] = feat['관리건축물대장PK'].astype(str).str.strip()

pk_lookup = reg.set_index('관리건축물대장PK')[['용적률(%)', '건폐율(%)', '용적률산정연면적(㎡)']].drop_duplicates()
feat['용적률(%)']          = feat['관리건축물대장PK'].map(pk_lookup['용적률(%)'])
feat['건폐율(%)']          = feat['관리건축물대장PK'].map(pk_lookup['건폐율(%)'])
feat['용적률산정연면적(㎡)'] = feat['관리건축물대장PK'].map(pk_lookup['용적률산정연면적(㎡)'])

for col in ['용적률(%)', '건폐율(%)', '용적률산정연면적(㎡)']:
    feat[col] = pd.to_numeric(feat[col], errors='coerce').fillna(0)

print(f'용적률 > 0: {(feat["용적률(%)"]>0).sum()}개 / 건폐율 > 0: {(feat["건폐율(%)"]>0).sum()}개')

# ── 3. 대지면적/건축면적 역산 (등기부등본 용적률/건폐율 사용) ─────────
# 대지면적: 용적률산정연면적 / 용적률 * 100
mask = (feat['대지면적(㎡)'] == 0) & (feat['용적률(%)'] > 0) & (feat['용적률산정연면적(㎡)'] > 0)
feat.loc[mask, '대지면적(㎡)'] = (feat.loc[mask, '용적률산정연면적(㎡)'] / feat.loc[mask, '용적률(%)'] * 100).round(2)
print(f'대지면적 역산 (용적률): {mask.sum()}건')

# 대지면적: 연면적 / 지상층수 (용적률 없는 경우)
mask2 = (feat['대지면적(㎡)'] == 0) & (feat['연면적(㎡)'] > 0) & (feat['지상층수'] > 0)
feat.loc[mask2, '대지면적(㎡)'] = (feat.loc[mask2, '연면적(㎡)'] / feat.loc[mask2, '지상층수']).round(2)
print(f'대지면적 역산 (층수): {mask2.sum()}건')

# 건축면적: 연면적 / 건폐율 * 100
mask3 = (feat['건축면적(㎡)'] == 0) & (feat['건폐율(%)'] > 0) & (feat['연면적(㎡)'] > 0)
feat.loc[mask3, '건축면적(㎡)'] = (feat.loc[mask3, '연면적(㎡)'] / feat.loc[mask3, '건폐율(%)'] * 100).round(2)
print(f'건축면적 역산 (건폐율): {mask3.sum()}건')

# 건축면적: 연면적 / 지상층수 (건폐율 없는 경우)
mask4 = (feat['건축면적(㎡)'] == 0) & (feat['연면적(㎡)'] > 0) & (feat['지상층수'] > 0)
feat.loc[mask4, '건축면적(㎡)'] = (feat.loc[mask4, '연면적(㎡)'] / feat.loc[mask4, '지상층수']).round(2)
print(f'건축면적 역산 (층수): {mask4.sum()}건')

# ── 4. landCd 생성 ────────────────────────────────────────────────────
def make_landcd(row):
    """서울시 토지정보 API에 필요한 19자리 토지 코드를 만든다.
    형식: 시군구(5) + 법정동(5) + 대지구분(1) + 본번(4) + 부번(4)"""
    try:
        sgg = str(int(float(row['시군구코드']))).zfill(5)
        bjd = str(int(float(row['법정동코드']))).zfill(5)
        gbn = '1'  # 대지구분: 1=일반(지번 토지)
        bun = str(int(float(row['번']))).zfill(4) if pd.notna(row['번']) else '0000'
        ji  = str(int(float(row['지']))).zfill(4) if pd.notna(row['지']) else '0000'
        return sgg + bjd + gbn + bun + ji
    except Exception:
        return None

feat['_landCd'] = feat.apply(make_landcd, axis=1)
print(f'\nlandCd 생성: {feat["_landCd"].notna().sum()}개')

# ── 5. 용도지역 API 조회 (land.seoul.go.kr) ───────────────────────────
session = requests.Session()
session.get('https://land.seoul.go.kr/land/wskras/generalInfo.do', timeout=10,
            headers={'User-Agent': 'Mozilla/5.0'})
session.headers.update({
    'User-Agent': 'Mozilla/5.0',
    'Referer': 'https://land.seoul.go.kr/land/wskras/generalInfo.do',
    'X-Requested-With': 'XMLHttpRequest',
    'Accept': 'application/json, text/javascript, */*; q=0.01',
    'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
})

unique_landcds = feat['_landCd'].dropna().unique()
print(f'고유 landCd: {len(unique_landcds)}개 조회 시작...')

zone_cache = {}
for i, landCd in enumerate(unique_landcds):
    try:
        r = session.post('https://land.seoul.go.kr/land/wskras/getKoreps00047.do',
                         data={'landCd': landCd}, timeout=8)
        data = r.json().get('result', {}).get('klisDao', {})
        zone_cache[landCd] = data.get('useRegnNm', '')
    except Exception:
        zone_cache[landCd] = ''
    if (i + 1) % 200 == 0:
        print(f'  진행: {i+1}/{len(unique_landcds)}')
    time.sleep(0.05)

print(f'용도지역 조회 완료: {sum(1 for v in zone_cache.values() if v)}개 성공')

feat['용도지역'] = feat['_landCd'].map(zone_cache).fillna('')

cnt = Counter(feat['용도지역'])
print('\n용도지역 분포 (상위10):')
for k, v in cnt.most_common(10):
    print(f'  {k if k else "(없음)"}: {v}개')

# ── 6. 법정상한 컬럼 추가 ─────────────────────────────────────────────
def get_limit(zone, kind):
    """용도지역명(zone)에서 ZONE_LIMITS 키와 부분 일치로 법정상한값을 반환한다."""
    for key, val in ZONE_LIMITS.items():
        if key in str(zone):
            return val[kind]
    return None

feat['건폐율_법정상한(%)'] = feat['용도지역'].apply(lambda z: get_limit(z, '건폐율'))
feat['용적률_법정상한(%)'] = feat['용도지역'].apply(lambda z: get_limit(z, '용적률'))

print(f'\n법정상한 매핑: 건폐율={feat["건폐율_법정상한(%)"].notna().sum()}개, '
      f'용적률={feat["용적률_법정상한(%)"].notna().sum()}개')

# ── 7. 법정상한으로 대지면적/건축면적 잔여 0값 역산 ──────────────────
mask5 = (feat['대지면적(㎡)'] == 0) & (feat['연면적(㎡)'] > 0) & feat['용적률_법정상한(%)'].notna()
feat.loc[mask5, '대지면적(㎡)'] = (
    feat.loc[mask5, '연면적(㎡)'] / feat.loc[mask5, '용적률_법정상한(%)'] * 100
).round(2)
print(f'대지면적 역산 (법정상한): {mask5.sum()}건 → 남은 0: {(feat["대지면적(㎡)"]==0).sum()}개')

mask6 = (feat['건축면적(㎡)'] == 0) & (feat['연면적(㎡)'] > 0) & feat['건폐율_법정상한(%)'].notna()
feat.loc[mask6, '건축면적(㎡)'] = (
    feat.loc[mask6, '연면적(㎡)'] / feat.loc[mask6, '건폐율_법정상한(%)'] * 100
).round(2)
print(f'건축면적 역산 (법정상한): {mask6.sum()}건 → 남은 0: {(feat["건축면적(㎡)"]==0).sum()}개')

# ── 8. 용적률/건폐율 0값 → 법정상한으로 채우기 ───────────────────────
for num_col, lim_col in [('용적률(%)', '용적률_법정상한(%)'), ('건폐율(%)', '건폐율_법정상한(%)')]:
    feat[num_col] = pd.to_numeric(feat[num_col], errors='coerce').fillna(0)
    feat[lim_col] = pd.to_numeric(feat[lim_col], errors='coerce').fillna(0)
    mask = (feat[num_col] == 0) & (feat[lim_col] > 0)
    feat.loc[mask, num_col] = feat.loc[mask, lim_col]
    print(f'{num_col}: {mask.sum()}개 채움 → 0 남음: {(feat[num_col]==0).sum()}개')

# ── 9. 정리 및 저장 ───────────────────────────────────────────────────
feat.drop(columns=['_landCd'], inplace=True)

out_path = os.path.join(BASE, 'data',
                        '\ub4f1\uae30\ubd80\ub4f1\ubcf8_\uc219\ubc15\uc5c5_\ud575\uc2ec\ud53c\ucc98_\ubc95\uc815\uc0c1\ud55c.csv')
feat.to_csv(out_path, index=False, encoding='utf-8-sig')
print(f'\n저장: {out_path} ({len(feat.columns)}컬럼)')

print('\n최종 0값 현황:')
for col in ['대지면적(㎡)', '건축면적(㎡)', '용적률(%)', '건폐율(%)', '건폐율_법정상한(%)', '용적률_법정상한(%)']:
    v = pd.to_numeric(feat[col], errors='coerce')
    print(f'  {col}: 0={( v==0).sum()}개 / null={feat[col].isna().sum()}개')
