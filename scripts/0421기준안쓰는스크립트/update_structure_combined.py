# -*- coding: utf-8 -*-
"""
구조_위험점수 + 노후도_점수 + 구조_노후_상호작용 → 합산 후 MinMax → 구조_노후_통합점수
핵심분析변수_테이블, 서울10구_숙소_소방거리_유클리드, 핵심서울10구... 에 반영
"""
import pandas as pd, numpy as np, glob, sys
sys.stdout.reconfigure(encoding='utf-8')

BASE = 'c:/Users/USER/Documents/GitHub/기말공모전/NJT-PJT/data'

def add_combined(df):
    # 3개 컬럼 합산 후 MinMax
    raw = df['구조_위험점수'] + df['노후도_점수'] + df['구조_노후_상호작용']
    mn, mx = raw.min(), raw.max()
    df['구조_노후_통합점수'] = ((raw - mn) / (mx - mn)).round(4)
    # 기존 3개 제거
    df = df.drop(columns=['구조_위험점수', '노후도_점수', '구조_노후_상호작용'], errors='ignore')
    return df

# 핵심분析변수_테이블
for f in glob.glob(f'{BASE}/*.csv'):
    if '핵심분析변수' in f:
        df = pd.read_csv(f, encoding='utf-8-sig')
        df = add_combined(df)
        df.to_csv(f, index=False, encoding='utf-8-sig')
        print(f'[저장] {f.split("/")[-1]}')
        print('컬럼:', df.columns.tolist())

# 메인 CSV (구조_위험점수, 노후도_점수, 구조_노후_상호작용 유지하되 통합점수 추가)
for f in glob.glob(f'{BASE}/*.csv'):
    if '서울10구_숙소_소방거리_유클리드' in f or '도로폭추가' in f:
        df = pd.read_csv(f, encoding='utf-8-sig', on_bad_lines='skip')
        if all(c in df.columns for c in ['구조_위험점수','노후도_점수','구조_노후_상호작용']):
            raw = df['구조_위험점수'] + df['노후도_점수'] + df['구조_노후_상호작용']
            mn, mx = raw.min(), raw.max()
            df['구조_노후_통합점수'] = ((raw - mn) / (mx - mn)).round(4)
            df.to_csv(f, index=False, encoding='utf-8-sig')
            print(f'[저장] {f.split("/")[-1]}')
