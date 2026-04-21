# -*- coding: utf-8 -*-
"""
AHP 기반 위험점수 산출 스크립트 (6개 변수)

[사용 변수]
  소방접근성_점수, 노후도_점수, 도로폭(공식도로폭m),
  반경_50m_건물수(밀집도), 집중도(%), 로그_주변대비_상대위험도_고유단속지점_50m(불법주정차)

[우선순위]
  소방접근성 > 노후도 > 도로폭 > 밀집도 > 집중도 > 불법주정차

[입출력]
  입력: data/서울10구_숙소_소방거리_유클리드.csv
  출력: data/서울10구_숙소_소방거리_유클리드.csv (위험점수_AHP 컬럼 갱신)
        data/분석변수_테이블.csv (위험점수_AHP 컬럼 갱신)
"""
import pandas as pd, numpy as np, sys
from sklearn.preprocessing import MinMaxScaler
sys.stdout.reconfigure(encoding='utf-8')

BASE     = 'c:/Users/USER/Documents/GitHub/기말공모전/NJT-PJT/data'
MAIN_CSV = f'{BASE}/서울10구_숙소_소방거리_유클리드.csv'
ANLY_CSV = f'{BASE}/분석변수_테이블.csv'

df = pd.read_csv(MAIN_CSV, encoding='utf-8-sig')
print(f'로드: {len(df)}행')

# ── 1. 위험 방향 통일 (높을수록 위험) ────────────────────────────────
scaler = MinMaxScaler()

df['소방_위험도']       = 1 - df['소방접근성_점수']                          # 거리 멀수록 위험
df['밀집도_정규화']     = scaler.fit_transform(df[['반경_50m_건물수']])       # 건물 많을수록 위험
df['집중도_정규화']     = df['집중도(%)'] / 100                               # 높을수록 위험
df['도로폭_정규화']     = 1 - scaler.fit_transform(df[['공식도로폭m']])       # 좁을수록 위험
df['불법주정차_정규화'] = scaler.fit_transform(
    df[['로그_주변대비_상대위험도_고유단속지점_50m']].fillna(
        df['로그_주변대비_상대위험도_고유단속지점_50m'].median()
    )
)

vars_risk = ['소방_위험도', '노후도_점수', '도로폭_정규화',
             '밀집도_정규화', '집중도_정규화', '불법주정차_정규화']
X = df[vars_risk].values

# ── 2. AHP 쌍대비교 행렬 ─────────────────────────────────────────────
# 소방접근성 > 노후도 > 도로폭 > 밀집도 > 집중도 > 불법주정차
ahp_matrix = np.array([
    [1,   2,   3,   4,   5,   6],   # 소방_위험도
    [1/2, 1,   2,   3,   4,   5],   # 노후도
    [1/3, 1/2, 1,   2,   3,   4],   # 도로폭
    [1/4, 1/3, 1/2, 1,   2,   3],   # 밀집도
    [1/5, 1/4, 1/3, 1/2, 1,   2],   # 집중도
    [1/6, 1/5, 1/4, 1/3, 1/2, 1],   # 불법주정차
], dtype=float)

# 고유벡터로 가중치 산출
eigenvalues, eigenvectors = np.linalg.eig(ahp_matrix)
max_idx    = np.argmax(eigenvalues.real)
ahp_weights = eigenvectors[:, max_idx].real
ahp_weights = ahp_weights / ahp_weights.sum()

# 일관성 비율(CR) 계산
lambda_max = eigenvalues[max_idx].real
n  = len(ahp_matrix)
CI = (lambda_max - n) / (n - 1)
RI = {1:0, 2:0, 3:0.58, 4:0.90, 5:1.12, 6:1.24, 7:1.32}[n]
CR = CI / RI

print('\n=== AHP 가중치 ===')
for v, w in zip(vars_risk, ahp_weights):
    print(f'  {v}: {w:.4f} ({w*100:.1f}%)')
print(f'  CR={CR:.3f}', '(일관성 OK)' if CR < 0.1 else '(재검토 필요)')

# ── 3. 위험점수 산출 (0~100) ──────────────────────────────────────────
score = (X * ahp_weights).sum(axis=1)
df['위험점수_AHP'] = ((score - score.min()) /
                      (score.max() - score.min()) * 100).round(2)

print(f'\n=== 위험점수_AHP 분포 ===')
print(f'  평균: {df["위험점수_AHP"].mean():.1f} | 최소: {df["위험점수_AHP"].min():.1f} | 최대: {df["위험점수_AHP"].max():.1f}')

# ── 4. 임시 컬럼 제거 후 저장 ────────────────────────────────────────
df = df.drop(columns=['소방_위험도','밀집도_정규화','집중도_정규화',
                       '도로폭_정규화','불법주정차_정규화'])
df.to_csv(MAIN_CSV, index=False, encoding='utf-8-sig')
print(f'\n[저장] {MAIN_CSV}')

# 분석변수_테이블.csv 도 갱신
anly = pd.read_csv(ANLY_CSV, encoding='utf-8-sig')
anly['위험점수_AHP'] = df['위험점수_AHP'].values[:len(anly)]
anly.to_csv(ANLY_CSV, index=False, encoding='utf-8-sig')
print(f'[저장] {ANLY_CSV}')

print('\n상위 위험 5개:')
top5 = df.nlargest(5, '위험점수_AHP')[['구','업소명','위험점수_AHP','소방접근성_점수','노후도_점수','공식도로폭m']]
print(top5.to_string(index=False))
