# -*- coding: utf-8 -*-
"""
AHP 기반 위험점수(0~100) 산출 스크립트 — 6개 변수.

사용 변수 (모두 "높을수록 위험" 방향으로 정규화):
    소방위험도_점수, 노후도_점수, 도로폭(공식도로폭m, 좁을수록 위험),
    반경_50m_건물수(밀집도), 집중도(%), 로그_주변대비_상대위험도_고유단속지점_50m(불법주정차)

AHP 우선순위(쌍대비교 기반):
    소방위험도 > 노후도 > 도로폭 > 밀집도 > 집중도 > 불법주정차

산출:
    원본 CSV (서울10구_숙소_소방거리_유클리드.csv) 의 위험점수_AHP 컬럼 갱신
    분석변수_테이블.csv 의 위험점수_AHP 컬럼 갱신
"""

import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import MinMaxScaler

# Windows 콘솔 한글 깨짐 방지
sys.stdout.reconfigure(encoding="utf-8")

BASE = "c:/Users/USER/Documents/GitHub/기말공모전/NJT-PJT/data"
MAIN_CSV = f"{BASE}/서울10구_숙소_소방거리_유클리드.csv"
ANLY_CSV = f"{BASE}/분석변수_테이블.csv"

df = pd.read_csv(MAIN_CSV, encoding="utf-8-sig")
print(f"로드: {len(df)}행")

# ── 1. 위험 방향 통일 (높을수록 위험) ────────────────────────────────
scaler = MinMaxScaler()

# 건물 많을수록 위험 — 0~1 정규화
df["밀집도_정규화"] = scaler.fit_transform(df[["반경_50m_건물수"]])
# 집중도(%)는 이미 0~100 -> 0~1로
df["집중도_정규화"] = df["집중도(%)"] / 100
# 도로폭은 좁을수록 위험 -> 1 - 정규화
df["도로폭_정규화"] = 1 - scaler.fit_transform(df[["공식도로폭m"]])
# 불법주정차 지표 — 결측은 중앙값 보강 후 0~1 정규화
df["불법주정차_정규화"] = scaler.fit_transform(
    df[["로그_주변대비_상대위험도_고유단속지점_50m"]].fillna(
        df["로그_주변대비_상대위험도_고유단속지점_50m"].median()
    )
)

# AHP 입력에 들어갈 6변수 (위 정규화 결과 + 이미 점수형인 두 변수)
vars_risk = [
    "소방위험도_점수",
    "노후도_점수",
    "도로폭_정규화",
    "밀집도_정규화",
    "집중도_정규화",
    "불법주정차_정규화",
]
X = df[vars_risk].values

# ── 2. AHP 쌍대비교 행렬 ─────────────────────────────────────────────
# 우선순위 인접 변수 사이 비교 척도(2~6)는 통상 Saaty 척도 사용
ahp_matrix = np.array(
    [
        [1, 2, 3, 4, 5, 6],          # 소방위험도
        [1 / 2, 1, 2, 3, 4, 5],      # 노후도
        [1 / 3, 1 / 2, 1, 2, 3, 4],  # 도로폭
        [1 / 4, 1 / 3, 1 / 2, 1, 2, 3],  # 밀집도
        [1 / 5, 1 / 4, 1 / 3, 1 / 2, 1, 2],  # 집중도
        [1 / 6, 1 / 5, 1 / 4, 1 / 3, 1 / 2, 1],  # 불법주정차
    ],
    dtype=float,
)

# 최대 고유값에 대응하는 고유벡터 -> 가중치 (정규화)
eigenvalues, eigenvectors = np.linalg.eig(ahp_matrix)
max_idx = np.argmax(eigenvalues.real)
ahp_weights = eigenvectors[:, max_idx].real
ahp_weights = ahp_weights / ahp_weights.sum()

# 일관성 비율(CR) — λ_max, n, RI 사용
lambda_max = eigenvalues[max_idx].real
n = len(ahp_matrix)
CI = (lambda_max - n) / (n - 1)
RI = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32}[n]
CR = CI / RI

print("\n=== AHP 가중치 ===")
for v, w in zip(vars_risk, ahp_weights):
    print(f"  {v}: {w:.4f} ({w * 100:.1f}%)")
# CR < 0.1 이면 일관성 양호로 본다
print(f"  CR={CR:.3f}", "(일관성 OK)" if CR < 0.1 else "(재검토 필요)")

# ── 3. 위험점수 산출 (0~100) ──────────────────────────────────────────
# 변수 가중합 후 0~100으로 minmax 정규화
score = (X * ahp_weights).sum(axis=1)
df["위험점수_AHP"] = ((score - score.min()) / (score.max() - score.min()) * 100).round(
    2
)

print("\n=== 위험점수_AHP 분포 ===")
print(
    f"  평균: {df['위험점수_AHP'].mean():.1f} | 최소: {df['위험점수_AHP'].min():.1f} | 최대: {df['위험점수_AHP'].max():.1f}"
)

# ── 4. 임시 정규화 컬럼 제거 후 저장 ────────────────────────────────
df = df.drop(
    columns=["밀집도_정규화", "집중도_정규화", "도로폭_정규화", "불법주정차_정규화"]
)
df.to_csv(MAIN_CSV, index=False, encoding="utf-8-sig")
print(f"\n[저장] {MAIN_CSV}")

# 분석변수_테이블.csv 도 같은 점수로 갱신 (행 수 보호 위해 슬라이스)
anly = pd.read_csv(ANLY_CSV, encoding="utf-8-sig")
anly["위험점수_AHP"] = df["위험점수_AHP"].values[: len(anly)]
anly.to_csv(ANLY_CSV, index=False, encoding="utf-8-sig")
print(f"[저장] {ANLY_CSV}")

# 검증 출력 — 상위 위험 5개 시설
print("\n상위 위험 5개:")
top5 = df.nlargest(5, "위험점수_AHP")[
    ["구", "업소명", "위험점수_AHP", "소방위험도_점수", "노후도_점수", "공식도로폭m"]
]
print(top5.to_string(index=False))
