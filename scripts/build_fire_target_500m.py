# -*- coding: utf-8 -*-
"""
숙박시설 위치 데이터에 반경별 화재 카운트 타겟 컬럼을 부착하는 스크립트.

목적:
    각 숙박시설에 대해 (300m, 500m, 1000m) 반경 안에서 발생한
    숙박 관련 화재 건수를 계산하고, 그 변환값(log1p, 발생여부)도 함께 저장.

적용 필터:
    화재 발화장소_소분류 ∈ {호텔, 모텔, 여관, 여인숙, 기타 숙박시설, 숙박공유업}
    발생시군구 ∈ 분석 대상 10개구
    화재 위/경도 결측 제외

거리 계산:
    Haversine 공식으로 (N=숙소 × M=화재) 거리 행렬 한 번에 계산
    (N=4246, M≈수천 → 메모리 OK)

산출:
    NJT-PJT/data/data_with_fire_targets.csv
    추가 컬럼: 반경{300,500,1000}m_화재수 / log1p_… / 화재발생여부 / 최근접화재_거리m / _log
"""

import pandas as pd
import numpy as np
import sys

# Windows 콘솔 한글 깨짐 방지
sys.stdout.reconfigure(encoding="utf-8")

# 데이터 루트
BASE = "c:/Users/USER/Documents/GitHub/기말공모전/NJT-PJT/data"

# ── 1. 데이터 로드 ────────────────────────────────────────────────────
print("=== 데이터 로드 ===")
# 숙소 위치 (분석 대상)
lodging = pd.read_csv(f"{BASE}/핵심서울0424.csv", encoding="utf-8-sig")
# 화재 출동 원천 데이터 (4년치)
fire_raw = pd.read_csv(
    f"{BASE}/화재출동/화재출동_2021_2024.csv", encoding="utf-8-sig", low_memory=False
)
print(f"숙소: {len(lodging)}개")
print(f"화재전체: {len(fire_raw)}건")

# ── 2. 숙박화재 필터링 ────────────────────────────────────────────────
# 분석 대상 시설 유형 (소분류 코드값)
LODGING_TYPES = ["호텔", "모텔", "여관", "여인숙", "기타 숙박시설", "숙박공유업"]
# 분석 대상 10개 구
TEN_GU = [
    "종로구",
    "중구",
    "용산구",
    "성동구",
    "마포구",
    "강서구",
    "영등포구",
    "강남구",
    "서초구",
    "송파구",
]

# 모든 조건을 동시에 만족하는 화재만 추리기
fire = fire_raw[
    fire_raw["발화장소_소분류"].str.strip().isin(LODGING_TYPES)
    & fire_raw["발생시군구"].str.strip().isin(TEN_GU)
    & fire_raw["위도"].notna()
    & fire_raw["경도"].notna()
].copy()

print(f"숙박화재(10개구 필터): {len(fire)}건")
# 시설 유형별 분포 — 데이터 점검
print(fire["발화장소_소분류"].value_counts().to_string())


# ── 3. Haversine 거리 함수 ─────────────────────────────────────────────
def haversine_matrix(lat1, lon1, lat2, lon2):
    """위경도 두 집합 사이의 모든 쌍에 대해 Haversine 거리 행렬을 계산.

    인자:
        lat1/lon1: (N,) 숙소 좌표
        lat2/lon2: (M,) 화재 좌표
    반환:
        (N, M) 거리 행렬 (단위: m)
    """
    R = 6371000  # 지구 반지름 (m)
    # broadcast 가능한 모양으로 reshape
    lat1 = np.radians(lat1)[:, None]  # (N, 1)
    lon1 = np.radians(lon1)[:, None]
    lat2 = np.radians(lat2)[None, :]  # (1, M)
    lon2 = np.radians(lon2)[None, :]

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))  # (N, M)


# ── 4. 반경별 화재수 계산 ──────────────────────────────────────────────
print("\n=== 공간 매칭 중 (4246 × {}행렬) ===".format(len(fire)))

# 위경도 numpy 배열 추출
lat_l = lodging["위도"].values
lon_l = lodging["경도"].values
lat_f = fire["위도"].values
lon_f = fire["경도"].values

# 한 번에 거리 행렬 계산 (메모리: N × M × 8byte)
dist = haversine_matrix(lat_l, lon_l, lat_f, lon_f)

# 반경 후보별 카운트/log1p/발생여부 계산
for r in [300, 500, 1000]:
    col = f"반경{r}m_화재수"
    # 반경 안에 있는 화재 개수 (행=숙소, 열=화재)
    lodging[col] = (dist <= r).sum(axis=1)
    # 분포 평탄화를 위한 log1p 변환
    lodging[f"log1p_반경{r}m"] = np.log1p(lodging[col])
    # 분류 모델용 0/1 라벨
    lodging[f"반경{r}m_화재발생여부"] = (lodging[col] >= 1).astype(int)
    # 진행 상황 출력 — 평균값과 0의 비율로 데이터 점검
    print(
        f"  {r}m — 평균: {lodging[col].mean():.3f}, 0값비율: {(lodging[col] == 0).mean() * 100:.1f}%"
    )

# 추가 — 가장 가까운 화재까지의 거리(m)
lodging["최근접화재_거리m"] = dist.min(axis=1)
lodging["최근접화재_거리_log"] = np.log1p(lodging["최근접화재_거리m"])

# ── 5. 저장 ───────────────────────────────────────────────────────────
out = f"{BASE}/data_with_fire_targets.csv"
lodging.to_csv(out, index=False, encoding="utf-8-sig")
print(f"\n[저장] {out}")
print("추가된 컬럼: 반경300/500/1000m 화재수, log1p, 발생여부, 최근접거리")

# ── 6. 요약 ───────────────────────────────────────────────────────────
print("\n=== 최종 Y 분포 (log1p_반경500m) ===")
y = lodging["log1p_반경500m"]
print(
    f"  평균: {y.mean():.3f}  std: {y.std():.3f}  min: {y.min():.3f}  max: {y.max():.3f}"
)
# 왜도 비교 — log1p 변환이 분포 평탄화에 얼마나 효과적이었는지 확인
print(f"  왜도: {y.skew():.3f}  (원본 왜도: {lodging['반경500m_화재수'].skew():.3f})")
