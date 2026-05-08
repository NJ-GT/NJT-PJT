# -*- coding: utf-8 -*-
"""
클러스터별 공간시차모형(SLM) ρ(rho) 추정 스크립트.

목적:
    클러스터(cluster) 컬럼이 부여된 변수 테이블에 대해,
    각 클러스터마다 KNN(k=12) 기반 공간 가중치 행렬을 만들고
    Maximum Likelihood Spatial Lag (ML_Lag) 모형을 적합하여
    공간 자기상관 계수 ρ, pseudo R², AIC를 추출한다.

산출물:
    NJT-PJT/0429/slm_rho_k12_fire_count_150m_by_cluster.csv
        cluster, n, knn_k, rho, pseudo_r2, aic 컬럼

배경:
    target 변수: fire_count_150m (반경 150m 내 화재 건수)
    target 컬럼이 입력 CSV에 없을 수 있으므로,
    team_pipeline_scored_dataset.csv 에서 (숙소명+위도+경도) 키로 부착한다.
"""

# 향후 typing 호환성 (PEP 563)
from __future__ import annotations

# 경로 처리
from pathlib import Path

# 수치 연산 (배열 reshape, NaN 처리 등)
import numpy as np
# 표 데이터 처리
import pandas as pd
# K-최근접 이웃 기반 공간 가중치 행렬
from libpysal.weights import KNN
# 변수 표준화 (평균 0, 분산 1)
from sklearn.preprocessing import StandardScaler
# 공간시차모형 (Spatial Lag Model, ML 추정)
from spreg import ML_Lag


# 프로젝트 루트(NJT-PJT/) — 0429 폴더 기준 한 단계 위
ROOT = Path(__file__).resolve().parents[1]
# 클러스터링 결과가 들어 있는 변수 테이블 경로 (0428 파이프라인 산출물)
DATA_PATH = (
    ROOT
    / "0424"
    / "data"
    / "cluster3_spatial_pipeline_fire_count_150m_0428"
    / "최최최종0428변수테이블.csv"
)
# fire_count_150m 타겟이 포함된 팀 검증 데이터셋 (병합용)
FIRE_TARGET_PATH = (
    ROOT / "data" / "team_pipeline_validation" / "team_pipeline_scored_dataset.csv"
)
# 클러스터별 ρ 결과 저장 위치
OUT_PATH = ROOT / "0429" / "slm_rho_k12_fire_count_150m_by_cluster.csv"

# 종속 변수
TARGET = "fire_count_150m"
# 클러스터 라벨 컬럼
CLUSTER_COL = "cluster"
# 5181 좌표(미터 기반) — KNN 거리 계산에 사용
COORD_COLS = ["x_5181", "y_5181"]
# 회귀에 투입할 독립 변수 목록 (모두 수치형이어야 함)
FEATURES = [
    "승인연도",
    "소방위험도_점수",
    "주변건물수",
    "집중도",
    "단속위험도",
    "구조노후도",
    "도로폭위험도",
    "최근접_소화용수_거리등급",
    "총층수",
    "연면적",
]


def name_key(s: pd.Series) -> pd.Series:
    """문자열 시리즈를 공백 제거 + strip으로 정규화 (조인 키 안정화)."""
    return s.astype(str).str.replace(r"\s+", "", regex=True).str.strip()


def attach_fire_target(df: pd.DataFrame) -> pd.DataFrame:
    """입력 df에 fire_count_150m 타겟 컬럼이 없으면 외부 CSV에서 매칭해 부착."""
    # 이미 타겟이 있으면 작업 불필요
    if TARGET in df.columns:
        return df
    # 타겟 보유 데이터셋 로드
    fire = pd.read_csv(FIRE_TARGET_PATH, encoding="utf-8-sig")
    # 매칭 키 구성: 정규화된 숙소명 + 위경도 6자리 반올림
    fire_key = pd.DataFrame(
        {
            "_name_key": name_key(fire["숙소명"]),
            "_lat_key": pd.to_numeric(fire["위도"], errors="coerce").round(6),
            "_lon_key": pd.to_numeric(fire["경도"], errors="coerce").round(6),
            TARGET: pd.to_numeric(fire[TARGET], errors="coerce"),
        }
    ).drop_duplicates(["_name_key", "_lat_key", "_lon_key"])
    # 원본 df 측에도 동일 키를 만들고 left merge
    keyed = df.copy()
    keyed["_name_key"] = name_key(keyed["숙소명"])
    keyed["_lat_key"] = pd.to_numeric(keyed["위도"], errors="coerce").round(6)
    keyed["_lon_key"] = pd.to_numeric(keyed["경도"], errors="coerce").round(6)
    keyed = keyed.merge(fire_key, on=["_name_key", "_lat_key", "_lon_key"], how="left")
    # 임시 키 컬럼은 제거하고 반환
    return keyed.drop(columns=["_name_key", "_lat_key", "_lon_key"], errors="ignore")


def main() -> None:
    """클러스터 단위 SLM 적합 -> rho 등 결과 CSV 저장."""
    # 변수 테이블 로드
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    # 타겟 컬럼 부착(필요 시)
    df = attach_fire_target(df)
    # 모델 입력에 쓰이는 컬럼은 모두 수치형으로 강제 변환
    for col in FEATURES + [TARGET, CLUSTER_COL] + COORD_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # 결측 한 행이라도 있으면 모델 적합 불가 -> 제거
    df = df.dropna(subset=FEATURES + [TARGET, CLUSTER_COL] + COORD_COLS).reset_index(
        drop=True
    )

    # 클러스터별 결과를 누적할 리스트
    rows = []
    # 정렬된 cluster id를 순회 (재현성 확보)
    for cluster_id in sorted(df[CLUSTER_COL].astype(int).unique()):
        # 해당 클러스터 데이터만 필터
        sub = df[df[CLUSTER_COL].astype(int).eq(cluster_id)].reset_index(drop=True)
        # 독립 변수 표준화 (스케일 차이 정규화)
        x = StandardScaler().fit_transform(sub[FEATURES].to_numpy(dtype=float))
        # 종속 변수 (n,1) 형태로 변환
        y = sub[TARGET].to_numpy(dtype=float).reshape(-1, 1)
        # KNN 입력용 좌표 행렬
        coords = sub[COORD_COLS].to_numpy(dtype=float)
        # 클러스터 표본이 12보다 적을 수 있으니 k는 max(1, n-1)로 안전 처리
        k = min(12, max(1, len(coords) - 1))
        # KNN 가중치 행렬 구성
        w = KNN.from_array(coords, k=k)
        # 행 표준화(row-standardized) — 각 행 합이 1이 되도록 정규화
        w.transform = "r"
        # 공간시차모형 적합 (ML 추정)
        model = ML_Lag(y, x, w=w, name_y=TARGET, name_x=FEATURES)
        # rho 추출 — 모델 객체에 따라 스칼라/배열일 수 있어 평탄화 후 첫 원소
        rho = float(np.asarray(model.rho).reshape(-1)[0])
        # 결과 한 줄 적재
        rows.append(
            {
                "cluster": cluster_id,
                "n": len(sub),
                "knn_k": k,
                "rho": rho,
                # 일부 버전에서 pr2/aic 속성이 없을 수 있어 getattr 안전 접근
                "pseudo_r2": float(getattr(model, "pr2", np.nan)),
                "aic": float(getattr(model, "aic", np.nan)),
            }
        )

    # DataFrame -> CSV 저장 (UTF-8 BOM, 엑셀 호환)
    out = pd.DataFrame(rows)
    out.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    # 산출 경로와 결과 미리보기 출력
    print(OUT_PATH)
    print(out.to_string(index=False))


# 직접 실행 시에만 main 호출
if __name__ == "__main__":
    main()
