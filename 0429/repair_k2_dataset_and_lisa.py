# -*- coding: utf-8 -*-
"""
K=2 클러스터링 데이터셋 정비 + 라벨 정렬 스크립트.

목적:
    cluster3 파이프라인의 산출물 CSV를 베이스로,
    fire_count_150m 타겟을 부착하고 결측/중복을 정리한 뒤
    KMeans(k=2)를 다시 적합하여 cluster_k2 라벨을 생성한다.

라벨 규칙:
    cluster_k2 = 1 이 "고위험/고밀도 그룹"이 되도록,
    최종_화재위험점수의 그룹별 평균이 큰 쪽을 1로 강제 정렬한다.
    (이전 분석 코드들과의 라벨 해석 일관성 유지를 위함)

산출물:
    NJT-PJT/0429/cluster2_spatial_pipeline_fire_count_150m_0429/
        최최최종0428변수테이블_cluster_k2.csv
"""

from __future__ import annotations

from pathlib import Path

# 표 데이터 처리
import pandas as pd
# K-평균 클러스터링
from sklearn.cluster import KMeans
# 클러스터 입력 변수 표준화
from sklearn.preprocessing import StandardScaler


# 프로젝트 루트 (0429 폴더 기준 한 단계 위)
ROOT = Path(__file__).resolve().parents[1]
# 입력 후보가 들어 있는 디렉터리 (가장 큰 CSV를 자동 선택)
DATA_DIR = ROOT / "0424" / "data" / "cluster3_spatial_pipeline_fire_count_150m_0428"
# fire_count_150m 보유 데이터셋 (타겟 머지용)
FIRE_TARGET_PATH = (
    ROOT / "data" / "team_pipeline_validation" / "team_pipeline_scored_dataset.csv"
)
# k=2 산출물 저장 디렉터리
K2_DIR = ROOT / "0429" / "cluster2_spatial_pipeline_fire_count_150m_0429"

# 회귀/리포트용 종속 변수
TARGET = "fire_count_150m"
# K=2 KMeans에 사용할 클러스터 입력 변수 (위험 + 도로/건물/소방 접근)
CLUSTER_FEATURES = [
    "최종_화재위험점수",
    "도로폭위험도",
    "집중도",
    "주변건물수",
    "최근접_소화용수_거리등급",
]
# 후속 회귀에 들어갈 독립 변수 (이 스크립트에선 결측 제거 기준으로 사용)
REG_FEATURES = [
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
    """숙소명 매칭 키 — 모든 공백 제거 + strip."""
    return s.astype(str).str.replace(r"\s+", "", regex=True).str.strip()


def main() -> None:
    """k=2 클러스터 결과 CSV 생성 메인."""
    # 후보 폴더에서 가장 용량이 큰 CSV를 베이스로 선택 (가장 완전한 변수 테이블 가정)
    main_csv = max(DATA_DIR.glob("*.csv"), key=lambda p: p.stat().st_size)
    df = pd.read_csv(main_csv, encoding="utf-8-sig")
    # 화재 타겟 데이터셋 로드
    fire = pd.read_csv(FIRE_TARGET_PATH, encoding="utf-8-sig")

    # 매칭 키: 정규화된 숙소명 + 위경도 6자리 반올림 + target 값
    fire_key = pd.DataFrame(
        {
            "_name_key": name_key(fire["숙소명"]),
            "_lat_key": pd.to_numeric(fire["위도"], errors="coerce").round(6),
            "_lon_key": pd.to_numeric(fire["경도"], errors="coerce").round(6),
            TARGET: pd.to_numeric(fire[TARGET], errors="coerce"),
        }
    ).drop_duplicates(["_name_key", "_lat_key", "_lon_key"])

    # 베이스 df 측 키 생성 + 좌측 병합으로 타겟 부착
    out = df.copy()
    out["_name_key"] = name_key(out["숙소명"])
    out["_lat_key"] = pd.to_numeric(out["위도"], errors="coerce").round(6)
    out["_lon_key"] = pd.to_numeric(out["경도"], errors="coerce").round(6)
    out = out.merge(fire_key, on=["_name_key", "_lat_key", "_lon_key"], how="left")
    # 임시 키 컬럼 제거
    out = out.drop(columns=["_name_key", "_lat_key", "_lon_key"], errors="ignore")

    # 최종 보존 컬럼 — 식별/좌표/타겟/회귀 변수/대표 위험 점수
    keep = [
        "구",
        "동",
        "숙소명",
        "경도",
        "위도",
        "x_5181",
        "y_5181",
        TARGET,
        *REG_FEATURES,
        "최종_화재위험점수",
    ]
    # 수치 변환이 필요한 컬럼 일괄 처리
    for col in [
        TARGET,
        "경도",
        "위도",
        "x_5181",
        "y_5181",
        *REG_FEATURES,
        "최종_화재위험점수",
    ]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    # 모델/지표 안정성을 위해 핵심 컬럼 결측 행 제거
    out = (
        out[keep]
        .dropna(subset=[TARGET, "경도", "위도", "x_5181", "y_5181", *REG_FEATURES])
        .copy()
    )
    # 동일 숙소가 여러 행으로 들어가는 케이스 방지 — 고유키 기준 중복 제거
    out = out.drop_duplicates(
        ["숙소명", "경도", "위도", "x_5181", "y_5181"]
    ).reset_index(drop=True)

    # 클러스터링 입력 변수 표준화 (스케일 차이 정규화)
    x = StandardScaler().fit_transform(out[CLUSTER_FEATURES].to_numpy(dtype=float))
    # KMeans k=2, 50회 초기화로 안정적 군집 산출 (random_state 고정으로 재현성)
    out["cluster_k2"] = KMeans(n_clusters=2, random_state=42, n_init=50).fit_predict(x)

    # KMeans는 라벨을 0/1로 임의 부여하므로,
    # 화재위험점수 평균이 큰 클러스터를 항상 1번으로 매핑 (해석 일관성)
    risk_mean = out.groupby("cluster_k2")["최종_화재위험점수"].mean()
    high_label = int(risk_mean.idxmax())
    if high_label != 1:
        # 0 <-> 1 라벨 스왑
        out["cluster_k2"] = 1 - out["cluster_k2"]

    # 결과 CSV 저장
    out_path = K2_DIR / "최최최종0428변수테이블_cluster_k2.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")

    # 검증 출력: 경로 / 클러스터별 표본 수 / 총 행 수
    print(out_path)
    print(out.groupby("cluster_k2").size().to_string())
    print("total", len(out))


# 직접 실행 시에만 동작
if __name__ == "__main__":
    main()
