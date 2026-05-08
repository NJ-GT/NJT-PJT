# -*- coding: utf-8 -*-
"""
0428 분석변수 테이블에 대해 가중합 화재위험점수 + K=3 KMeans 클러스터링을 수행하는 스크립트.

가중치(WEIGHTS):
    구조노후도 24%, 단속위험도 16%, 도로폭위험도 14%,
    최근접_소화용수_거리등급 12%, 소방위험도_점수 11%, 연면적 9%,
    집중도 7%, 주변건물수 5%, 총층수 2%
    (모두 합 = 1.00)

처리 흐름:
    1) 입력 CSV 로드 (utf-8-sig 우선, 실패 시 cp949 fallback)
    2) MinMaxScaler로 0~1 정규화 + 가중합 -> 0~100 스케일의 '최종_화재위험점수'
    3) 정규화된 변수로 KMeans(K=3) 적합 -> cluster_k3 / cluster 라벨
    4) 군집별 평균, 전체 TOP20, 군집별 TOP10 출력
    5) 결과 CSV 저장
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
# 변수별 스케일 차이를 0~1로 통일 (가중합 비교 가능)
from sklearn.preprocessing import MinMaxScaler


# scripts 기준 한 단계 위 (NJT-PJT/)
BASE = Path(__file__).resolve().parents[1]
INPUT_PATH = BASE / "0424" / "data" / "분석변수_최종테이블0428.csv"
OUTPUT_PATH = BASE / "0424" / "data" / "최종_화재위험_분석결과_0428.csv"

# 가중합에 들어갈 변수와 가중치 (전문가 검토 기준)
WEIGHTS = {
    "구조노후도": 0.24,
    "단속위험도": 0.16,
    "도로폭위험도": 0.14,
    "최근접_소화용수_거리등급": 0.12,
    "소방위험도_점수": 0.11,
    "연면적": 0.09,
    "집중도": 0.07,
    "주변건물수": 0.05,
    "총층수": 0.02,
}


def read_csv_with_fallback(path: Path) -> pd.DataFrame:
    """UTF-8 BOM 우선, 실패하면 cp949(엑셀 저장)로 폴백 로드."""
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp949")


def main() -> None:
    """가중합 + 군집화 + 보고용 출력 + CSV 저장."""
    df = read_csv_with_fallback(INPUT_PATH)
    # 헤더 양끝 공백 제거 (혼합 입력 안전)
    df.columns = df.columns.str.strip()

    # 가중치에 등장하는 모든 변수가 컬럼에 존재해야 함
    features = list(WEIGHTS.keys())
    missing = [col for col in features if col not in df.columns]
    if missing:
        raise KeyError(f"필수 컬럼이 없습니다: {missing}")

    # 0~1 정규화 (변수 간 스케일 차이 제거)
    scaler = MinMaxScaler()
    df_scaled = df.copy()
    # 수치 변환 + NaN -> 0 (정규화 안전)
    df_scaled[features] = df[features].apply(pd.to_numeric, errors="coerce").fillna(0)
    df_scaled[features] = scaler.fit_transform(df_scaled[features])

    # 가중합 후 0~100 스케일 — 직관적 점수화
    df["최종_화재위험점수"] = (df_scaled[features] * pd.Series(WEIGHTS)).sum(
        axis=1
    ) * 100

    # K=3 KMeans (정규화된 변수 사용, 재현 가능성 확보)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df["cluster_k3"] = kmeans.fit_predict(df_scaled[features])
    # 후속 분석에서 'cluster'를 기대하는 코드와의 호환을 위해 별도 컬럼으로 복제
    df["cluster"] = df["cluster_k3"]

    # 군집별 평균 — 군집 해석에 필수
    cluster_summary = df.groupby("cluster_k3")[["최종_화재위험점수"] + features].mean()
    # 전체 TOP 20 (점수 내림차순)
    top_20 = (
        df[
            [
                "구",
                "동",
                "숙소명",
                "최종_화재위험점수",
                "cluster_k3",
                "최근접_소화용수_거리등급",
            ]
        ]
        .sort_values(by="최종_화재위험점수", ascending=False)
        .head(20)
    )

    # 콘솔 검증 출력 — 군집별 평균
    print("--- [군집별 위험도 및 변수 평균] ---")
    print(cluster_summary.T)

    print("\n--- [최종 화재 위험 시설 TOP 20] ---")
    print(top_20)

    # 군집별 TOP10 — 정책 대상 우선순위 식별용
    print("\n--- [군집별 고위험 TOP 10] ---")
    for cluster_id in sorted(df["cluster_k3"].unique()):
        print(f"\n[Cluster {cluster_id} - 고위험 TOP 10]")
        top_10 = (
            df[df["cluster_k3"] == cluster_id][
                ["구", "동", "숙소명", "최종_화재위험점수"]
            ]
            .sort_values(by="최종_화재위험점수", ascending=False)
            .head(10)
        )
        print(top_10)

    # 결과 저장
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\n저장 완료: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
