# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


BASE = Path(__file__).resolve().parents[1]
INPUT_PATH = BASE / "0424" / "data" / "분석변수_최종테이블0428.csv"
OUTPUT_PATH = BASE / "0424" / "data" / "최종_화재위험_분석결과_0428.csv"

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
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp949")


def main() -> None:
    df = read_csv_with_fallback(INPUT_PATH)
    df.columns = df.columns.str.strip()

    features = list(WEIGHTS.keys())
    missing = [col for col in features if col not in df.columns]
    if missing:
        raise KeyError(f"필수 컬럼이 없습니다: {missing}")

    scaler = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[features] = df[features].apply(pd.to_numeric, errors="coerce").fillna(0)
    df_scaled[features] = scaler.fit_transform(df_scaled[features])

    df["최종_화재위험점수"] = (df_scaled[features] * pd.Series(WEIGHTS)).sum(axis=1) * 100

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df["cluster_k3"] = kmeans.fit_predict(df_scaled[features])
    df["cluster"] = df["cluster_k3"]

    cluster_summary = df.groupby("cluster_k3")[["최종_화재위험점수"] + features].mean()
    top_20 = (
        df[["구", "동", "숙소명", "최종_화재위험점수", "cluster_k3", "최근접_소화용수_거리등급"]]
        .sort_values(by="최종_화재위험점수", ascending=False)
        .head(20)
    )

    print("--- [군집별 위험도 및 변수 평균] ---")
    print(cluster_summary.T)

    print("\n--- [최종 화재 위험 시설 TOP 20] ---")
    print(top_20)

    print("\n--- [군집별 고위험 TOP 10] ---")
    for cluster_id in sorted(df["cluster_k3"].unique()):
        print(f"\n[Cluster {cluster_id} - 고위험 TOP 10]")
        top_10 = (
            df[df["cluster_k3"] == cluster_id][["구", "동", "숙소명", "최종_화재위험점수"]]
            .sort_values(by="최종_화재위험점수", ascending=False)
            .head(10)
        )
        print(top_10)

    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\n저장 완료: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
