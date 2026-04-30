# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


BASE = Path(__file__).resolve().parents[1]
INPUT_PATH = BASE / "0424" / "data" / "분석변수_최종테이블0428.csv"
OUTPUT_PATH = BASE / "0424" / "data" / "최종_화재위험_분석결과_0428_로그연면적.csv"

AREA_COL = "연면적"
LOG_AREA_COL = "log1p_연면적"

WEIGHTS = {
    "구조노후도": 0.24,
    "단속위험도": 0.16,
    "도로폭위험도": 0.14,
    "최근접_소화용수_거리등급": 0.12,
    "소방위험도_점수": 0.11,
    LOG_AREA_COL: 0.09,
    "집중도": 0.07,
    "주변건물수": 0.05,
    "총층수": 0.02,
}


def main() -> None:
    df = pd.read_csv(INPUT_PATH, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()

    area = pd.to_numeric(df[AREA_COL], errors="coerce").clip(lower=0)
    df[LOG_AREA_COL] = np.log1p(area)

    # Put the log area column right after the original area column.
    cols = [col for col in df.columns if col != LOG_AREA_COL]
    insert_at = cols.index(AREA_COL) + 1
    df = df[cols[:insert_at] + [LOG_AREA_COL] + cols[insert_at:]]

    features = list(WEIGHTS.keys())
    x = df[features].apply(pd.to_numeric, errors="coerce").fillna(0)
    scaled = pd.DataFrame(MinMaxScaler().fit_transform(x), columns=features, index=df.index)

    df["최종_화재위험점수_로그연면적"] = (scaled[features] * pd.Series(WEIGHTS)).sum(axis=1) * 100

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df["cluster_k3_로그연면적"] = kmeans.fit_predict(scaled[features])

    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"saved={OUTPUT_PATH}")
    print(df[[LOG_AREA_COL, "최종_화재위험점수_로그연면적", "cluster_k3_로그연면적"]].describe().to_string())


if __name__ == "__main__":
    main()
