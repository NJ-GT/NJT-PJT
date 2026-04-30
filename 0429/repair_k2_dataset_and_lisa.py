from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "0424" / "data" / "cluster3_spatial_pipeline_fire_count_150m_0428"
FIRE_TARGET_PATH = ROOT / "data" / "team_pipeline_validation" / "team_pipeline_scored_dataset.csv"
K2_DIR = ROOT / "0429" / "cluster2_spatial_pipeline_fire_count_150m_0429"

TARGET = "fire_count_150m"
CLUSTER_FEATURES = [
    "최종_화재위험점수",
    "도로폭위험도",
    "집중도",
    "주변건물수",
    "최근접_소화용수_거리등급",
]
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
    return s.astype(str).str.replace(r"\s+", "", regex=True).str.strip()


def main() -> None:
    main_csv = max(DATA_DIR.glob("*.csv"), key=lambda p: p.stat().st_size)
    df = pd.read_csv(main_csv, encoding="utf-8-sig")
    fire = pd.read_csv(FIRE_TARGET_PATH, encoding="utf-8-sig")

    fire_key = pd.DataFrame(
        {
            "_name_key": name_key(fire["숙소명"]),
            "_lat_key": pd.to_numeric(fire["위도"], errors="coerce").round(6),
            "_lon_key": pd.to_numeric(fire["경도"], errors="coerce").round(6),
            TARGET: pd.to_numeric(fire[TARGET], errors="coerce"),
        }
    ).drop_duplicates(["_name_key", "_lat_key", "_lon_key"])

    out = df.copy()
    out["_name_key"] = name_key(out["숙소명"])
    out["_lat_key"] = pd.to_numeric(out["위도"], errors="coerce").round(6)
    out["_lon_key"] = pd.to_numeric(out["경도"], errors="coerce").round(6)
    out = out.merge(fire_key, on=["_name_key", "_lat_key", "_lon_key"], how="left")
    out = out.drop(columns=["_name_key", "_lat_key", "_lon_key"], errors="ignore")

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
    for col in [TARGET, "경도", "위도", "x_5181", "y_5181", *REG_FEATURES, "최종_화재위험점수"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out[keep].dropna(subset=[TARGET, "경도", "위도", "x_5181", "y_5181", *REG_FEATURES]).copy()
    out = out.drop_duplicates(["숙소명", "경도", "위도", "x_5181", "y_5181"]).reset_index(drop=True)

    x = StandardScaler().fit_transform(out[CLUSTER_FEATURES].to_numpy(dtype=float))
    out["cluster_k2"] = KMeans(n_clusters=2, random_state=42, n_init=50).fit_predict(x)

    # Keep the earlier interpretation: cluster 1 should be the high-risk/high-density group.
    risk_mean = out.groupby("cluster_k2")["최종_화재위험점수"].mean()
    high_label = int(risk_mean.idxmax())
    if high_label != 1:
        out["cluster_k2"] = 1 - out["cluster_k2"]

    out_path = K2_DIR / "최최최종0428변수테이블_cluster_k2.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(out_path)
    print(out.groupby("cluster_k2").size().to_string())
    print("total", len(out))


if __name__ == "__main__":
    main()
