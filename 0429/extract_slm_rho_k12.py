from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from libpysal.weights import KNN
from sklearn.preprocessing import StandardScaler
from spreg import ML_Lag


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "0424" / "data" / "cluster3_spatial_pipeline_fire_count_150m_0428" / "최최최종0428변수테이블.csv"
FIRE_TARGET_PATH = ROOT / "data" / "team_pipeline_validation" / "team_pipeline_scored_dataset.csv"
OUT_PATH = ROOT / "0429" / "slm_rho_k12_fire_count_150m_by_cluster.csv"

TARGET = "fire_count_150m"
CLUSTER_COL = "cluster"
COORD_COLS = ["x_5181", "y_5181"]
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
    return s.astype(str).str.replace(r"\s+", "", regex=True).str.strip()


def attach_fire_target(df: pd.DataFrame) -> pd.DataFrame:
    if TARGET in df.columns:
        return df
    fire = pd.read_csv(FIRE_TARGET_PATH, encoding="utf-8-sig")
    fire_key = pd.DataFrame(
        {
            "_name_key": name_key(fire["숙소명"]),
            "_lat_key": pd.to_numeric(fire["위도"], errors="coerce").round(6),
            "_lon_key": pd.to_numeric(fire["경도"], errors="coerce").round(6),
            TARGET: pd.to_numeric(fire[TARGET], errors="coerce"),
        }
    ).drop_duplicates(["_name_key", "_lat_key", "_lon_key"])
    keyed = df.copy()
    keyed["_name_key"] = name_key(keyed["숙소명"])
    keyed["_lat_key"] = pd.to_numeric(keyed["위도"], errors="coerce").round(6)
    keyed["_lon_key"] = pd.to_numeric(keyed["경도"], errors="coerce").round(6)
    keyed = keyed.merge(fire_key, on=["_name_key", "_lat_key", "_lon_key"], how="left")
    return keyed.drop(columns=["_name_key", "_lat_key", "_lon_key"], errors="ignore")


def main() -> None:
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    df = attach_fire_target(df)
    for col in FEATURES + [TARGET, CLUSTER_COL] + COORD_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=FEATURES + [TARGET, CLUSTER_COL] + COORD_COLS).reset_index(drop=True)

    rows = []
    for cluster_id in sorted(df[CLUSTER_COL].astype(int).unique()):
        sub = df[df[CLUSTER_COL].astype(int).eq(cluster_id)].reset_index(drop=True)
        x = StandardScaler().fit_transform(sub[FEATURES].to_numpy(dtype=float))
        y = sub[TARGET].to_numpy(dtype=float).reshape(-1, 1)
        coords = sub[COORD_COLS].to_numpy(dtype=float)
        k = min(12, max(1, len(coords) - 1))
        w = KNN.from_array(coords, k=k)
        w.transform = "r"
        model = ML_Lag(y, x, w=w, name_y=TARGET, name_x=FEATURES)
        rho = float(np.asarray(model.rho).reshape(-1)[0])
        rows.append(
            {
                "cluster": cluster_id,
                "n": len(sub),
                "knn_k": k,
                "rho": rho,
                "pseudo_r2": float(getattr(model, "pr2", np.nan)),
                "aic": float(getattr(model, "aic", np.nan)),
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(OUT_PATH)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
