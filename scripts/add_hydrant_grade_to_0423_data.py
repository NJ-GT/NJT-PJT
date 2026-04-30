# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree


BASE = Path(__file__).resolve().parents[1]
TARGET_PATH = BASE / "data" / "분석변수_최종테이블0423.csv"
HYDRANT_PATH = BASE / "0424" / "서울시 소화용수 위치정보 (좌표계_ ITRF2000).csv"


def main() -> None:
    target = pd.read_csv(TARGET_PATH, encoding="utf-8-sig")
    hydrant = pd.read_csv(HYDRANT_PATH, encoding="utf-8-sig")

    for col in ["위도", "경도"]:
        target[col] = pd.to_numeric(target[col], errors="coerce")
    for col in ["X좌표", "Y좌표"]:
        hydrant[col] = pd.to_numeric(hydrant[col], errors="coerce")

    valid_target = target[["위도", "경도"]].notna().all(axis=1)
    valid_hydrant = hydrant[["X좌표", "Y좌표"]].notna().all(axis=1)

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:5181", always_xy=True)
    x_5181, y_5181 = transformer.transform(
        target.loc[valid_target, "경도"].to_numpy(),
        target.loc[valid_target, "위도"].to_numpy(),
    )
    target_xy = np.column_stack([x_5181, y_5181])

    hydrant_xy = hydrant.loc[valid_hydrant, ["X좌표", "Y좌표"]].drop_duplicates().to_numpy(dtype=float)
    nearest_dist, _ = cKDTree(hydrant_xy).query(target_xy, k=1)

    target["소화용수"] = pd.NA
    target.loc[valid_target, "소화용수"] = np.select(
        [
            nearest_dist <= 20,
            (nearest_dist > 20) & (nearest_dist <= 40),
        ],
        [0, 1],
        default=2,
    ).astype(int)
    target["소화용수"] = target["소화용수"].astype("Int64")

    # Place the new feature after 경도, keeping the original table otherwise unchanged.
    cols = [c for c in target.columns if c != "소화용수"]
    insert_at = cols.index("경도") + 1
    target = target[cols[:insert_at] + ["소화용수"] + cols[insert_at:]]

    target.to_csv(TARGET_PATH, index=False, encoding="utf-8-sig")

    print(f"updated={TARGET_PATH}")
    print(f"rows={len(target)} valid_target_coords={int(valid_target.sum())}")
    print(f"unique_hydrant_xy={len(hydrant_xy)}")
    print(target["소화용수"].value_counts(dropna=False).sort_index().to_string())
    print(target[["숙소명", "위도", "경도", "소화용수"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
