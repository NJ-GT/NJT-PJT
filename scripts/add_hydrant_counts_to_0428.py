# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree


BASE = Path(__file__).resolve().parents[1]
LODGING_PATH = BASE / "0424" / "data" / "분석변수_최종테이블0428.csv"
HYDRANT_PATH = BASE / "0424" / "서울시 소화용수 위치정보 (좌표계_ ITRF2000).csv"


def main() -> None:
    lodging = pd.read_csv(LODGING_PATH, encoding="utf-8-sig")
    hydrant = pd.read_csv(HYDRANT_PATH, encoding="utf-8-sig")

    for col in ["위도", "경도"]:
        lodging[col] = pd.to_numeric(lodging[col], errors="coerce")
    for col in ["X좌표", "Y좌표"]:
        hydrant[col] = pd.to_numeric(hydrant[col], errors="coerce")

    valid_lodging = lodging[["위도", "경도"]].notna().all(axis=1)
    valid_hydrant = hydrant[["X좌표", "Y좌표"]].notna().all(axis=1)

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:5181", always_xy=True)
    x_5181, y_5181 = transformer.transform(
        lodging.loc[valid_lodging, "경도"].to_numpy(),
        lodging.loc[valid_lodging, "위도"].to_numpy(),
    )

    lodging["x_5181"] = np.nan
    lodging["y_5181"] = np.nan
    lodging.loc[valid_lodging, "x_5181"] = x_5181
    lodging.loc[valid_lodging, "y_5181"] = y_5181

    # The source contains many different addresses sharing one identical coordinate.
    # Count unique water-supply locations so one bad duplicated point does not dominate.
    hydrant_unique = hydrant.loc[valid_hydrant, ["X좌표", "Y좌표"]].drop_duplicates()
    hyd_xy = hydrant_unique.to_numpy(dtype=float)
    lod_xy = lodging.loc[valid_lodging, ["x_5181", "y_5181"]].to_numpy(dtype=float)
    tree = cKDTree(hyd_xy)
    nearest_dist, _ = tree.query(lod_xy, k=1)

    grade_col = "최근접_소화용수_거리등급"
    lodging[grade_col] = pd.NA
    lodging.loc[valid_lodging, grade_col] = np.select(
        [
            nearest_dist <= 20,
            (nearest_dist > 20) & (nearest_dist <= 40),
            (nearest_dist > 40) & (nearest_dist <= 100),
        ],
        [0, 1, 2],
        default=2,
    ).astype(int)
    lodging[grade_col] = lodging[grade_col].astype("Int64")

    old_count_cols = [
        "소화용수_40m초과_100m이내_개수",
        "소화용수_40m이내_개수",
        "소화용수_20m이내_개수",
    ]
    lodging = lodging.drop(columns=old_count_cols, errors="ignore")

    new_cols = ["x_5181", "y_5181", grade_col]
    old_cols = [c for c in lodging.columns if c not in new_cols]
    insert_at = old_cols.index("경도") + 1
    lodging = lodging[old_cols[:insert_at] + new_cols + old_cols[insert_at:]]

    lodging.to_csv(LODGING_PATH, index=False, encoding="utf-8-sig")

    print(f"updated={LODGING_PATH}")
    print(f"lodging_rows={len(lodging)} valid_lodging_coords={int(valid_lodging.sum())}")
    print(f"hydrant_rows={len(hydrant)} valid_hydrant_coords={int(valid_hydrant.sum())}")
    print(f"unique_hydrant_xy={len(hydrant_unique)}")
    print(
        lodging[
            [
                "숙소명",
                "위도",
                "경도",
                "x_5181",
                "y_5181",
                grade_col,
            ]
        ]
        .head(10)
        .to_string(index=False)
    )
    print("\ngrade_counts")
    print(lodging[grade_col].value_counts(dropna=False).sort_index().to_string())


if __name__ == "__main__":
    main()
