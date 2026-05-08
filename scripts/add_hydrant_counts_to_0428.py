# -*- coding: utf-8 -*-
"""
0428 분석변수 테이블에 EPSG:5181 평면좌표 + 소화용수 거리 등급 컬럼을 부착.

부착 컬럼:
    - x_5181, y_5181  : 한국 중부원점 평면좌표 (m)
    - 최근접_소화용수_거리등급 : 0(≤20m) / 1(≤40m) / 2(>40m, 또는 ≤100m로 표기)

특이 사항:
    소화용수 좌표는 같은 위치가 여러 행에 중복으로 들어가는 경우가 많다.
    cKDTree에서 중복 좌표가 결과를 왜곡하지 않도록 drop_duplicates 후 인덱싱한다.
    이전 단계에서 별도로 만들었던 카운트 컬럼들은 본 스크립트에서 제거한다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
# 좌표 변환 (EPSG:4326 -> EPSG:5181)
from pyproj import Transformer
# 최근접 이웃 거리 계산 (KD-Tree)
from scipy.spatial import cKDTree


# scripts/ 기준 한 단계 위
BASE = Path(__file__).resolve().parents[1]
LODGING_PATH = BASE / "0424" / "data" / "분석변수_최종테이블0428.csv"
HYDRANT_PATH = BASE / "0424" / "서울시 소화용수 위치정보 (좌표계_ ITRF2000).csv"


def main() -> None:
    """평면좌표 + 거리 등급 부착 후 in-place 갱신."""
    lodging = pd.read_csv(LODGING_PATH, encoding="utf-8-sig")
    hydrant = pd.read_csv(HYDRANT_PATH, encoding="utf-8-sig")

    # 좌표 컬럼 수치 변환
    for col in ["위도", "경도"]:
        lodging[col] = pd.to_numeric(lodging[col], errors="coerce")
    for col in ["X좌표", "Y좌표"]:
        hydrant[col] = pd.to_numeric(hydrant[col], errors="coerce")

    # 좌표가 모두 있는 행만 KD-Tree 입력 후보로 사용
    valid_lodging = lodging[["위도", "경도"]].notna().all(axis=1)
    valid_hydrant = hydrant[["X좌표", "Y좌표"]].notna().all(axis=1)

    # 위경도 -> 미터 좌표(EPSG:5181)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:5181", always_xy=True)
    x_5181, y_5181 = transformer.transform(
        lodging.loc[valid_lodging, "경도"].to_numpy(),
        lodging.loc[valid_lodging, "위도"].to_numpy(),
    )

    # 평면좌표 컬럼 부착 (좌표 결측 행은 NaN 그대로)
    lodging["x_5181"] = np.nan
    lodging["y_5181"] = np.nan
    lodging.loc[valid_lodging, "x_5181"] = x_5181
    lodging.loc[valid_lodging, "y_5181"] = y_5181

    # 소화용수 좌표는 중복이 많음 — 고유 위치만 KD-Tree에 사용
    hydrant_unique = hydrant.loc[valid_hydrant, ["X좌표", "Y좌표"]].drop_duplicates()
    hyd_xy = hydrant_unique.to_numpy(dtype=float)
    lod_xy = lodging.loc[valid_lodging, ["x_5181", "y_5181"]].to_numpy(dtype=float)
    tree = cKDTree(hyd_xy)
    # 각 숙소 -> 최근접 소화용수 거리(m)
    nearest_dist, _ = tree.query(lod_xy, k=1)

    # 거리 등급 (3단계)
    grade_col = "최근접_소화용수_거리등급"
    lodging[grade_col] = pd.NA
    lodging.loc[valid_lodging, grade_col] = np.select(
        [
            nearest_dist <= 20,
            (nearest_dist > 20) & (nearest_dist <= 40),
            (nearest_dist > 40) & (nearest_dist <= 100),
        ],
        [0, 1, 2],
        # 100m 초과도 일단 2로 둠 (취약 구간으로 통합)
        default=2,
    ).astype(int)
    lodging[grade_col] = lodging[grade_col].astype("Int64")

    # 이전 단계에서 만들었던 카운트 컬럼들은 정리(중복 정보 제거)
    old_count_cols = [
        "소화용수_40m초과_100m이내_개수",
        "소화용수_40m이내_개수",
        "소화용수_20m이내_개수",
    ]
    lodging = lodging.drop(columns=old_count_cols, errors="ignore")

    # 새 컬럼은 '경도' 바로 뒤에 위치하도록 컬럼 순서 정렬
    new_cols = ["x_5181", "y_5181", grade_col]
    old_cols = [c for c in lodging.columns if c not in new_cols]
    insert_at = old_cols.index("경도") + 1
    lodging = lodging[old_cols[:insert_at] + new_cols + old_cols[insert_at:]]

    # 원본 경로에 덮어쓰기
    lodging.to_csv(LODGING_PATH, index=False, encoding="utf-8-sig")

    # 검증 출력 — 행 수, 좌표 유효 개수, 등급 분포, 일부 미리보기
    print(f"updated={LODGING_PATH}")
    print(
        f"lodging_rows={len(lodging)} valid_lodging_coords={int(valid_lodging.sum())}"
    )
    print(
        f"hydrant_rows={len(hydrant)} valid_hydrant_coords={int(valid_hydrant.sum())}"
    )
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
