# -*- coding: utf-8 -*-
"""
분석변수_최종테이블0423.csv 에 '소화용수'(거리 등급) 컬럼을 추가하는 스크립트.

거리 등급 정의 (소화용수 점에서 가장 가까운 거리 d, EPSG:5181 미터 단위):
    d ≤ 20m       -> 0  (양호)
    20m < d ≤ 40m -> 1  (보통)
    d > 40m       -> 2  (취약)

처리 흐름:
    1) 분석 테이블과 소화용수 위치 CSV 로드 + 좌표 수치 변환
    2) 위경도(EPSG:4326) -> EPSG:5181 변환으로 모든 좌표를 미터 단위로 통일
    3) cKDTree 로 가장 가까운 소화용수 거리 산출
    4) 임계값 기준으로 등급 분류 후 '소화용수' 컬럼으로 부착
    5) 결과를 원본 CSV에 덮어쓰기

출력:
    BASE/0424/data/분석변수_최종테이블0423.csv (in-place 갱신)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
# 좌표 변환 (EPSG:4326 -> EPSG:5181)
from pyproj import Transformer
# 최근접 이웃 거리 계산 (벡터화)
from scipy.spatial import cKDTree


# scripts 기준 한 단계 위 — 프로젝트 루트
BASE = Path(__file__).resolve().parents[1]
# 분석 변수 메인 테이블 (in-place 갱신 대상)
TARGET_PATH = BASE / "0424" / "data" / "분석변수_최종테이블0423.csv"
# 서울시 소화용수 위치 (좌표는 ITRF2000 = EPSG:5181 호환)
HYDRANT_PATH = BASE / "0424" / "서울시 소화용수 위치정보 (좌표계_ ITRF2000).csv"


def main() -> None:
    """소화용수 거리 등급을 부착한 후 원본 CSV에 덮어쓰기."""
    target = pd.read_csv(TARGET_PATH, encoding="utf-8-sig")
    hydrant = pd.read_csv(HYDRANT_PATH, encoding="utf-8-sig")

    # 좌표 컬럼을 수치형으로 강제 변환 (errors="coerce" -> NaN으로)
    for col in ["위도", "경도"]:
        target[col] = pd.to_numeric(target[col], errors="coerce")
    for col in ["X좌표", "Y좌표"]:
        hydrant[col] = pd.to_numeric(hydrant[col], errors="coerce")

    # 좌표가 모두 있는 행만 분석 대상
    valid_target = target[["위도", "경도"]].notna().all(axis=1)
    valid_hydrant = hydrant[["X좌표", "Y좌표"]].notna().all(axis=1)

    # 분석 테이블의 위경도(EPSG:4326) -> EPSG:5181(미터) 변환
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:5181", always_xy=True)
    x_5181, y_5181 = transformer.transform(
        target.loc[valid_target, "경도"].to_numpy(),
        target.loc[valid_target, "위도"].to_numpy(),
    )
    target_xy = np.column_stack([x_5181, y_5181])

    # 소화용수 좌표는 이미 ITRF2000 — 중복 제거 후 그대로 사용
    hydrant_xy = (
        hydrant.loc[valid_hydrant, ["X좌표", "Y좌표"]]
        .drop_duplicates()
        .to_numpy(dtype=float)
    )
    # 각 분석 행에서 가장 가까운 소화용수까지의 거리(m)
    nearest_dist, _ = cKDTree(hydrant_xy).query(target_xy, k=1)

    # 0/1/2 등급 부여 (NaN은 그대로 둠)
    target["소화용수"] = pd.NA
    target.loc[valid_target, "소화용수"] = np.select(
        [
            nearest_dist <= 20,
            (nearest_dist > 20) & (nearest_dist <= 40),
        ],
        [0, 1],
        default=2,
    ).astype(int)
    # 정수 등급(NaN 허용)으로 dtype 고정
    target["소화용수"] = target["소화용수"].astype("Int64")

    # 새 컬럼은 '경도' 바로 뒤에 위치하도록 컬럼 순서 재배열
    cols = [c for c in target.columns if c != "소화용수"]
    insert_at = cols.index("경도") + 1
    target = target[cols[:insert_at] + ["소화용수"] + cols[insert_at:]]

    # 원본 경로에 덮어쓰기
    target.to_csv(TARGET_PATH, index=False, encoding="utf-8-sig")

    # 검증 출력 — 행수, 좌표 유효 개수, 등급 분포, 일부 미리보기
    print(f"updated={TARGET_PATH}")
    print(f"rows={len(target)} valid_target_coords={int(valid_target.sum())}")
    print(f"unique_hydrant_xy={len(hydrant_xy)}")
    print(target["소화용수"].value_counts(dropna=False).sort_index().to_string())
    print(
        target[["숙소명", "위도", "경도", "소화용수"]].head(10).to_string(index=False)
    )


if __name__ == "__main__":
    main()
