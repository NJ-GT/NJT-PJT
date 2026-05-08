# -*- coding: utf-8 -*-
"""
서울 격자 기반 공간 회귀 비교 대시보드 입력 데이터 생성기.

목적:
    - 250m / 500m 격자 두 종류로 서울 영역을 나누고 숙박시설 변수를 집계
    - 각 격자에서 OLS / SLM(GM_Lag) / SEM(GM_Error) 비교 + Moran's I 검정
    - Queen 인접 가중치 + 250m 격자에는 DistanceBand 500m 보조 비교
    - 모란 / 모델 비교 / 격자 GeoJSON / 메타데이터를 산출 폴더에 저장

입력:
    - data/seoul_neighborhoods_geo_simple.json (서울 경계)
    - data/data_with_fire_targets.csv (숙박시설 + 화재 타깃)

출력 (data/grid_spatial_dashboard/ 하위):
    - seoul_grid_250m.geojson, seoul_grid_500m.geojson
    - grid_moran_summary.csv  격자/가중치별 Moran's I
    - grid_model_comparison.csv  격자/가중치별 OLS·SLM·SEM 비교
    - metadata.json  사용한 워크플로/입력 정보
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

# 공간 데이터 처리
import geopandas as gpd
import numpy as np
import pandas as pd
# OLS
import statsmodels.api as sm
# Moran's I
from esda.moran import Moran
# 공간 가중행렬: Queen 인접, DistanceBand
from libpysal.weights import DistanceBand, Queen
# 사각형 격자 만들기
from shapely.geometry import box
from sklearn.preprocessing import StandardScaler
# Spatial Lag / Error (GMM)
from spreg import GM_Error, GM_Lag


# 경로 상수
BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
OUT = DATA / "grid_spatial_dashboard"
OUT.mkdir(parents=True, exist_ok=True)

# 입력 파일 / 평면 좌표계
SEOUL_BOUNDARY = DATA / "seoul_neighborhoods_geo_simple.json"
FACILITY_SOURCE = DATA / "data_with_fire_targets.csv"
CRS_METER = "EPSG:5179"  # UTM-K (m 단위, 격자 만들기 좋음)

# 회귀 타깃과 X 변수
TARGET = "log1p_fire_sum"
FEATURES = [
    "facility_count",
    "mean_fire_risk",
    "mean_building_age",
    "mean_nearby_buildings",
    "mean_density",
    "mean_enforcement",
    "mean_road_risk",
    "mean_structure_age",
]


def load_boundary() -> gpd.GeoDataFrame:
    """서울 경계 GeoJSON 로드 후 평면좌표계로 변환, 단일 polygon 으로 dissolve."""
    boundary = gpd.read_file(SEOUL_BOUNDARY).to_crs(CRS_METER)
    # 부서진 도형 보정 (make_valid + buffer 0)
    boundary["geometry"] = boundary.geometry.make_valid().buffer(0)
    # 모든 행정동 경계를 union 해 서울 외곽 1개 폴리곤 반환
    return gpd.GeoDataFrame(geometry=[boundary.geometry.union_all()], crs=CRS_METER)


def make_grid(boundary: gpd.GeoDataFrame, cell_size: int) -> gpd.GeoDataFrame:
    """주어진 cell_size(m)로 서울 경계 안 정사각형 격자 GeoDataFrame 반환."""
    minx, miny, maxx, maxy = boundary.total_bounds
    # 셀 단위 정렬된 좌표 시퀀스
    xs = np.arange(np.floor(minx / cell_size) * cell_size, maxx + cell_size, cell_size)
    ys = np.arange(np.floor(miny / cell_size) * cell_size, maxy + cell_size, cell_size)
    # 격자 셀 폴리곤 생성
    cells = [box(x, y, x + cell_size, y + cell_size) for x in xs[:-1] for y in ys[:-1]]
    grid = gpd.GeoDataFrame({"geometry": cells}, crs=CRS_METER)
    # 경계 클리핑 (서울 안만 유지)
    grid = gpd.clip(grid, boundary, keep_geom_type=True).reset_index(drop=True)
    # 식별자/속성 부여
    grid["grid_id"] = [f"g{cell_size}_{i:05d}" for i in range(len(grid))]
    grid["cell_size_m"] = cell_size
    grid["area_m2"] = grid.geometry.area
    return grid


def load_facilities() -> gpd.GeoDataFrame:
    """숙박시설 csv 로드 → 컬럼 영문화 → GeoDataFrame(EPSG:5179)."""
    df = pd.read_csv(FACILITY_SOURCE, encoding="utf-8-sig")
    # 한글 → 영문 컬럼 매핑
    cols = {
        "반경100m_화재수": "fire_100m",
        "소방위험도_점수": "fire_risk",
        "건물나이": "building_age",
        "반경_50m_건물수": "nearby_buildings",
        "집중도(%)": "density",
        "고유단속지점수_50m": "enforcement",
        "도로폭_위험도": "road_risk",
        "구조_노후_통합점수": "structure_age",
    }
    keep = ["업소명", "구", "위도", "경도", *cols.keys()]
    df = df[keep].rename(columns=cols).copy()
    # 모두 숫자형 강제
    for col in cols.values():
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["위도", "경도"])
    # 좌표 → GeoDataFrame, 평면좌표로 투영
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["경도"], df["위도"]),
        crs="EPSG:4326",
    ).to_crs(CRS_METER)
    return gdf


def aggregate_to_grid(
    grid: gpd.GeoDataFrame, facilities: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """포인트(시설)을 격자 안에 spatial join → 셀별 변수 집계."""
    # 각 시설이 어느 셀(grid_id)에 속하는지 부착
    joined = gpd.sjoin(
        facilities,
        grid[["grid_id", "geometry"]],
        how="left",
        predicate="within",
    )
    # 셀별 시설수/구개수/화재합/평균지표 집계
    grouped = (
        joined.dropna(subset=["grid_id"])
        .groupby("grid_id")
        .agg(
            facility_count=("업소명", "size"),
            gu_count=("구", "nunique"),
            fire_sum=("fire_100m", "sum"),
            mean_fire_risk=("fire_risk", "mean"),
            mean_building_age=("building_age", "mean"),
            mean_nearby_buildings=("nearby_buildings", "mean"),
            mean_density=("density", "mean"),
            mean_enforcement=("enforcement", "mean"),
            mean_road_risk=("road_risk", "mean"),
            mean_structure_age=("structure_age", "mean"),
        )
    )
    out = grid.merge(grouped, on="grid_id", how="left")
    # 시설 없는 셀은 0 으로 채움
    numeric = [
        "facility_count",
        "gu_count",
        "fire_sum",
        "mean_fire_risk",
        "mean_building_age",
        "mean_nearby_buildings",
        "mean_density",
        "mean_enforcement",
        "mean_road_risk",
        "mean_structure_age",
    ]
    out[numeric] = out[numeric].fillna(0)
    # 회귀 타깃: log1p(fire_sum)
    out[TARGET] = np.log1p(out["fire_sum"])
    # 단위 면적당 시설밀도 + 셀 중심점 좌표
    out["facility_density_km2"] = out["facility_count"] / (out["area_m2"] / 1_000_000)
    out["centroid_x"] = out.geometry.centroid.x
    out["centroid_y"] = out.geometry.centroid.y
    return out


def build_weights(grid: gpd.GeoDataFrame, method: str):
    """격자 GeoDataFrame 에서 가중행렬 생성 (Queen / DistanceBand 500m)."""
    if method == "queen":
        # 변/꼭짓점 공유 인접
        w = Queen.from_dataframe(grid, ids=grid["grid_id"].tolist(), use_index=False)
    elif method == "distance_500m":
        # 셀 중심점 간 500m 이하 binary 연결
        coords = np.column_stack([grid["centroid_x"], grid["centroid_y"]])
        w = DistanceBand.from_array(
            coords, threshold=500, binary=True, ids=grid["grid_id"].tolist()
        )
    else:
        raise ValueError(method)
    w.transform = "r"  # row standardization
    return w


def model_rows(grid: gpd.GeoDataFrame, method: str, label: str) -> list[dict]:
    """OLS / SLM / SEM 한 세트를 격자 위에서 적합 → 비교 행 리스트 반환."""
    use = grid.copy()
    # inf/-inf → NaN, NaN → 0 으로 보정 (회귀 안정성)
    use[FEATURES] = use[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = use[TARGET].to_numpy().reshape((-1, 1))
    x = StandardScaler().fit_transform(use[FEATURES].to_numpy(dtype=float))
    w = build_weights(use, method)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # 1) OLS (HC3) + 잔차 Moran's I
        ols_sm = sm.OLS(y.flatten(), sm.add_constant(x)).fit(cov_type="HC3")
        ols_resid = ols_sm.resid
        ols_moran = Moran(ols_resid, w, permutations=499)
        # 2) SLM (Spatial Lag, GMM)
        lag = GM_Lag(y, x, w=w, name_y=TARGET, name_x=FEATURES)
        # 3) SEM (Spatial Error, GMM)
        err = GM_Error(y, x, w=w, name_y=TARGET, name_x=FEATURES)
        lag_resid = np.asarray(lag.u).flatten()
        err_resid = np.asarray(err.u).flatten()
        lag_moran = Moran(lag_resid, w, permutations=499)
        err_moran = Moran(err_resid, w, permutations=499)

    # 예측값 = y - 잔차
    lag_pred = y.flatten() - lag_resid
    err_pred = y.flatten() - err_resid
    tss = float(np.sum((y.flatten() - y.mean()) ** 2))

    def pseudo_r2(pred: np.ndarray) -> float:
        """SLM/SEM 의 pseudo R² (1 - RSS/TSS)."""
        rss = float(np.sum((y.flatten() - pred) ** 2))
        return 1 - rss / tss if tss else np.nan

    # 모델별 한 행씩 리스트로 반환
    return [
        {
            "grid_size_m": int(use["cell_size_m"].iloc[0]),
            "weights": label,
            "model": "OLS",
            "r2": float(ols_sm.rsquared),
            "aic": float(ols_sm.aic),
            "spatial_param": np.nan,
            "residual_moran_I": float(ols_moran.I),
            "residual_moran_p": float(ols_moran.p_sim),
            "n_cells": len(use),
            "n_active_cells": int((use["facility_count"] > 0).sum()),
            "mean_neighbors": float(np.mean(list(w.cardinalities.values()))),
            "islands": len(w.islands),
        },
        {
            "grid_size_m": int(use["cell_size_m"].iloc[0]),
            "weights": label,
            "model": "SLM",
            "r2": pseudo_r2(lag_pred),
            "aic": np.nan,
            "spatial_param": float(np.asarray(lag.rho).flatten()[0]),
            "residual_moran_I": float(lag_moran.I),
            "residual_moran_p": float(lag_moran.p_sim),
            "n_cells": len(use),
            "n_active_cells": int((use["facility_count"] > 0).sum()),
            "mean_neighbors": float(np.mean(list(w.cardinalities.values()))),
            "islands": len(w.islands),
        },
        {
            "grid_size_m": int(use["cell_size_m"].iloc[0]),
            "weights": label,
            "model": "SEM",
            "r2": pseudo_r2(err_pred),
            "aic": np.nan,
            # GM_Error 의 마지막 베타 = lambda
            "spatial_param": float(np.asarray(err.betas).flatten()[-1]),
            "residual_moran_I": float(err_moran.I),
            "residual_moran_p": float(err_moran.p_sim),
            "n_cells": len(use),
            "n_active_cells": int((use["facility_count"] > 0).sum()),
            "mean_neighbors": float(np.mean(list(w.cardinalities.values()))),
            "islands": len(w.islands),
        },
    ]


def moran_rows(grid: gpd.GeoDataFrame, method: str, label: str) -> dict:
    """격자 + 가중치별 Y 의 Moran's I 1행 dict 반환."""
    w = build_weights(grid, method)
    mi = Moran(grid[TARGET].to_numpy(), w, permutations=499)
    return {
        "grid_size_m": int(grid["cell_size_m"].iloc[0]),
        "weights": label,
        "moran_I": float(mi.I),
        "p_value": float(mi.p_sim),
        "z_score": float(mi.z_sim),
        "n_cells": len(grid),
        "n_active_cells": int((grid["facility_count"] > 0).sum()),
        "mean_neighbors": float(np.mean(list(w.cardinalities.values()))),
        "islands": len(w.islands),
    }


def main() -> None:
    """경계 + 시설 로드 → 250/500m 격자 처리 → 결과 csv/json 저장."""
    boundary = load_boundary()
    facilities = load_facilities()

    all_models: list[dict] = []
    all_moran: list[dict] = []
    grids: dict[int, gpd.GeoDataFrame] = {}

    # 두 가지 셀 크기로 반복 (250m / 500m)
    for cell_size in (250, 500):
        grid = aggregate_to_grid(make_grid(boundary, cell_size), facilities)
        grids[cell_size] = grid
        # 격자 GeoJSON 저장 (대시보드에서 시각화용)
        grid.to_crs("EPSG:4326").to_file(
            OUT / f"seoul_grid_{cell_size}m.geojson", driver="GeoJSON"
        )
        # Queen 가중치로 Moran 검정 + 모델 비교
        all_moran.append(moran_rows(grid, "queen", "Queen"))
        all_models.extend(model_rows(grid, "queen", "Queen"))

    # 250m 격자에 대해서는 DistanceBand 500m 보조 비교 추가
    all_moran.append(moran_rows(grids[250], "distance_500m", "DistanceBand 500m"))
    all_models.extend(model_rows(grids[250], "distance_500m", "DistanceBand 500m"))

    # 결과 CSV 저장
    pd.DataFrame(all_moran).to_csv(
        OUT / "grid_moran_summary.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(all_models).to_csv(
        OUT / "grid_model_comparison.csv", index=False, encoding="utf-8-sig"
    )
    # 메타데이터 (워크플로 흐름) 기록
    metadata = {
        "boundary": str(SEOUL_BOUNDARY.relative_to(BASE)),
        "facility_source": str(FACILITY_SOURCE.relative_to(BASE)),
        "crs_meter": CRS_METER,
        "target": TARGET,
        "features": FEATURES,
        "workflow": [
            "250m grid 생성",
            "spatial join으로 변수 집계",
            "Moran's I 확인",
            "W = Queen 인접, DistanceBand(500m) 보조 확인",
            "OLS -> SLM/SEM 비교",
            "500m grid robustness check",
        ],
    }
    (OUT / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"saved: {OUT}")


if __name__ == "__main__":
    main()
