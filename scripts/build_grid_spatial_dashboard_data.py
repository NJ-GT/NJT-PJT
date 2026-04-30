# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import statsmodels.api as sm
from esda.moran import Moran
from libpysal.weights import DistanceBand, Queen
from shapely.geometry import box
from sklearn.preprocessing import StandardScaler
from spreg import GM_Error, GM_Lag


BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
OUT = DATA / "grid_spatial_dashboard"
OUT.mkdir(parents=True, exist_ok=True)

SEOUL_BOUNDARY = DATA / "seoul_neighborhoods_geo_simple.json"
FACILITY_SOURCE = DATA / "data_with_fire_targets.csv"
CRS_METER = "EPSG:5179"

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
    boundary = gpd.read_file(SEOUL_BOUNDARY).to_crs(CRS_METER)
    boundary["geometry"] = boundary.geometry.make_valid().buffer(0)
    return gpd.GeoDataFrame(geometry=[boundary.geometry.union_all()], crs=CRS_METER)


def make_grid(boundary: gpd.GeoDataFrame, cell_size: int) -> gpd.GeoDataFrame:
    minx, miny, maxx, maxy = boundary.total_bounds
    xs = np.arange(np.floor(minx / cell_size) * cell_size, maxx + cell_size, cell_size)
    ys = np.arange(np.floor(miny / cell_size) * cell_size, maxy + cell_size, cell_size)
    cells = [box(x, y, x + cell_size, y + cell_size) for x in xs[:-1] for y in ys[:-1]]
    grid = gpd.GeoDataFrame({"geometry": cells}, crs=CRS_METER)
    grid = gpd.clip(grid, boundary, keep_geom_type=True).reset_index(drop=True)
    grid["grid_id"] = [f"g{cell_size}_{i:05d}" for i in range(len(grid))]
    grid["cell_size_m"] = cell_size
    grid["area_m2"] = grid.geometry.area
    return grid


def load_facilities() -> gpd.GeoDataFrame:
    df = pd.read_csv(FACILITY_SOURCE, encoding="utf-8-sig")
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
    for col in cols.values():
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["위도", "경도"])
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["경도"], df["위도"]),
        crs="EPSG:4326",
    ).to_crs(CRS_METER)
    return gdf


def aggregate_to_grid(grid: gpd.GeoDataFrame, facilities: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    joined = gpd.sjoin(
        facilities,
        grid[["grid_id", "geometry"]],
        how="left",
        predicate="within",
    )
    grouped = joined.dropna(subset=["grid_id"]).groupby("grid_id").agg(
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
    out = grid.merge(grouped, on="grid_id", how="left")
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
    out[TARGET] = np.log1p(out["fire_sum"])
    out["facility_density_km2"] = out["facility_count"] / (out["area_m2"] / 1_000_000)
    out["centroid_x"] = out.geometry.centroid.x
    out["centroid_y"] = out.geometry.centroid.y
    return out


def build_weights(grid: gpd.GeoDataFrame, method: str):
    if method == "queen":
        w = Queen.from_dataframe(grid, ids=grid["grid_id"].tolist(), use_index=False)
    elif method == "distance_500m":
        coords = np.column_stack([grid["centroid_x"], grid["centroid_y"]])
        w = DistanceBand.from_array(coords, threshold=500, binary=True, ids=grid["grid_id"].tolist())
    else:
        raise ValueError(method)
    w.transform = "r"
    return w


def model_rows(grid: gpd.GeoDataFrame, method: str, label: str) -> list[dict]:
    use = grid.copy()
    use[FEATURES] = use[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = use[TARGET].to_numpy().reshape((-1, 1))
    x = StandardScaler().fit_transform(use[FEATURES].to_numpy(dtype=float))
    w = build_weights(use, method)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ols_sm = sm.OLS(y.flatten(), sm.add_constant(x)).fit(cov_type="HC3")
        ols_resid = ols_sm.resid
        ols_moran = Moran(ols_resid, w, permutations=499)
        lag = GM_Lag(y, x, w=w, name_y=TARGET, name_x=FEATURES)
        err = GM_Error(y, x, w=w, name_y=TARGET, name_x=FEATURES)
        lag_resid = np.asarray(lag.u).flatten()
        err_resid = np.asarray(err.u).flatten()
        lag_moran = Moran(lag_resid, w, permutations=499)
        err_moran = Moran(err_resid, w, permutations=499)

    lag_pred = y.flatten() - lag_resid
    err_pred = y.flatten() - err_resid
    tss = float(np.sum((y.flatten() - y.mean()) ** 2))

    def pseudo_r2(pred: np.ndarray) -> float:
        rss = float(np.sum((y.flatten() - pred) ** 2))
        return 1 - rss / tss if tss else np.nan

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
    boundary = load_boundary()
    facilities = load_facilities()

    all_models: list[dict] = []
    all_moran: list[dict] = []
    grids: dict[int, gpd.GeoDataFrame] = {}

    for cell_size in (250, 500):
        grid = aggregate_to_grid(make_grid(boundary, cell_size), facilities)
        grids[cell_size] = grid
        grid.to_crs("EPSG:4326").to_file(OUT / f"seoul_grid_{cell_size}m.geojson", driver="GeoJSON")
        all_moran.append(moran_rows(grid, "queen", "Queen"))
        all_models.extend(model_rows(grid, "queen", "Queen"))

    all_moran.append(moran_rows(grids[250], "distance_500m", "DistanceBand 500m"))
    all_models.extend(model_rows(grids[250], "distance_500m", "DistanceBand 500m"))

    pd.DataFrame(all_moran).to_csv(OUT / "grid_moran_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(all_models).to_csv(OUT / "grid_model_comparison.csv", index=False, encoding="utf-8-sig")
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
    (OUT / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {OUT}")


if __name__ == "__main__":
    main()
