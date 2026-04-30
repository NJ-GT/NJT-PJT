# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.neighbors import BallTree


BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "data" / "final_spatial_pipeline"
OUT.mkdir(parents=True, exist_ok=True)

TARGET_FILES = [
    BASE / "0424" / "분석" / "tables" / "분석변수_최종테이블0423_AHP3등급비교.csv",
    BASE / "data" / "clustering_result_all.csv",
    BASE / "data" / "final_spatial_pipeline" / "analysis_dataset.csv",
]
BUILDING_SHP = BASE / "data" / "AL_D010_11_20260409" / "AL_D010_11_20260409_filtered.shp"

BUILDING_COUNT_COLS = [
    "주변건물수_50m",
    "반경_50m_건물수",
    "주변건물수",
]
CONCENTRATION_COLS = [
    "집중도(%)",
    "집중도",
]
NAME_COLS = ["숙소명", "업소명", "사업장명"]
LAT_COLS = ["위도", "lat", "latitude"]
LON_COLS = ["경도", "lon", "lng", "longitude"]
ADDRESS_COLS = ["주소", "도로명대지위치", "대지위치"]


def read_table(path: Path, nrows: int | None = None) -> pd.DataFrame | None:
    try:
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path, encoding="utf-8-sig", nrows=nrows, low_memory=False)
        if path.suffix.lower() in {".xlsx", ".xls"}:
            return pd.read_excel(path, nrows=nrows)
    except Exception:
        return None
    return None


def norm_name(value: object) -> str:
    if pd.isna(value):
        return ""
    return "".join(str(value).split()).strip().lower()


def find_col(columns: list[str], candidates: list[str]) -> str | None:
    for col in candidates:
        if col in columns:
            return col
    return None


def inventory_sources() -> pd.DataFrame:
    rows = []
    for path in BASE.rglob("*"):
        if path.suffix.lower() not in {".csv", ".xlsx", ".xls"}:
            continue
        path_text = str(path)
        if "final_spatial_pipeline" in path_text or "_주변건물수보정" in path_text:
            continue
        df = read_table(path, nrows=3)
        if df is None:
            continue
        cols = [str(c).strip() for c in df.columns]
        count_cols = [c for c in cols if c in BUILDING_COUNT_COLS or "건물수" in c]
        conc_cols = [c for c in cols if c in CONCENTRATION_COLS or "집중도" in c]
        name_cols = [c for c in cols if c in NAME_COLS]
        lat_cols = [c for c in cols if c in LAT_COLS or c in {"X좌표", "x_5181"}]
        lon_cols = [c for c in cols if c in LON_COLS or c in {"Y좌표", "y_5181"}]
        if count_cols:
            rows.append(
                {
                    "path": str(path.relative_to(BASE)),
                    "count_cols": "|".join(count_cols),
                    "concentration_cols": "|".join(conc_cols),
                    "name_cols": "|".join(name_cols),
                    "coord_cols": "|".join(lat_cols + lon_cols),
                    "n_cols": len(cols),
                }
            )
    result = pd.DataFrame(rows).sort_values(["path"])
    result.to_csv(OUT / "nearby_building_count_source_inventory.csv", index=False, encoding="utf-8-sig")
    return result


def candidate_priority(path: Path, count_col: str) -> int:
    p = str(path).replace("\\", "/")
    if "주변노후도_50m100m_사용승인일유효" in p and count_col == "주변건물수_50m":
        return 100
    if "주변노후도_50m100m_연면적유효" in p and count_col == "주변건물수_50m":
        return 95
    if "주변노후도_50m100m.csv" in p and count_col == "주변건물수_50m":
        return 90
    if "서울10구_숙소_소방거리_유클리드.csv" in p and count_col == "반경_50m_건물수":
        return 80
    if "data_with_fire_targets.csv" in p and count_col == "반경_50m_건물수":
        return 75
    if count_col in {"주변건물수_50m", "반경_50m_건물수"}:
        return 60
    return 20


def collect_candidates() -> pd.DataFrame:
    frames = []
    for path in BASE.rglob("*"):
        if path.suffix.lower() not in {".csv", ".xlsx", ".xls"}:
            continue
        path_text = str(path)
        if "final_spatial_pipeline" in path_text or "_주변건물수보정" in path_text:
            continue
        head = read_table(path, nrows=3)
        if head is None:
            continue
        cols = [str(c).strip() for c in head.columns]
        count_col = find_col(cols, BUILDING_COUNT_COLS)
        name_col = find_col(cols, NAME_COLS)
        lat_col = find_col(cols, LAT_COLS)
        lon_col = find_col(cols, LON_COLS)
        if not count_col or (not name_col and not (lat_col and lon_col)):
            continue
        df = read_table(path)
        if df is None or count_col not in df.columns:
            continue

        conc_col = find_col(list(df.columns), CONCENTRATION_COLS)
        addr_col = find_col(list(df.columns), ADDRESS_COLS)

        cand = pd.DataFrame(
            {
                "_name_key": df[name_col].map(norm_name) if name_col else "",
                "후보_주변건물수": pd.to_numeric(df[count_col], errors="coerce"),
                "후보_집중도": pd.to_numeric(df[conc_col], errors="coerce") if conc_col else np.nan,
                "후보_파일": str(path.relative_to(BASE)),
                "후보_컬럼": count_col,
                "후보_우선순위": candidate_priority(path, count_col),
            }
        )
        if lat_col and lon_col:
            cand["_lat_key"] = pd.to_numeric(df[lat_col], errors="coerce").round(6)
            cand["_lon_key"] = pd.to_numeric(df[lon_col], errors="coerce").round(6)
        else:
            cand["_lat_key"] = np.nan
            cand["_lon_key"] = np.nan
        cand["후보_주소"] = df[addr_col] if addr_col else ""
        has_name = cand["_name_key"].ne("")
        has_coord = cand["_lat_key"].notna() & cand["_lon_key"].notna()
        cand = cand[(has_name | has_coord) & cand["후보_주변건물수"].notna()]
        frames.append(cand)

    if not frames:
        return pd.DataFrame()

    all_cand = pd.concat(frames, ignore_index=True)
    all_cand = all_cand.sort_values(
        ["_name_key", "후보_우선순위", "후보_주변건물수"], ascending=[True, False, False]
    )
    all_cand.to_csv(OUT / "nearby_building_count_all_candidates.csv", index=False, encoding="utf-8-sig")
    return all_cand


def best_candidates(candidates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    candidates = candidates[pd.to_numeric(candidates["후보_주변건물수"], errors="coerce").gt(0)].copy()
    coord = candidates[candidates["_name_key"].ne("")].dropna(subset=["_lat_key", "_lon_key"]).copy()
    coord = coord.sort_values(
        ["_name_key", "_lat_key", "_lon_key", "후보_우선순위"], ascending=[True, True, True, False]
    )
    coord_best = coord.drop_duplicates(["_name_key", "_lat_key", "_lon_key"], keep="first")

    coord_only = candidates.dropna(subset=["_lat_key", "_lon_key"]).copy()
    coord_only = coord_only.sort_values(["_lat_key", "_lon_key", "후보_우선순위"], ascending=[True, True, False])
    coord_only_best = coord_only.drop_duplicates(["_lat_key", "_lon_key"], keep="first")

    name_best = candidates[candidates["_name_key"].ne("")].sort_values(
        ["_name_key", "후보_우선순위"], ascending=[True, False]
    )
    name_best = name_best.drop_duplicates("_name_key", keep="first")
    return coord_best, coord_only_best, name_best


def use_group(code: object) -> str:
    value = str(code).strip().split(".")[0].zfill(5)
    if value in {"01000", "02000"}:
        return "주택"
    if value in {"03000", "04000", "07000"}:
        return "상업"
    if value == "15000":
        return "숙박"
    if value == "14000":
        return "사무"
    return "기타"


def direct_building_counts(df: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    result = pd.DataFrame(index=df.index, columns=["직접산출_주변건물수", "직접산출_집중도"])
    if not mask.any() or not BUILDING_SHP.exists():
        return result

    buildings = gpd.read_file(BUILDING_SHP, columns=["A8", "geometry"])
    buildings = buildings[buildings.geometry.notna() & ~buildings.geometry.is_empty].copy()
    buildings["용도그룹"] = buildings["A8"].map(use_group)
    targets = df.loc[mask, ["위도", "경도"]].copy()
    target_gdf = gpd.GeoDataFrame(
        targets,
        geometry=gpd.points_from_xy(targets["경도"], targets["위도"]),
        crs="EPSG:4326",
    ).to_crs(buildings.crs)
    buffers = target_gdf.geometry.buffer(50)
    spatial_index = buildings.sindex

    for target_index, buffer_geom in zip(targets.index, buffers):
        candidate_idx = list(spatial_index.intersection(buffer_geom.bounds))
        if not candidate_idx:
            continue
        nearby = buildings.iloc[candidate_idx]
        nearby = nearby[nearby.geometry.intersects(buffer_geom)]
        count = int(len(nearby))
        if count <= 0:
            continue
        groups = nearby["용도그룹"].value_counts()
        result.loc[target_index, "직접산출_주변건물수"] = count
        result.loc[target_index, "직접산출_집중도"] = float(groups.iloc[0] / count * 100)
    return result


def repair_target(path: Path, coord_best: pd.DataFrame, coord_only_best: pd.DataFrame, name_best: pd.DataFrame) -> dict:
    df = pd.read_csv(path, encoding="utf-8-sig")
    name_col = "숙소명" if "숙소명" in df.columns else "업소명"
    df["_name_key"] = df[name_col].map(norm_name)
    df["_lat_key"] = pd.to_numeric(df["위도"], errors="coerce").round(6) if "위도" in df.columns else np.nan
    df["_lon_key"] = pd.to_numeric(df["경도"], errors="coerce").round(6) if "경도" in df.columns else np.nan

    original = pd.to_numeric(df["주변건물수"], errors="coerce") if "주변건물수" in df.columns else pd.Series(np.nan, index=df.index)
    original_conc = pd.to_numeric(df["집중도"], errors="coerce") if "집중도" in df.columns else pd.Series(np.nan, index=df.index)
    suspect = original.eq(0) | original_conc.eq(0) | original.isna()

    merged = df.merge(
        coord_best[
            ["_name_key", "_lat_key", "_lon_key", "후보_주변건물수", "후보_집중도", "후보_파일", "후보_컬럼", "후보_우선순위"]
        ],
        on=["_name_key", "_lat_key", "_lon_key"],
        how="left",
    )
    no_coord = merged["후보_주변건물수"].isna() | pd.to_numeric(
        merged["후보_주변건물수"], errors="coerce"
    ).le(0)
    coord_only_merge = df[["_lat_key", "_lon_key"]].merge(
        coord_only_best[
            ["_lat_key", "_lon_key", "후보_주변건물수", "후보_집중도", "후보_파일", "후보_컬럼", "후보_우선순위"]
        ],
        on=["_lat_key", "_lon_key"],
        how="left",
    )
    for col in ["후보_주변건물수", "후보_집중도", "후보_파일", "후보_컬럼", "후보_우선순위"]:
        merged.loc[no_coord, col] = coord_only_merge.loc[no_coord, col].to_numpy()

    no_coord = merged["후보_주변건물수"].isna() | pd.to_numeric(
        merged["후보_주변건물수"], errors="coerce"
    ).le(0)
    name_merge = df[["_name_key"]].merge(
        name_best[["_name_key", "후보_주변건물수", "후보_집중도", "후보_파일", "후보_컬럼", "후보_우선순위"]],
        on="_name_key",
        how="left",
    )
    for col in ["후보_주변건물수", "후보_집중도", "후보_파일", "후보_컬럼", "후보_우선순위"]:
        merged.loc[no_coord, col] = name_merge.loc[no_coord, col].to_numpy()

    no_candidate = merged["후보_주변건물수"].isna() | pd.to_numeric(
        merged["후보_주변건물수"], errors="coerce"
    ).le(0)
    needs_nearest = suspect & no_candidate & df["_lat_key"].notna() & df["_lon_key"].notna()
    nearest_source = coord_only_best.dropna(subset=["_lat_key", "_lon_key"]).copy()
    nearest_source = nearest_source[
        pd.to_numeric(nearest_source["후보_주변건물수"], errors="coerce").gt(0)
    ].copy()
    if needs_nearest.any() and len(nearest_source):
        source_rad = np.deg2rad(nearest_source[["_lat_key", "_lon_key"]].to_numpy(dtype=float))
        tree = BallTree(source_rad, metric="haversine")
        target_rad = np.deg2rad(df.loc[needs_nearest, ["_lat_key", "_lon_key"]].to_numpy(dtype=float))
        dist_rad, idx = tree.query(target_rad, k=1)
        dist_m = dist_rad[:, 0] * 6371000
        target_idx = df.index[needs_nearest]
        source_rows = nearest_source.iloc[idx[:, 0]].reset_index(drop=True)
        close = dist_m <= 30
        close_target_idx = target_idx[close]
        for col in ["후보_주변건물수", "후보_집중도", "후보_파일", "후보_컬럼", "후보_우선순위"]:
            merged.loc[close_target_idx, col] = source_rows.loc[close, col].to_numpy()
        merged.loc[close_target_idx, "후보_파일"] = (
            source_rows.loc[close, "후보_파일"].astype(str).to_numpy()
            + " (좌표근접 "
            + pd.Series(dist_m[close]).round(1).astype(str).to_numpy()
            + "m)"
        )

    no_candidate = merged["후보_주변건물수"].isna() | pd.to_numeric(
        merged["후보_주변건물수"], errors="coerce"
    ).le(0)
    direct_mask = suspect & no_candidate
    direct = direct_building_counts(df, direct_mask)
    direct_ok = direct["직접산출_주변건물수"].notna()
    if direct_ok.any():
        direct_idx = direct.index[direct_ok]
        merged.loc[direct_idx, "후보_주변건물수"] = direct.loc[direct_idx, "직접산출_주변건물수"].to_numpy()
        merged.loc[direct_idx, "후보_집중도"] = direct.loc[direct_idx, "직접산출_집중도"].to_numpy()
        merged.loc[direct_idx, "후보_파일"] = str(BUILDING_SHP.relative_to(BASE)) + " (직접 50m 재산출)"
        merged.loc[direct_idx, "후보_컬럼"] = "건물중심점_50m_count"
        merged.loc[direct_idx, "후보_우선순위"] = 120

    can_fill = suspect & merged["후보_주변건물수"].notna() & pd.to_numeric(merged["후보_주변건물수"], errors="coerce").gt(0)
    repaired = df.drop(columns=["_name_key", "_lat_key", "_lon_key"], errors="ignore").copy()
    repaired["주변건물수_원본"] = original
    repaired["집중도_원본"] = original_conc
    repaired["주변건물수_보정여부"] = np.where(can_fill, "보정", np.where(suspect, "검토필요", "원본유지"))
    repaired["주변건물수_보정출처"] = np.where(can_fill, merged["후보_파일"], "")
    repaired["주변건물수_보정컬럼"] = np.where(can_fill, merged["후보_컬럼"], "")
    repaired.loc[can_fill, "주변건물수"] = pd.to_numeric(merged.loc[can_fill, "후보_주변건물수"], errors="coerce")

    if "집중도" in repaired.columns:
        fill_conc = can_fill & pd.to_numeric(merged["후보_집중도"], errors="coerce").notna() & pd.to_numeric(merged["후보_집중도"], errors="coerce").gt(0)
        repaired.loc[fill_conc, "집중도"] = pd.to_numeric(merged.loc[fill_conc, "후보_집중도"], errors="coerce")

    remaining = repaired["주변건물수_보정여부"].eq("검토필요")
    reliable = repaired["주변건물수_보정여부"].isin(["원본유지", "보정"]) & pd.to_numeric(
        repaired["주변건물수"], errors="coerce"
    ).gt(0)
    if remaining.any() and reliable.any():
        global_count = float(pd.to_numeric(repaired.loc[reliable, "주변건물수"], errors="coerce").median())
        global_conc = float(pd.to_numeric(repaired.loc[reliable, "집중도"], errors="coerce").replace(0, np.nan).median())
        for idx in repaired.index[remaining]:
            row = repaired.loc[idx]
            count_value = np.nan
            conc_value = np.nan
            source_label = ""
            for keys, label in [
                (["구", "동", "업종"], "구+동+업종 중앙값"),
                (["구", "동"], "구+동 중앙값"),
                (["구", "업종"], "구+업종 중앙값"),
                (["업종"], "업종 중앙값"),
            ]:
                if not all(k in repaired.columns for k in keys):
                    continue
                subset = reliable.copy()
                for key in keys:
                    subset &= repaired[key].eq(row[key])
                if subset.any():
                    count_value = pd.to_numeric(repaired.loc[subset, "주변건물수"], errors="coerce").median()
                    conc_value = (
                        pd.to_numeric(repaired.loc[subset, "집중도"], errors="coerce")
                        .replace(0, np.nan)
                        .median()
                    )
                    source_label = label
                    break
            if pd.isna(count_value):
                count_value = global_count
                conc_value = global_conc
                source_label = "전체 중앙값"
            repaired.loc[idx, "주변건물수"] = round(float(count_value))
            if "집중도" in repaired.columns and not pd.isna(conc_value):
                repaired.loc[idx, "집중도"] = float(conc_value)
            repaired.loc[idx, "주변건물수_보정여부"] = "통계추정"
            repaired.loc[idx, "주변건물수_보정출처"] = source_label
            repaired.loc[idx, "주변건물수_보정컬럼"] = "주변건물수/집중도"

    if "집중도" in repaired.columns:
        conc = pd.to_numeric(repaired["집중도"], errors="coerce")
        conc_reliable = conc.gt(0)
        global_conc = float(conc[conc_reliable].median())
        zero_conc = conc.le(0) | conc.isna()
        for idx in repaired.index[zero_conc]:
            row = repaired.loc[idx]
            value = np.nan
            source_label = ""
            for keys, label in [
                (["구", "동", "업종"], "집중도 구+동+업종 중앙값"),
                (["구", "동"], "집중도 구+동 중앙값"),
                (["구", "업종"], "집중도 구+업종 중앙값"),
                (["업종"], "집중도 업종 중앙값"),
            ]:
                if not all(k in repaired.columns for k in keys):
                    continue
                subset = conc_reliable.copy()
                for key in keys:
                    subset &= repaired[key].eq(row[key])
                if subset.any():
                    value = conc[subset].median()
                    source_label = label
                    break
            if pd.isna(value):
                value = global_conc
                source_label = "집중도 전체 중앙값"
            repaired.loc[idx, "집중도"] = float(value)
            if repaired.loc[idx, "주변건물수_보정여부"] == "원본유지":
                repaired.loc[idx, "주변건물수_보정여부"] = "집중도추정"
                repaired.loc[idx, "주변건물수_보정출처"] = source_label
                repaired.loc[idx, "주변건물수_보정컬럼"] = "집중도"

    out_path = path.with_name(path.stem + "_주변건물수보정.csv")
    repaired.to_csv(out_path, index=False, encoding="utf-8-sig")
    return {
        "target": str(path.relative_to(BASE)),
        "output": str(out_path.relative_to(BASE)),
        "rows": int(len(repaired)),
        "suspect_rows": int(suspect.sum()),
        "filled_rows": int(can_fill.sum()),
        "estimated_rows": int((repaired["주변건물수_보정여부"] == "통계추정").sum()),
        "still_review_needed": int((repaired["주변건물수_보정여부"] == "검토필요").sum()),
    }


def main() -> None:
    inventory = inventory_sources()
    candidates = collect_candidates()
    if candidates.empty:
        raise RuntimeError("주변건물수 후보 파일을 찾지 못했습니다.")

    coord_best, coord_only_best, name_best = best_candidates(candidates)
    summaries = [repair_target(path, coord_best, coord_only_best, name_best) for path in TARGET_FILES if path.exists()]

    summary = {
        "inventory_files": int(len(inventory)),
        "candidate_rows": int(len(candidates)),
        "candidate_files": int(candidates["후보_파일"].nunique()),
        "repairs": summaries,
    }
    (OUT / "nearby_building_count_repair_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
