# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import geopandas as gpd
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
TARGET = BASE / "0424" / "data" / "분석변수_최종테이블0423.csv"
OUT_DIR = BASE / "0424" / "data"
CACHE_PATH = OUT_DIR / "tmap_geocode_cache.json"
API_URL = "https://apis.openapi.sk.com/tmap/geo/fullAddrGeo"

ADDRESS_SOURCES = [
    BASE / "data" / "data_with_fire_targets.csv",
    BASE / "data" / "서울10구_숙소_소방거리_유클리드.csv",
    BASE / "data" / "통합숙박시설_최종안0421.csv",
    BASE / "0424" / "data" / "data_with_fire_targets.csv",
    BASE / "0424" / "data" / "핵심서울0424.csv",
]


def norm(value: object) -> str:
    if pd.isna(value):
        return ""
    return "".join(str(value).split()).strip().lower()


def load_cache() -> dict:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def save_cache(cache: dict) -> None:
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def build_address_map() -> pd.DataFrame:
    frames = []
    for path in ADDRESS_SOURCES:
        if not path.exists():
            continue
        df = read_csv(path)
        name_col = "숙소명" if "숙소명" in df.columns else "업소명" if "업소명" in df.columns else None
        if not name_col or "주소" not in df.columns:
            continue
        cols = [c for c in ["구", "동", name_col, "주소"] if c in df.columns]
        tmp = df[cols].copy()
        tmp = tmp.rename(columns={name_col: "숙소명"})
        tmp["_name_key"] = tmp["숙소명"].map(norm)
        tmp["_gu_key"] = tmp["구"].map(norm) if "구" in tmp.columns else ""
        tmp["_dong_key"] = tmp["동"].map(norm) if "동" in tmp.columns else ""
        tmp["주소출처"] = str(path.relative_to(BASE))
        tmp = tmp[tmp["_name_key"].ne("") & tmp["주소"].notna()]
        frames.append(tmp)
    if not frames:
        return pd.DataFrame(columns=["_name_key", "_gu_key", "_dong_key", "주소", "주소출처"])
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(["_name_key", "_gu_key", "_dong_key"], keep="first")
    return merged


def parse_lon_lat(payload: dict) -> tuple[float | None, float | None, str]:
    text = json.dumps(payload, ensure_ascii=False)
    status = "not_found"
    coord_info = payload.get("coordinateInfo", {}) if isinstance(payload, dict) else {}
    coords = coord_info.get("coordinate", [])
    if isinstance(coords, dict):
        coords = [coords]
    for item in coords:
        if not isinstance(item, dict):
            continue
        lon = item.get("newLon") or item.get("lon") or item.get("longitude") or item.get("x")
        lat = item.get("newLat") or item.get("lat") or item.get("latitude") or item.get("y")
        if lon and lat:
            return float(lon), float(lat), "ok"
    # Fallback for small response schema changes.
    for lon_key, lat_key in [("lon", "lat"), ("longitude", "latitude"), ("x", "y")]:
        if lon_key in text and lat_key in text:
            break
    return None, None, status


def geocode(address: str, app_key: str, cache: dict, sleep_sec: float) -> dict:
    if address in cache:
        return cache[address]
    query = urlencode({"version": "1", "format": "json", "coordType": "WGS84GEO", "fullAddr": address})
    req = Request(f"{API_URL}?{query}", headers={"appKey": app_key})
    with urlopen(req, timeout=15) as response:
        payload = json.loads(response.read().decode("utf-8"))
    lon, lat, status = parse_lon_lat(payload)
    cache[address] = {"lon": lon, "lat": lat, "status": status, "raw": payload}
    time.sleep(sleep_sec)
    return cache[address]


def add_projected_xy(df: pd.DataFrame) -> pd.DataFrame:
    valid = df["보정_경도"].notna() & df["보정_위도"].notna()
    df["x_5181"] = pd.NA
    df["y_5181"] = pd.NA
    df["x_5179"] = pd.NA
    df["y_5179"] = pd.NA
    if valid.any():
        gdf = gpd.GeoDataFrame(
            df.loc[valid].copy(),
            geometry=gpd.points_from_xy(df.loc[valid, "보정_경도"], df.loc[valid, "보정_위도"]),
            crs="EPSG:4326",
        )
        p5181 = gdf.to_crs(epsg=5181)
        p5179 = gdf.to_crs(epsg=5179)
        df.loc[valid, "x_5181"] = p5181.geometry.x.round(2).to_numpy()
        df.loc[valid, "y_5181"] = p5181.geometry.y.round(2).to_numpy()
        df.loc[valid, "x_5179"] = p5179.geometry.x.round(2).to_numpy()
        df.loc[valid, "y_5179"] = p5179.geometry.y.round(2).to_numpy()
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="테스트용 호출 개수 제한")
    parser.add_argument("--sleep", type=float, default=0.12, help="API 호출 간 대기초")
    parser.add_argument("--key-env", default="TMAP_APP_KEY", help="SK OpenAPI appKey 환경변수명")
    args = parser.parse_args()

    target = read_csv(TARGET)
    target["_name_key"] = target["숙소명"].map(norm)
    target["_gu_key"] = target["구"].map(norm)
    target["_dong_key"] = target["동"].map(norm)

    addr_map = build_address_map()
    result = target.merge(
        addr_map[["_name_key", "_gu_key", "_dong_key", "주소", "주소출처"]],
        on=["_name_key", "_gu_key", "_dong_key"],
        how="left",
    )
    no_address = result["주소"].isna()
    if no_address.any():
        name_only = addr_map.drop_duplicates("_name_key", keep="first")
        fallback = target.loc[no_address, ["_name_key"]].merge(
            name_only[["_name_key", "주소", "주소출처"]], on="_name_key", how="left"
        )
        result.loc[no_address, "주소"] = fallback["주소"].to_numpy()
        result.loc[no_address, "주소출처"] = fallback["주소출처"].to_numpy()

    result.to_csv(OUT_DIR / "tmap_geocode_input_addresses.csv", index=False, encoding="utf-8-sig")

    app_key = os.getenv(args.key_env) or os.getenv("SK_OPENAPI_APP_KEY") or os.getenv("TMAP_API_KEY")
    if not app_key:
        summary = {
            "status": "need_app_key",
            "message": f"환경변수 {args.key_env} 또는 SK_OPENAPI_APP_KEY/TMAP_API_KEY에 appKey를 넣은 뒤 다시 실행하세요.",
            "address_matched": int(result["주소"].notna().sum()),
            "address_missing": int(result["주소"].isna().sum()),
            "input_file": str((OUT_DIR / "tmap_geocode_input_addresses.csv").relative_to(BASE)),
        }
        (OUT_DIR / "tmap_geocode_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    cache = load_cache()
    work = result[result["주소"].notna()].copy()
    if args.limit:
        work = work.head(args.limit)

    for idx, row in work.iterrows():
        info = geocode(str(row["주소"]), app_key, cache, args.sleep)
        result.loc[idx, "보정_경도"] = info.get("lon")
        result.loc[idx, "보정_위도"] = info.get("lat")
        result.loc[idx, "지오코딩상태"] = info.get("status")
        if (idx + 1) % 100 == 0:
            save_cache(cache)

    save_cache(cache)
    result["보정_경도"] = pd.to_numeric(result.get("보정_경도"), errors="coerce")
    result["보정_위도"] = pd.to_numeric(result.get("보정_위도"), errors="coerce")
    result = add_projected_xy(result)
    result = result.drop(columns=["_name_key", "_gu_key", "_dong_key"], errors="ignore")
    out_path = OUT_DIR / "분석변수_최종테이블0423_TMAP좌표보정.csv"
    result.to_csv(out_path, index=False, encoding="utf-8-sig")

    summary = {
        "status": "done",
        "rows": int(len(result)),
        "address_matched": int(result["주소"].notna().sum()),
        "geocoded": int(result["보정_위도"].notna().sum()),
        "output": str(out_path.relative_to(BASE)),
    }
    (OUT_DIR / "tmap_geocode_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
