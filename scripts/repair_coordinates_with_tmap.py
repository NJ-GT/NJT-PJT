# -*- coding: utf-8 -*-
"""
TMAP fullAddrGeo 지오코딩 API 로 분석변수_최종테이블0423 의 좌표를 보정·재계산하는 스크립트.

목적:
    - 0423 분석 테이블에 들어 있지 않은 주소 정보를 다른 csv 들에서 보강
    - TMAP API 로 주소 → WGS84 좌표 변환 후 보정_경도/보정_위도 부착
    - 보정 좌표를 EPSG:5181 / EPSG:5179 평면좌표로 추가 변환
    - 캐시(json) 와 결과 csv, 요약 json 저장 (API 키가 없으면 입력 csv 만 만들고 종료)

CLI 옵션:
    --limit N         : 테스트용으로 호출 N건 제한
    --sleep S         : API 호출 간 대기(초), 기본 0.12
    --key-env NAME    : appKey 가 들어있는 환경변수명 (기본 TMAP_APP_KEY)

처리 흐름:
    1) 0423 테이블 + 주소 소스 csv 들 로드, 숙소명/구/동 키로 주소 병합
    2) 환경변수에서 TMAP appKey 확인 (없으면 입력 csv만 만들고 안내)
    3) 주소가 채워진 행만 골라 TMAP API 로 지오코딩 (캐시 사용)
    4) 보정 좌표 → 5181/5179 평면 좌표 변환
    5) 결과 csv + 요약 json 저장
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
# URL 인코딩 / HTTP 요청 (외부 라이브러리 의존도 최소화)
from urllib.parse import urlencode
from urllib.request import Request, urlopen

# GeoDataFrame 으로 좌표계 변환
import geopandas as gpd
import pandas as pd


# 경로 상수
BASE = Path(__file__).resolve().parents[1]
TARGET = BASE / "0424" / "data" / "분석변수_최종테이블0423.csv"
OUT_DIR = BASE / "0424" / "data"
CACHE_PATH = OUT_DIR / "tmap_geocode_cache.json"
# TMAP fullAddrGeo: 전체 주소를 받아 좌표를 돌려주는 API
API_URL = "https://apis.openapi.sk.com/tmap/geo/fullAddrGeo"

# 주소가 들어있을 가능성이 있는 csv 후보들 (있는 것만 사용)
ADDRESS_SOURCES = [
    BASE / "data" / "data_with_fire_targets.csv",
    BASE / "data" / "서울10구_숙소_소방거리_유클리드.csv",
    BASE / "data" / "통합숙박시설_최종안0421.csv",
    BASE / "0424" / "data" / "data_with_fire_targets.csv",
    BASE / "0424" / "data" / "핵심서울0424.csv",
]


def norm(value: object) -> str:
    """공백 제거 + 소문자화 → 매칭 키 정규화."""
    if pd.isna(value):
        return ""
    return "".join(str(value).split()).strip().lower()


def load_cache() -> dict:
    """이전 호출 결과 캐시 로드 (없으면 빈 dict)."""
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def save_cache(cache: dict) -> None:
    """캐시 dict 를 JSON 으로 저장."""
    CACHE_PATH.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def build_address_map() -> pd.DataFrame:
    """후보 csv 들에서 (숙소명, 구, 동) → 주소 매핑 테이블 구성."""
    frames = []
    for path in ADDRESS_SOURCES:
        if not path.exists():
            continue
        df = read_csv(path)
        # csv 마다 컬럼명이 '숙소명' or '업소명' 으로 다를 수 있음
        name_col = (
            "숙소명"
            if "숙소명" in df.columns
            else "업소명"
            if "업소명" in df.columns
            else None
        )
        if not name_col or "주소" not in df.columns:
            continue
        cols = [c for c in ["구", "동", name_col, "주소"] if c in df.columns]
        tmp = df[cols].copy()
        tmp = tmp.rename(columns={name_col: "숙소명"})
        # 매칭 키 (정규화된 값)
        tmp["_name_key"] = tmp["숙소명"].map(norm)
        tmp["_gu_key"] = tmp["구"].map(norm) if "구" in tmp.columns else ""
        tmp["_dong_key"] = tmp["동"].map(norm) if "동" in tmp.columns else ""
        # 출처 csv 경로(상대) 보존 → 추적성
        tmp["주소출처"] = str(path.relative_to(BASE))
        # 빈 키 제거
        tmp = tmp[tmp["_name_key"].ne("") & tmp["주소"].notna()]
        frames.append(tmp)
    if not frames:
        return pd.DataFrame(
            columns=["_name_key", "_gu_key", "_dong_key", "주소", "주소출처"]
        )
    merged = pd.concat(frames, ignore_index=True)
    # 동일 키는 첫 출처만 유지
    merged = merged.drop_duplicates(["_name_key", "_gu_key", "_dong_key"], keep="first")
    return merged


def parse_lon_lat(payload: dict) -> tuple[float | None, float | None, str]:
    """TMAP 응답 JSON 에서 (lon, lat, status) 추출."""
    text = json.dumps(payload, ensure_ascii=False)
    status = "not_found"
    coord_info = payload.get("coordinateInfo", {}) if isinstance(payload, dict) else {}
    coords = coord_info.get("coordinate", [])
    # 리스트/딕셔너리 모두 허용
    if isinstance(coords, dict):
        coords = [coords]
    # 우선순위: newLon/newLat → lon/lat → longitude/latitude → x/y
    for item in coords:
        if not isinstance(item, dict):
            continue
        lon = (
            item.get("newLon")
            or item.get("lon")
            or item.get("longitude")
            or item.get("x")
        )
        lat = (
            item.get("newLat")
            or item.get("lat")
            or item.get("latitude")
            or item.get("y")
        )
        if lon and lat:
            return float(lon), float(lat), "ok"
    # 응답 스키마가 살짝 다를 때를 위한 폴백 (no-op 루프)
    for lon_key, lat_key in [("lon", "lat"), ("longitude", "latitude"), ("x", "y")]:
        if lon_key in text and lat_key in text:
            break
    return None, None, status


def geocode(address: str, app_key: str, cache: dict, sleep_sec: float) -> dict:
    """주소 1건 지오코딩. 캐시 hit 시 즉시 반환, 미스는 API 호출 후 캐시에 저장."""
    if address in cache:
        return cache[address]
    # 쿼리 파라미터 구성 (WGS84)
    query = urlencode(
        {"version": "1", "format": "json", "coordType": "WGS84GEO", "fullAddr": address}
    )
    req = Request(f"{API_URL}?{query}", headers={"appKey": app_key})
    with urlopen(req, timeout=15) as response:
        payload = json.loads(response.read().decode("utf-8"))
    lon, lat, status = parse_lon_lat(payload)
    cache[address] = {"lon": lon, "lat": lat, "status": status, "raw": payload}
    # API rate limit 보호용 sleep
    time.sleep(sleep_sec)
    return cache[address]


def add_projected_xy(df: pd.DataFrame) -> pd.DataFrame:
    """보정 위경도를 EPSG:5181 / 5179 평면 좌표로 추가 변환."""
    valid = df["보정_경도"].notna() & df["보정_위도"].notna()
    # 결과 컬럼 초기화
    df["x_5181"] = pd.NA
    df["y_5181"] = pd.NA
    df["x_5179"] = pd.NA
    df["y_5179"] = pd.NA
    if valid.any():
        # 유효 좌표만 GeoDataFrame 화 → CRS 변환
        gdf = gpd.GeoDataFrame(
            df.loc[valid].copy(),
            geometry=gpd.points_from_xy(
                df.loc[valid, "보정_경도"], df.loc[valid, "보정_위도"]
            ),
            crs="EPSG:4326",
        )
        p5181 = gdf.to_crs(epsg=5181)
        p5179 = gdf.to_crs(epsg=5179)
        # 소수 둘째자리까지 round 후 원본 df 의 해당 행에 채워 넣기
        df.loc[valid, "x_5181"] = p5181.geometry.x.round(2).to_numpy()
        df.loc[valid, "y_5181"] = p5181.geometry.y.round(2).to_numpy()
        df.loc[valid, "x_5179"] = p5179.geometry.x.round(2).to_numpy()
        df.loc[valid, "y_5179"] = p5179.geometry.y.round(2).to_numpy()
    return df


def main() -> None:
    """CLI 엔트리포인트."""
    # 명령줄 인수 파싱
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit", type=int, default=None, help="테스트용 호출 개수 제한"
    )
    parser.add_argument("--sleep", type=float, default=0.12, help="API 호출 간 대기초")
    parser.add_argument(
        "--key-env", default="TMAP_APP_KEY", help="SK OpenAPI appKey 환경변수명"
    )
    args = parser.parse_args()

    # 0423 분석 테이블 로드 + 매칭 키 부착
    target = read_csv(TARGET)
    target["_name_key"] = target["숙소명"].map(norm)
    target["_gu_key"] = target["구"].map(norm)
    target["_dong_key"] = target["동"].map(norm)

    # 주소 맵 구성 후 좌측 조인 (구·동·숙소명 3-키)
    addr_map = build_address_map()
    result = target.merge(
        addr_map[["_name_key", "_gu_key", "_dong_key", "주소", "주소출처"]],
        on=["_name_key", "_gu_key", "_dong_key"],
        how="left",
    )
    # 미매칭 행은 숙소명만으로 폴백 매칭
    no_address = result["주소"].isna()
    if no_address.any():
        name_only = addr_map.drop_duplicates("_name_key", keep="first")
        fallback = target.loc[no_address, ["_name_key"]].merge(
            name_only[["_name_key", "주소", "주소출처"]], on="_name_key", how="left"
        )
        result.loc[no_address, "주소"] = fallback["주소"].to_numpy()
        result.loc[no_address, "주소출처"] = fallback["주소출처"].to_numpy()

    # 입력(주소가 채워진 상태) csv 저장 — appKey 가 없을 때 그대로 활용 가능
    result.to_csv(
        OUT_DIR / "tmap_geocode_input_addresses.csv", index=False, encoding="utf-8-sig"
    )

    # 환경변수에서 appKey 탐색 (여러 이름 허용)
    app_key = (
        os.getenv(args.key_env)
        or os.getenv("SK_OPENAPI_APP_KEY")
        or os.getenv("TMAP_API_KEY")
    )
    if not app_key:
        # appKey 가 없으면 안내만 남기고 종료
        summary = {
            "status": "need_app_key",
            "message": f"환경변수 {args.key_env} 또는 SK_OPENAPI_APP_KEY/TMAP_API_KEY에 appKey를 넣은 뒤 다시 실행하세요.",
            "address_matched": int(result["주소"].notna().sum()),
            "address_missing": int(result["주소"].isna().sum()),
            "input_file": str(
                (OUT_DIR / "tmap_geocode_input_addresses.csv").relative_to(BASE)
            ),
        }
        (OUT_DIR / "tmap_geocode_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    # 캐시 로드 후 주소가 있는 행만 호출
    cache = load_cache()
    work = result[result["주소"].notna()].copy()
    if args.limit:
        work = work.head(args.limit)

    # 한 행씩 지오코딩 (100건마다 캐시 저장 → 중간 실패 대비)
    for idx, row in work.iterrows():
        info = geocode(str(row["주소"]), app_key, cache, args.sleep)
        result.loc[idx, "보정_경도"] = info.get("lon")
        result.loc[idx, "보정_위도"] = info.get("lat")
        result.loc[idx, "지오코딩상태"] = info.get("status")
        if (idx + 1) % 100 == 0:
            save_cache(cache)

    # 마지막 캐시 + 좌표 변환 + 결과 저장
    save_cache(cache)
    result["보정_경도"] = pd.to_numeric(result.get("보정_경도"), errors="coerce")
    result["보정_위도"] = pd.to_numeric(result.get("보정_위도"), errors="coerce")
    result = add_projected_xy(result)
    # 임시 키 컬럼 제거
    result = result.drop(columns=["_name_key", "_gu_key", "_dong_key"], errors="ignore")
    out_path = OUT_DIR / "분석변수_최종테이블0423_TMAP좌표보정.csv"
    result.to_csv(out_path, index=False, encoding="utf-8-sig")

    # 요약 json 저장
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
