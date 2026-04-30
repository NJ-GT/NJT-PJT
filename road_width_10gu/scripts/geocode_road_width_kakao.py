from __future__ import annotations

import csv
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests


BASE_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = BASE_DIR.parent
INPUT_PATH = BASE_DIR / "data" / "seoul_road_width_viRoutDt_한글컬럼.csv"
OUTPUT_PATH = BASE_DIR / "data" / "seoul_road_width_viRoutDt_한글컬럼_대표좌표.csv"
CACHE_PATH = BASE_DIR / "data" / "seoul_road_width_viRoutDt_도로명좌표_cache.csv"
KAKAO_ADDRESS_URL = "https://dapi.kakao.com/v2/local/search/address.json"
REQUEST_DELAY = float(os.getenv("KAKAO_REQUEST_DELAY", "0.03"))


def load_kakao_key() -> str:
    key = os.getenv("KAKAO_REST_API_KEY") or os.getenv("KAKAO_KEY")
    if key:
        return key.strip()

    # Reuse the key already present in the local project without duplicating it.
    candidates = [
        PROJECT_DIR / "scripts" / "geocode_firestation.py",
        PROJECT_DIR / "scripts" / "fill_from_api.py",
        PROJECT_DIR / "scripts" / "process_final_hospitality_0415.py",
    ]
    for path in candidates:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        match = re.search(r"KAKAO_KEY\s*=\s*['\"]([^'\"]+)['\"]", text)
        if match:
            return match.group(1).strip()

    raise RuntimeError("Kakao REST API key not found. Set KAKAO_REST_API_KEY first.")


def load_cache() -> dict[str, dict[str, str]]:
    if not CACHE_PATH.exists():
        return {}
    with CACHE_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        return {row["도로명"]: row for row in csv.DictReader(f)}


def save_cache(cache: dict[str, dict[str, str]]) -> None:
    fieldnames = [
        "도로명",
        "위도",
        "경도",
        "좌표조회주소",
        "좌표후보수",
        "좌표조회결과",
        "좌표오류",
    ]
    rows = [cache[key] for key in sorted(cache)]
    with CACHE_PATH.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def choose_document(road_name: str, docs: list[dict]) -> dict | None:
    seoul_docs = [
        doc
        for doc in docs
        if str(doc.get("address_name", "")).startswith(("서울 ", "서울특별시 "))
    ]
    candidates = seoul_docs or docs
    if not candidates:
        return None

    exact = [
        doc
        for doc in candidates
        if str(doc.get("address_name", "")).strip().endswith(f" {road_name}")
    ]
    return (exact or candidates)[0]


def geocode_road(session: requests.Session, road_name: str) -> dict[str, str]:
    query = f"서울특별시 {road_name}"
    try:
        for attempt in range(3):
            resp = session.get(
                KAKAO_ADDRESS_URL,
                params={"query": query, "size": 10},
                timeout=10,
            )
            if resp.status_code in {429, 500, 502, 503, 504} and attempt < 2:
                time.sleep(1.0 + attempt)
                continue
            break
        resp.raise_for_status()
        docs = resp.json().get("documents", [])
        doc = choose_document(road_name, docs)
        if not doc:
            return {
                "도로명": road_name,
                "위도": "",
                "경도": "",
                "좌표조회주소": "",
                "좌표후보수": str(len(docs)),
                "좌표조회결과": "미조회",
                "좌표오류": "",
            }
        return {
            "도로명": road_name,
            "위도": str(doc.get("y", "")),
            "경도": str(doc.get("x", "")),
            "좌표조회주소": str(doc.get("address_name", "")),
            "좌표후보수": str(len(docs)),
            "좌표조회결과": "성공",
            "좌표오류": "",
        }
    except Exception as exc:
        return {
            "도로명": road_name,
            "위도": "",
            "경도": "",
            "좌표조회주소": "",
            "좌표후보수": "",
            "좌표조회결과": "오류",
            "좌표오류": str(exc)[:200],
        }


def main() -> int:
    if not INPUT_PATH.exists():
        print(f"Input not found: {INPUT_PATH}", file=sys.stderr)
        return 1

    key = load_kakao_key()
    df = pd.read_csv(INPUT_PATH, dtype=str, encoding="utf-8-sig").fillna("")
    roads = list(dict.fromkeys(df["도로명"].astype(str).str.strip()))
    cache = load_cache()
    missing_roads = [road for road in roads if road and road not in cache]

    print(f"Input rows: {len(df)}", flush=True)
    print(f"Unique roads: {len(roads)}", flush=True)
    print(f"Cached roads: {len(cache)}", flush=True)
    print(f"Roads to geocode: {len(missing_roads)}", flush=True)

    session = requests.Session()
    session.headers.update({"Authorization": f"KakaoAK {key}"})

    for idx, road in enumerate(missing_roads, start=1):
        cache[road] = geocode_road(session, road)
        if idx % 100 == 0 or idx == len(missing_roads):
            save_cache(cache)
            success = sum(1 for row in cache.values() if row.get("좌표조회결과") == "성공")
            print(f"Geocoded {idx}/{len(missing_roads)} new roads, success cache={success}", flush=True)
        time.sleep(REQUEST_DELAY)

    save_cache(cache)
    cache_df = pd.DataFrame(cache.values())
    out = df.merge(cache_df, on="도로명", how="left")
    cols = [
        "순번",
        "도로명",
        "도로구분",
        "도로기능",
        "도로규모",
        "도로폭",
        "위도",
        "경도",
        "좌표조회주소",
        "좌표후보수",
        "좌표조회결과",
        "작업일시",
    ]
    out = out[[col for col in cols if col in out.columns]]
    out.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    success_rows = (out["좌표조회결과"] == "성공").sum()
    print(f"Saved rows: {len(out)}", flush=True)
    print(f"Rows with coordinates: {success_rows}", flush=True)
    print(f"Output: {OUTPUT_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
