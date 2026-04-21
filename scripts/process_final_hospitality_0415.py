# -*- coding: utf-8 -*-
"""
[파일 설명]
통합숙박시설최종안0415.csv의 누락값을 외부 API와 로컬 CSV 여러 소스를 통해
자동으로 채워 최종 정비하는 스크립트.

주요 역할:
  1. 건축물대장 API로 사용승인일 채우기 → 사용승인일 없는 행 삭제
  2. 로컬 CSV(통합표제부·인허가 원본)에서 도로명주소·좌표 채우기
  3. 카카오 지오코딩 API로 남은 도로명주소·좌표 채우기
  4. 처리 결과를 JSON 보고서로 저장

입력: data/통합숙박시설최종안0415.csv
      원본데이터/통합숙박시설표제부0414.csv (로컬 후보 소스)
      원본데이터/서울시 통합 숙박시설 0414.csv 등 4개 인허가 원본
출력: data/통합숙박시설최종안0415.csv (덮어쓰기)
      reports/통합숙박시설최종안0415_정비보고서_20260415.json
"""
from __future__ import annotations

import csv
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import requests
from pyproj import Transformer


BASE_DIR = Path(__file__).resolve().parent.parent
TARGET_PATH = BASE_DIR / "data" / "통합숙박시설최종안0415.csv"
SOURCE_DIR = BASE_DIR / "원본데이터"
INTEGRATED_PATH = SOURCE_DIR / "통합숙박시설표제부0414.csv"
RAW_PATHS = [
    SOURCE_DIR / "서울시 통합 숙박시설 0414.csv",
    SOURCE_DIR / "서울시 관광숙박업 인허가 정보.csv",
    SOURCE_DIR / "서울시 숙박업 인허가 정보.csv",
    SOURCE_DIR / "서울시 외국인관광도시민박업 인허가 정보.csv",
]
REPORT_PATH = BASE_DIR / "reports" / "통합숙박시설최종안0415_정비보고서_20260415.json"

BLDG_KEY = "1c1ea0b782ec251d390c4d34426e6ac87281041591d929dec42b641d51098eff"
BLDG_URL = "https://apis.data.go.kr/1613000/BldRgstHubService/getBrTitleInfo"
KAKAO_KEY = "96172db4c3b086f76853ed89242acefa"
KAKAO_ADDRESS_URL = "https://dapi.kakao.com/v2/local/search/address.json"
TRANSFORMER = Transformer.from_crs("EPSG:4326", "EPSG:5174", always_xy=True)


def norm(value: object) -> str:
    """None이면 빈 문자열, 아니면 문자열로 변환 후 앞뒤 공백 제거."""
    return "" if value is None else str(value).strip()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """UTF-8-BOM CSV를 읽어 (헤더 목록, 행 목록) 튜플 반환."""
    with path.open("r", encoding="utf-8-sig", newline="") as fp:
        reader = csv.DictReader(fp)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    """UTF-8-BOM CSV로 저장."""
    with path.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def valid_date(value: object) -> bool:
    """8자리 숫자 날짜(YYYYMMDD)이고 00000000이나 앞자리 0으로 시작하지 않으면 유효."""
    text = norm(value)
    if text.endswith(".0"):
        text = text[:-2]
    return len(text) == 8 and text.isdigit() and text != "00000000" and not text.startswith("0")


def normalize_code(value: str, width: int) -> str:
    """시군구코드·법정동코드를 width 자리 0패딩 문자열로 정규화."""
    text = norm(value)
    if text.endswith(".0"):
        text = text[:-2]
    return text.zfill(width) if text else "0" * width


def normalize_number(value: str) -> str:
    """본번·부번을 4자리 0패딩 문자열로 정규화 (예: '3' → '0003')."""
    text = norm(value)
    if not text:
        return "0000"
    if text.endswith(".0"):
        text = text[:-2]
    try:
        return str(int(float(text))).zfill(4)
    except ValueError:
        return "0000"


def clean_road_address(value: object) -> str:
    """도로명주소에서 쉼표 이후 부가정보 제거, 괄호 안 동명 첫 번째 항목만 유지."""
    text = norm(value)
    if not text:
        return ""
    matches = re.findall(r"\(([^)]*)\)", text)
    first_paren = ""
    if matches:
        first = matches[-1].split(",")[0].strip()
        if first:
            first_paren = f" ({first})"
    base = text.split(",")[0].strip()
    base = re.sub(r"\s+", " ", base)
    if first_paren and "(" not in base:
        return f"{base}{first_paren}"
    return base


def fetch_bldg_items(sigungu_cd: str, bjdong_cd: str, bun: str, ji: str, session: requests.Session) -> list[dict]:
    """건축물대장 API에서 해당 필지의 건물 표제부 목록을 반환한다."""
    url = (
        f"{BLDG_URL}?serviceKey={BLDG_KEY}"
        f"&sigunguCd={normalize_code(sigungu_cd, 5)}"
        f"&bjdongCd={normalize_code(bjdong_cd, 5)}"
        f"&bun={normalize_number(bun)}"
        f"&ji={normalize_number(ji)}"
        f"&numOfRows=20&pageNo=1&_type=json"
    )
    resp = session.get(url, timeout=20)
    resp.raise_for_status()
    body = resp.json().get("response", {}).get("body", {})
    items = body.get("items", {})
    if not items:
        return []
    data = items.get("item", [])
    if isinstance(data, dict):
        return [data]  # 단일 항목이면 리스트로 감쌈
    return data


def choose_use_approval_date(row: dict[str, str], items: list[dict]) -> tuple[str, str]:
    """API 응답 items에서 가장 신뢰할 수 있는 사용승인일을 선택한다.

    우선순위:
      1. 관리건축물대장PK 정확 일치
      2. 유효한 날짜가 한 개뿐일 때
      3. 주건축물 레코드에서 날짜가 한 개뿐일 때
    """
    pk = norm(row.get("관리건축물대장PK"))

    for item in items:
        if norm(item.get("mgmBldrgstPk")) == pk and valid_date(item.get("useAprDay")):
            return norm(item.get("useAprDay")), "pk_exact"

    valid_items = [item for item in items if valid_date(item.get("useAprDay"))]
    unique_dates = sorted({norm(item.get("useAprDay")) for item in valid_items})
    if len(unique_dates) == 1:
        return unique_dates[0], "single_unique_date"

    main_items = [item for item in valid_items if norm(item.get("mainAtchGbCdNm")) == "주건축물"]
    main_dates = sorted({norm(item.get("useAprDay")) for item in main_items})
    if len(main_dates) == 1:
        return main_dates[0], "main_building_unique_date"

    return "", "unresolved"


def build_local_candidates() -> tuple[dict[str, list[dict]], dict[str, list[dict]]]:
    """통합표제부와 인허가 원본 CSV에서 PK별·사업장명별 좌표·도로명 후보를 수집한다."""
    by_pk: defaultdict[str, list[dict]] = defaultdict(list)
    by_name: defaultdict[str, list[dict]] = defaultdict(list)

    _, integrated_rows = read_csv(INTEGRATED_PATH)
    for row in integrated_rows:
        rec = {
            "source": "integrated",
            "pk": norm(row.get("selected_registry_pk")),
            "name": norm(row.get("사업장명")),
            "road": norm(row.get("registry_도로명대지위치")),
            "x": norm(row.get("좌표정보(X)")),
            "y": norm(row.get("좌표정보(Y)")),
        }
        if rec["pk"]:
            by_pk[rec["pk"]].append(rec)
        if rec["name"]:
            by_name[rec["name"]].append(rec)

    for raw_path in RAW_PATHS:
        _, rows = read_csv(raw_path)
        for row in rows:
            rec = {
                "source": raw_path.name,
                "pk": "",
                "name": norm(row.get("사업장명")),
                "road": clean_road_address(row.get("도로명주소") or row.get("도로명대지위치")),
                "x": norm(row.get("좌표정보(X)") or row.get("X좌표")),
                "y": norm(row.get("좌표정보(Y)") or row.get("Y좌표")),
            }
            if rec["name"]:
                by_name[rec["name"]].append(rec)

    return dict(by_pk), dict(by_name)


def choose_local_road(row: dict[str, str], by_pk: dict[str, list[dict]], by_name: dict[str, list[dict]]) -> tuple[str, str]:
    pk = norm(row.get("관리건축물대장PK"))
    name = norm(row.get("사업장명"))

    pk_roads = sorted({cand["road"] for cand in by_pk.get(pk, []) if cand["road"]})
    if len(pk_roads) == 1:
        return pk_roads[0], "pk_local"

    fire_road = clean_road_address(row.get("소방청_도로명주소_매칭"))
    if fire_road:
        return fire_road, "fire_match"

    name_roads = sorted({cand["road"] for cand in by_name.get(name, []) if cand["road"]})
    if len(name_roads) == 1:
        return name_roads[0], "name_local"

    return "", "unresolved"


def choose_local_xy(row: dict[str, str], by_pk: dict[str, list[dict]], by_name: dict[str, list[dict]]) -> tuple[str, str, str]:
    pk = norm(row.get("관리건축물대장PK"))
    name = norm(row.get("사업장명"))

    def pairs(cands: list[dict]) -> list[tuple[str, str]]:
        return sorted({(cand["x"], cand["y"]) for cand in cands if cand["x"] and cand["y"]})

    pk_pairs = pairs(by_pk.get(pk, []))
    if len(pk_pairs) == 1:
        return pk_pairs[0][0], pk_pairs[0][1], "pk_local"

    name_pairs = pairs(by_name.get(name, []))
    if len(name_pairs) == 1:
        return name_pairs[0][0], name_pairs[0][1], "name_local"

    return "", "", "unresolved"


def kakao_address_doc(query: str, session: requests.Session, cache: dict[str, dict]) -> dict:
    query = norm(query)
    if not query:
        return {}
    if query in cache:
        return cache[query]

    try:
        resp = session.get(KAKAO_ADDRESS_URL, params={"query": query, "size": 1}, timeout=10)
        resp.raise_for_status()
        docs = resp.json().get("documents", [])
        cache[query] = docs[0] if docs else {}
        return cache[query]
    except Exception:
        cache[query] = {}
        return {}


def kakao_road_from_query(query: str, session: requests.Session, cache: dict[str, dict]) -> str:
    doc = kakao_address_doc(query, session, cache)
    road = (doc.get("road_address") or {}).get("address_name") or ""
    return norm(road)


def kakao_xy_from_query(query: str, session: requests.Session, cache: dict[str, dict]) -> tuple[str, str]:
    doc = kakao_address_doc(query, session, cache)
    if not doc:
        return "", ""
    try:
        lon = float(doc["x"])
        lat = float(doc["y"])
        x, y = TRANSFORMER.transform(lon, lat)
        return f"{x:.6f}".rstrip("0").rstrip("."), f"{y:.6f}".rstrip("0").rstrip(".")
    except Exception:
        return "", ""


def main() -> None:
    fieldnames, rows = read_csv(TARGET_PATH)
    report: dict[str, object] = {}

    before_counts = {
        "rows": len(rows),
        "blank_useAprDay": sum(1 for row in rows if not norm(row.get("사용승인일"))),
        "blank_x": sum(1 for row in rows if not norm(row.get("X좌표"))),
        "blank_y": sum(1 for row in rows if not norm(row.get("Y좌표"))),
        "blank_road": sum(1 for row in rows if not norm(row.get("도로명대지위치"))),
    }
    report["before"] = before_counts

    bldg_session = requests.Session()
    query_cache: dict[tuple[str, str, str, str], list[dict]] = {}
    query_failures: list[dict] = []
    date_stats = Counter()

    blank_rows = [row for row in rows if not norm(row.get("사용승인일"))]
    query_keys = sorted(
        {
            (
                norm(row.get("시군구코드")),
                norm(row.get("법정동코드")),
                norm(row.get("번")),
                norm(row.get("지")),
            )
            for row in blank_rows
        }
    )

    for idx, key in enumerate(query_keys, start=1):
        try:
            query_cache[key] = fetch_bldg_items(*key, session=bldg_session)
        except Exception as exc:
            query_cache[key] = []
            query_failures.append({"key": key, "error": str(exc)})
        if idx % 25 == 0 or idx == len(query_keys):
            print(f"사용승인일 API 조회: {idx}/{len(query_keys)}")
        time.sleep(0.03)

    for row in rows:
        if norm(row.get("사용승인일")):
            continue
        key = (
            norm(row.get("시군구코드")),
            norm(row.get("법정동코드")),
            norm(row.get("번")),
            norm(row.get("지")),
        )
        selected, strategy = choose_use_approval_date(row, query_cache.get(key, []))
        date_stats[strategy] += 1
        if selected:
            row["사용승인일"] = selected

    remaining_blank_date = [row for row in rows if not norm(row.get("사용승인일"))]
    deleted_count = len(remaining_blank_date)
    rows = [row for row in rows if norm(row.get("사용승인일"))]

    by_pk, by_name = build_local_candidates()
    kakao_session = requests.Session()
    kakao_session.headers.update({"Authorization": f"KakaoAK {KAKAO_KEY}"})
    kakao_cache: dict[str, dict] = {}
    fill_stats = Counter()

    for row in rows:
        if not norm(row.get("도로명대지위치")):
            road, strategy = choose_local_road(row, by_pk, by_name)
            if not road:
                road = kakao_road_from_query(norm(row.get("대지위치")) or norm(row.get("소방청_지번주소_매칭")), kakao_session, kakao_cache)
                strategy = "kakao_jibun" if road else "unresolved"
            if road:
                row["도로명대지위치"] = road
            fill_stats[f"road_{strategy}"] += 1

        if not norm(row.get("X좌표")) or not norm(row.get("Y좌표")):
            x, y, strategy = choose_local_xy(row, by_pk, by_name)
            if not (x and y):
                query = (
                    norm(row.get("도로명대지위치"))
                    or clean_road_address(row.get("소방청_도로명주소_매칭"))
                    or norm(row.get("대지위치"))
                    or norm(row.get("소방청_지번주소_매칭"))
                )
                x, y = kakao_xy_from_query(query, kakao_session, kakao_cache)
                strategy = "kakao_address" if (x and y) else "unresolved"
            if x and y:
                row["X좌표"] = x
                row["Y좌표"] = y
            fill_stats[f"xy_{strategy}"] += 1

    write_csv(TARGET_PATH, fieldnames, rows)

    after_counts = {
        "rows": len(rows),
        "blank_useAprDay": sum(1 for row in rows if not norm(row.get("사용승인일"))),
        "blank_x": sum(1 for row in rows if not norm(row.get("X좌표"))),
        "blank_y": sum(1 for row in rows if not norm(row.get("Y좌표"))),
        "blank_road": sum(1 for row in rows if not norm(row.get("도로명대지위치"))),
    }

    report["use_approval_fill"] = {
        "query_key_count": len(query_keys),
        "strategy_counts": dict(date_stats),
        "api_failures": query_failures[:20],
    }
    report["deleted_rows_after_use_approval_fill"] = deleted_count
    report["xy_road_fill"] = dict(fill_stats)
    report["after"] = after_counts
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8-sig")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
