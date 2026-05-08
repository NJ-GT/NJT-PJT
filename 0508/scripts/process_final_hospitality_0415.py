# -*- coding: utf-8 -*-
"""
통합숙박시설 최종안 0415 — 결측 정비 통합 파이프라인.

목적:
    - data/통합숙박시설최종안0415.csv 의 사용승인일/도로명/좌표 결측을 채워 데이터 품질 향상.
    - 외부 API 1회 호출량을 줄이려 로컬 CSV(통합표제부/인허가 4종) 후보 매칭을 우선 시도하고,
      로컬에서 채울 수 없는 행만 카카오 지오코딩 API 로 보완.

처리 단계:
    1) 건축물대장 API 로 사용승인일 채우기 → 끝까지 못 채운 행 삭제
    2) 통합표제부+인허가 4개 원본에서 PK/사업장명 기반 도로명·좌표 후보 수집
    3) 카카오 지오코딩 API 로 잔존 결측 보완 (캐시 사용)
    4) 통계/실패 키 등을 JSON 보고서로 저장

입력:
    - data/통합숙박시설최종안0415.csv                  (정비 대상)
    - 원본데이터/통합숙박시설표제부0414.csv             (PK 기반 후보)
    - 원본데이터/서울시 통합 숙박시설 0414.csv 등 4종   (사업장명 기반 후보)

출력:
    - data/통합숙박시설최종안0415.csv                              (덮어쓰기)
    - reports/통합숙박시설최종안0415_정비보고서_20260415.json
"""

from __future__ import annotations

import csv
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import requests  # 건축물대장/카카오 API 호출
from pyproj import Transformer  # 카카오 결과(WGS84) → EPSG:5174 변환


# 경로
BASE_DIR = Path(__file__).resolve().parent.parent
TARGET_PATH = BASE_DIR / "data" / "통합숙박시설최종안0415.csv"
SOURCE_DIR = BASE_DIR / "원본데이터"
INTEGRATED_PATH = SOURCE_DIR / "통합숙박시설표제부0414.csv"
# 사업장명/주소 기반 추가 후보용 인허가 4종
RAW_PATHS = [
    SOURCE_DIR / "서울시 통합 숙박시설 0414.csv",
    SOURCE_DIR / "서울시 관광숙박업 인허가 정보.csv",
    SOURCE_DIR / "서울시 숙박업 인허가 정보.csv",
    SOURCE_DIR / "서울시 외국인관광도시민박업 인허가 정보.csv",
]
REPORT_PATH = BASE_DIR / "reports" / "통합숙박시설최종안0415_정비보고서_20260415.json"

# 외부 API 키와 엔드포인트
BLDG_KEY = "1c1ea0b782ec251d390c4d34426e6ac87281041591d929dec42b641d51098eff"
BLDG_URL = "https://apis.data.go.kr/1613000/BldRgstHubService/getBrTitleInfo"
KAKAO_KEY = "96172db4c3b086f76853ed89242acefa"
KAKAO_ADDRESS_URL = "https://dapi.kakao.com/v2/local/search/address.json"
# 카카오는 WGS84(경도/위도) 반환 → EPSG:5174 (구 GRS80 중부원점) 평면좌표로 변환
TRANSFORMER = Transformer.from_crs("EPSG:4326", "EPSG:5174", always_xy=True)


def norm(value: object) -> str:
    """None 이면 빈 문자열, 아니면 문자열 변환 후 좌우 공백 제거."""
    return "" if value is None else str(value).strip()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """UTF-8-BOM CSV 를 읽어 (헤더 목록, 행 dict 목록) 튜플 반환."""
    with path.open("r", encoding="utf-8-sig", newline="") as fp:
        reader = csv.DictReader(fp)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    """UTF-8-BOM CSV 로 저장 — 한글 엑셀 호환을 위해 BOM 포함."""
    with path.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def valid_date(value: object) -> bool:
    """8자리 숫자(YYYYMMDD) 이고 0 시작이 아닌 유효 날짜인지 확인."""
    text = norm(value)
    # 엑셀에서 정수가 .0 접미사로 들어온 경우 제거
    if text.endswith(".0"):
        text = text[:-2]
    return (
        len(text) == 8
        and text.isdigit()
        and text != "00000000"
        and not text.startswith("0")  # 1000년대 이전은 비정상으로 간주
    )


def normalize_code(value: str, width: int) -> str:
    """시군구코드/법정동코드 등을 width 자리 0패딩 문자열로 정규화."""
    text = norm(value)
    if text.endswith(".0"):
        text = text[:-2]
    return text.zfill(width) if text else "0" * width


def normalize_number(value: str) -> str:
    """본번·부번을 4자리 0패딩 문자열로 — '3' → '0003' (API 인자 규격)."""
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
    """
    도로명주소에서 쉼표 이후 부가정보 제거 + 마지막 괄호 안 동명만 보존.

    예: '서울특별시 마포구 양화로 100, 3층 (서교동, 빌딩명)'
        → '서울특별시 마포구 양화로 100 (서교동)'
    """
    text = norm(value)
    if not text:
        return ""
    # 모든 괄호 안 내용 추출
    matches = re.findall(r"\(([^)]*)\)", text)
    first_paren = ""
    if matches:
        # 마지막 괄호의 첫 번째 항목(쉼표 분리)만 유지
        first = matches[-1].split(",")[0].strip()
        if first:
            first_paren = f" ({first})"
    # 쉼표 이전 본문만 + 공백 정규화
    base = text.split(",")[0].strip()
    base = re.sub(r"\s+", " ", base)
    if first_paren and "(" not in base:
        return f"{base}{first_paren}"
    return base


def fetch_bldg_items(
    sigungu_cd: str, bjdong_cd: str, bun: str, ji: str, session: requests.Session
) -> list[dict]:
    """건축물대장 API (BldRgstHubService) 에서 해당 필지의 표제부 목록 반환."""
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
        return [data]  # 단일 항목이면 리스트로 감쌈 (스키마 일관성)
    return data


def choose_use_approval_date(row: dict[str, str], items: list[dict]) -> tuple[str, str]:
    """
    API 응답에서 가장 신뢰할 수 있는 사용승인일 1개를 선택.

    우선순위:
        1. 관리건축물대장PK 가 정확히 일치하는 항목
        2. 유효 날짜가 단 1개일 때 그 값
        3. '주건축물' 항목들 중에서도 단 1개일 때 그 값
        그 외에는 빈 문자열 ('unresolved')
    """
    pk = norm(row.get("관리건축물대장PK"))

    # 1) PK 정확 일치
    for item in items:
        if norm(item.get("mgmBldrgstPk")) == pk and valid_date(item.get("useAprDay")):
            return norm(item.get("useAprDay")), "pk_exact"

    # 2) 유효 날짜 유일
    valid_items = [item for item in items if valid_date(item.get("useAprDay"))]
    unique_dates = sorted({norm(item.get("useAprDay")) for item in valid_items})
    if len(unique_dates) == 1:
        return unique_dates[0], "single_unique_date"

    # 3) 주건축물 한정 유일
    main_items = [
        item for item in valid_items if norm(item.get("mainAtchGbCdNm")) == "주건축물"
    ]
    main_dates = sorted({norm(item.get("useAprDay")) for item in main_items})
    if len(main_dates) == 1:
        return main_dates[0], "main_building_unique_date"

    return "", "unresolved"


def build_local_candidates() -> tuple[dict[str, list[dict]], dict[str, list[dict]]]:
    """
    로컬 CSV 5종에서 PK/사업장명 기반 도로명·좌표 후보를 수집.

    반환:
        by_pk   : {관리건축물대장PK → [후보 dict, ...]}  (통합표제부에서만)
        by_name : {사업장명 → [후보 dict, ...]}          (5종 모두)
    """
    by_pk: defaultdict[str, list[dict]] = defaultdict(list)
    by_name: defaultdict[str, list[dict]] = defaultdict(list)

    # 통합표제부 — PK 컬럼이 있어 가장 신뢰도 높은 매칭 가능
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

    # 인허가 4종 — PK 없이 사업장명 기반 후보
    for raw_path in RAW_PATHS:
        _, rows = read_csv(raw_path)
        for row in rows:
            rec = {
                "source": raw_path.name,
                "pk": "",
                "name": norm(row.get("사업장명")),
                # 컬럼명이 데이터마다 다를 수 있어 OR 로 fallback
                "road": clean_road_address(
                    row.get("도로명주소") or row.get("도로명대지위치")
                ),
                "x": norm(row.get("좌표정보(X)") or row.get("X좌표")),
                "y": norm(row.get("좌표정보(Y)") or row.get("Y좌표")),
            }
            if rec["name"]:
                by_name[rec["name"]].append(rec)

    return dict(by_pk), dict(by_name)


def choose_local_road(
    row: dict[str, str], by_pk: dict[str, list[dict]], by_name: dict[str, list[dict]]
) -> tuple[str, str]:
    """
    로컬 후보에서 도로명주소 1개 선택.

    우선순위: PK 기반 단일값 → 소방청 매칭 → 사업장명 단일값 → unresolved
    """
    pk = norm(row.get("관리건축물대장PK"))
    name = norm(row.get("사업장명"))

    # PK 후보가 단 1개 도로명만 가질 때 채택
    pk_roads = sorted({cand["road"] for cand in by_pk.get(pk, []) if cand["road"]})
    if len(pk_roads) == 1:
        return pk_roads[0], "pk_local"

    # 소방청 매칭 도로명이 있으면 채택 (별도 정제 거침)
    fire_road = clean_road_address(row.get("소방청_도로명주소_매칭"))
    if fire_road:
        return fire_road, "fire_match"

    # 사업장명으로 좁힌 후보가 단 1개 도로명만 가질 때 채택
    name_roads = sorted(
        {cand["road"] for cand in by_name.get(name, []) if cand["road"]}
    )
    if len(name_roads) == 1:
        return name_roads[0], "name_local"

    return "", "unresolved"


def choose_local_xy(
    row: dict[str, str], by_pk: dict[str, list[dict]], by_name: dict[str, list[dict]]
) -> tuple[str, str, str]:
    """
    로컬 후보에서 (x, y) 좌표 1쌍 선택.

    우선순위: PK 기반 단일 좌표쌍 → 사업장명 단일 좌표쌍 → unresolved
    """
    pk = norm(row.get("관리건축물대장PK"))
    name = norm(row.get("사업장명"))

    def pairs(cands: list[dict]) -> list[tuple[str, str]]:
        """후보 중 (x, y) 모두 채워진 것만 추려 set 으로 중복 제거."""
        return sorted(
            {(cand["x"], cand["y"]) for cand in cands if cand["x"] and cand["y"]}
        )

    pk_pairs = pairs(by_pk.get(pk, []))
    if len(pk_pairs) == 1:
        return pk_pairs[0][0], pk_pairs[0][1], "pk_local"

    name_pairs = pairs(by_name.get(name, []))
    if len(name_pairs) == 1:
        return name_pairs[0][0], name_pairs[0][1], "name_local"

    return "", "", "unresolved"


def kakao_address_doc(
    query: str, session: requests.Session, cache: dict[str, dict]
) -> dict:
    """카카오 지오코딩 API — 1회 호출 후 cache 에 저장 (반복 호출 방지)."""
    query = norm(query)
    if not query:
        return {}
    if query in cache:
        return cache[query]

    try:
        resp = session.get(
            KAKAO_ADDRESS_URL, params={"query": query, "size": 1}, timeout=10
        )
        resp.raise_for_status()
        docs = resp.json().get("documents", [])
        cache[query] = docs[0] if docs else {}
        return cache[query]
    except Exception:
        # 실패도 빈 dict 로 캐싱 — 같은 쿼리 재시도 회피
        cache[query] = {}
        return {}


def kakao_road_from_query(
    query: str, session: requests.Session, cache: dict[str, dict]
) -> str:
    """카카오 응답에서 도로명주소 문자열 추출 (없으면 빈 문자열)."""
    doc = kakao_address_doc(query, session, cache)
    road = (doc.get("road_address") or {}).get("address_name") or ""
    return norm(road)


def kakao_xy_from_query(
    query: str, session: requests.Session, cache: dict[str, dict]
) -> tuple[str, str]:
    """카카오 응답의 위경도(WGS84) → EPSG:5174 평면좌표 변환."""
    doc = kakao_address_doc(query, session, cache)
    if not doc:
        return "", ""
    try:
        lon = float(doc["x"])
        lat = float(doc["y"])
        x, y = TRANSFORMER.transform(lon, lat)
        # 소수점 이하 불필요한 0 제거 — '123456.000000' → '123456'
        return f"{x:.6f}".rstrip("0").rstrip("."), f"{y:.6f}".rstrip("0").rstrip(".")
    except Exception:
        return "", ""


def main() -> None:
    """전체 결측 정비 파이프라인 실행 + JSON 보고서 저장."""
    fieldnames, rows = read_csv(TARGET_PATH)
    report: dict[str, object] = {}

    # 정비 전 결측 카운트 — 보고서 비교 baseline
    before_counts = {
        "rows": len(rows),
        "blank_useAprDay": sum(1 for row in rows if not norm(row.get("사용승인일"))),
        "blank_x": sum(1 for row in rows if not norm(row.get("X좌표"))),
        "blank_y": sum(1 for row in rows if not norm(row.get("Y좌표"))),
        "blank_road": sum(1 for row in rows if not norm(row.get("도로명대지위치"))),
    }
    report["before"] = before_counts

    # ── 1단계: 건축물대장 API 로 사용승인일 채우기 ──
    bldg_session = requests.Session()
    query_cache: dict[tuple[str, str, str, str], list[dict]] = {}
    query_failures: list[dict] = []
    date_stats = Counter()

    # 사용승인일이 비어있는 행만 대상 — 동일 필지는 한 번만 조회 (중복 제거)
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

    # API 호출 — 25개마다 진행 표시, 30ms 슬립으로 rate 보호
    for idx, key in enumerate(query_keys, start=1):
        try:
            query_cache[key] = fetch_bldg_items(*key, session=bldg_session)
        except Exception as exc:
            query_cache[key] = []
            query_failures.append({"key": key, "error": str(exc)})
        if idx % 25 == 0 or idx == len(query_keys):
            print(f"사용승인일 API 조회: {idx}/{len(query_keys)}")
        time.sleep(0.03)

    # 캐시 결과로 각 행에 사용승인일 채우기
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

    # 사용승인일 끝까지 못 채운 행은 분석에서 제외 — 행 자체 삭제
    remaining_blank_date = [row for row in rows if not norm(row.get("사용승인일"))]
    deleted_count = len(remaining_blank_date)
    rows = [row for row in rows if norm(row.get("사용승인일"))]

    # ── 2~3단계: 로컬 후보 + 카카오로 도로명·좌표 채우기 ──
    by_pk, by_name = build_local_candidates()
    kakao_session = requests.Session()
    # 카카오 API 인증 — Authorization 헤더 고정
    kakao_session.headers.update({"Authorization": f"KakaoAK {KAKAO_KEY}"})
    kakao_cache: dict[str, dict] = {}
    fill_stats = Counter()

    for row in rows:
        # 도로명대지위치 결측 → 로컬 후보 → 카카오 폴백
        if not norm(row.get("도로명대지위치")):
            road, strategy = choose_local_road(row, by_pk, by_name)
            if not road:
                # 지번주소 또는 소방청 지번 매칭으로 카카오 검색
                road = kakao_road_from_query(
                    norm(row.get("대지위치")) or norm(row.get("소방청_지번주소_매칭")),
                    kakao_session,
                    kakao_cache,
                )
                strategy = "kakao_jibun" if road else "unresolved"
            if road:
                row["도로명대지위치"] = road
            fill_stats[f"road_{strategy}"] += 1

        # X/Y 좌표 결측 → 로컬 후보 → 카카오 폴백
        if not norm(row.get("X좌표")) or not norm(row.get("Y좌표")):
            x, y, strategy = choose_local_xy(row, by_pk, by_name)
            if not (x and y):
                # 카카오 검색 쿼리 — 도로명 우선, 차례로 fallback
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

    # 정비된 데이터 덮어쓰기
    write_csv(TARGET_PATH, fieldnames, rows)

    # 정비 후 결측 카운트
    after_counts = {
        "rows": len(rows),
        "blank_useAprDay": sum(1 for row in rows if not norm(row.get("사용승인일"))),
        "blank_x": sum(1 for row in rows if not norm(row.get("X좌표"))),
        "blank_y": sum(1 for row in rows if not norm(row.get("Y좌표"))),
        "blank_road": sum(1 for row in rows if not norm(row.get("도로명대지위치"))),
    }

    # 보고서 저장 — 처음 20개 실패 키만 노출
    report["use_approval_fill"] = {
        "query_key_count": len(query_keys),
        "strategy_counts": dict(date_stats),
        "api_failures": query_failures[:20],
    }
    report["deleted_rows_after_use_approval_fill"] = deleted_count
    report["xy_road_fill"] = dict(fill_stats)
    report["after"] = after_counts
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8-sig"
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
