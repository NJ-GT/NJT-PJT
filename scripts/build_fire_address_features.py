# -*- coding: utf-8 -*-
"""
[파일 설명]
소방청 특정소방대상물소방시설정보서비스 CSV와 통합숙박시설 표제부를 매핑하여
각 소방 대상물에 지번주소·도로명주소·등기부등본 PK를 부여하는 스크립트.

주요 역할:
  1. 숙박시설 표제부에서 이름/주소 역인덱스 생성
  2. 소방청 각 행을 대상으로 이름+주소 → 이름만 → 주소만 순으로 매핑 시도
  3. 매핑 실패 행은 카카오 API로 주소→좌표 변환 후 재시도
  4. 최종 결과와 미매칭 행 각각 CSV 저장

입력: 소방청_특정소방대상물소방시설정보서비스0414.csv
      통합숙박시설표제부0414.csv
출력: data/소방청_특정소방대상물_주소피처.csv (매핑 결과)
      data/소방청_특정소방대상물_주소피처_미매칭.csv (미매핑 행)
"""
import csv
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd
import requests


BASE_DIR = Path(__file__).resolve().parent.parent
FIRE_PATH = BASE_DIR / "소방청_특정소방대상물소방시설정보서비스0414.csv"
HOSPITALITY_PATH = BASE_DIR / "통합숙박시설표제부0414.csv"
OUTPUT_DIR = BASE_DIR / "data"
OUTPUT_PATH = OUTPUT_DIR / "소방청_특정소방대상물_주소피처.csv"
UNMATCHED_PATH = OUTPUT_DIR / "소방청_특정소방대상물_주소피처_미매칭.csv"

KAKAO_KEY = "96172db4c3b086f76853ed89242acefa"


@dataclass
class MatchResult:
    lot_addr: str = ""
    road_addr: str = ""
    reg_name: str = ""
    registry_pk: str = ""
    source_file: str = ""
    match_type: str = ""
    match_score: float = 0.0


def clean_cell(value):
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]
    text = text.replace('""', '"').strip()
    return re.sub(r"\s+", " ", text)


def normalize_address(text):
    text = clean_cell(text)
    text = re.sub(r"\([^)]*\)", " ", text)
    text = text.replace("번지", " ")
    text = text.replace(",", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compact_korean(text):
    text = normalize_address(text)
    text = re.sub(r"[^0-9A-Za-z가-힣]", "", text)
    return text.lower()


def extract_gu(text):
    match = re.search(r"(?:서울특별시\s+)?([가-힣]+구)", clean_cell(text))
    return match.group(1) if match else ""


def extract_dong(text):
    tokens = normalize_address(text).split()
    for token in tokens:
        if re.fullmatch(r"[가-힣0-9]+(?:동|가)", token):
            return token
    return ""


def name_variants(text):
    raw = clean_cell(text)
    items = {raw}
    items.add(re.sub(r"\([^)]*\)", " ", raw))
    for group in re.findall(r"\(([^)]*)\)", raw):
        items.add(group)

    expanded = set()
    for item in items:
        for piece in re.split(r"[,/·]", item):
            expanded.add(piece)

    variants = set()
    for item in expanded:
        item = clean_cell(item)
        item = re.sub(r"^(?:구|현|신)\.?\s*", "", item)
        item = re.sub(r"\b(?:old|new)\b", " ", item, flags=re.I)
        key = re.sub(r"[^0-9A-Za-z가-힣]", "", item).lower()
        if len(key) >= 2:
            variants.add(key)

    return sorted(variants, key=lambda x: (-len(x), x))


def best_name_similarity(left_variants, right_variants):
    best = 0.0
    for left in left_variants:
        for right in right_variants:
            if left == right:
                return 1.0
            if left in right or right in left:
                shorter = min(len(left), len(right))
                longer = max(len(left), len(right))
                score = 0.92 + (shorter / max(longer, 1)) * 0.05
            else:
                score = SequenceMatcher(None, left, right).ratio()
            if score > best:
                best = score
    return best


def read_fire_csv(path):
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as fp:
        reader = csv.reader(fp)
        header = next(reader)
        rows = []
        for row in reader:
            if len(row) > len(header):
                row = row[:11] + [",".join(row[11:-1]), row[-1]]
            elif len(row) < len(header):
                row = row + [""] * (len(header) - len(row))
            rows.append(row)

    frame = pd.DataFrame(rows, columns=header)
    for column in frame.columns:
        frame[column] = frame[column].map(clean_cell)
    return frame


def discover_registry_files(base_dir):
    patterns = ["03. 표제부_*.csv", "등기부등본_표제부_*.csv"]
    files = []
    for pattern in patterns:
        files.extend(sorted(base_dir.glob(pattern)))
    return files


def load_registry(files):
    frames = []
    for path in files:
        frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
        frame = frame[["대지위치", "도로명대지위치", "건물명", "관리건축물대장PK"]].copy()
        frame["source_file"] = path.name
        frames.append(frame)

    registry = pd.concat(frames, ignore_index=True)
    registry = registry.drop_duplicates(subset=["관리건축물대장PK"]).copy()
    registry["대지위치"] = registry["대지위치"].map(clean_cell)
    registry["도로명대지위치"] = registry["도로명대지위치"].map(clean_cell)
    registry["건물명"] = registry["건물명"].map(clean_cell)
    registry["관리건축물대장PK"] = registry["관리건축물대장PK"].map(clean_cell)
    registry["source_file"] = registry["source_file"].map(clean_cell)
    registry["gu"] = registry["대지위치"].map(extract_gu)
    registry["dong"] = registry["대지위치"].map(extract_dong)
    registry["lot_key"] = registry["대지위치"].map(compact_korean)
    registry["road_key"] = registry["도로명대지위치"].map(compact_korean)
    registry["name_variants"] = registry["건물명"].map(name_variants)
    registry = registry[
        registry["gu"].ne("") & registry["dong"].ne("") & registry["name_variants"].map(bool)
    ].copy()
    return registry


def load_hospitality_bridge(path):
    if not path.exists():
        return pd.DataFrame()

    frame = pd.read_csv(
        path,
        encoding="utf-8-sig",
        low_memory=False,
        usecols=[
            "사업장명",
            "지번주소",
            "도로명주소",
            "selected_registry_pk",
            "registry_대지위치",
            "registry_도로명대지위치",
        ],
    ).copy()

    frame["대지위치"] = frame["registry_대지위치"].fillna(frame["지번주소"])
    frame["도로명대지위치"] = frame["registry_도로명대지위치"].fillna(frame["도로명주소"])
    frame["건물명"] = frame["사업장명"]
    frame["관리건축물대장PK"] = frame["selected_registry_pk"].fillna("")
    frame["source_file"] = path.name
    frame = frame[["대지위치", "도로명대지위치", "건물명", "관리건축물대장PK", "source_file"]].copy()

    frame["대지위치"] = frame["대지위치"].map(clean_cell)
    frame["도로명대지위치"] = frame["도로명대지위치"].map(clean_cell)
    frame["건물명"] = frame["건물명"].map(clean_cell)
    frame["관리건축물대장PK"] = frame["관리건축물대장PK"].map(clean_cell)
    frame["source_file"] = frame["source_file"].map(clean_cell)
    frame["gu"] = frame["대지위치"].map(extract_gu)
    frame["dong"] = frame["대지위치"].map(extract_dong)
    frame["lot_key"] = frame["대지위치"].map(compact_korean)
    frame["road_key"] = frame["도로명대지위치"].map(compact_korean)
    frame["name_variants"] = frame["건물명"].map(name_variants)

    frame = frame[
        frame["gu"].ne("") & frame["dong"].ne("") & frame["name_variants"].map(bool)
    ].copy()
    frame = frame.drop_duplicates(subset=["건물명", "대지위치", "도로명대지위치"]).copy()
    return frame


def build_registry_indexes(registry):
    exact_gu_dong = defaultdict(set)
    exact_gu = defaultdict(set)
    by_gu_dong = defaultdict(list)
    by_gu = defaultdict(list)

    for idx, row in registry.iterrows():
        gu = row["gu"]
        dong = row["dong"]
        if gu:
            by_gu[gu].append(idx)
        if gu and dong:
            by_gu_dong[(gu, dong)].append(idx)

        for variant in row["name_variants"]:
            if gu and dong:
                exact_gu_dong[(gu, dong, variant)].add(idx)
            if gu:
                exact_gu[(gu, variant)].add(idx)

    return exact_gu_dong, exact_gu, by_gu_dong, by_gu


def score_candidate(row, fire_name_variants, fire_addr_key, fire_gu, fire_dong):
    name_score = best_name_similarity(fire_name_variants, row["name_variants"])
    score = name_score * 10.0

    if fire_gu and row["gu"] == fire_gu:
        score += 1.0
    if fire_dong and row["dong"] == fire_dong:
        score += 1.5

    if fire_addr_key:
        if fire_addr_key in row["lot_key"] or row["lot_key"] in fire_addr_key:
            score += 2.5
        if fire_addr_key in row["road_key"] or row["road_key"] in fire_addr_key:
            score += 2.0
        if re.search(r"\d", fire_addr_key) and (
            fire_addr_key in row["lot_key"] or row["lot_key"] in fire_addr_key
        ):
            score += 2.0

    return score, name_score


def match_from_registry(
    fire_row,
    registry,
    exact_gu_dong,
    exact_gu,
    by_gu_dong,
    by_gu,
):
    fire_name = clean_cell(fire_row["대상물명"])
    fire_addr = clean_cell(fire_row["주소"])
    fire_gu = clean_cell(fire_row["시군구명"]) or extract_gu(fire_addr)
    fire_dong = clean_cell(fire_row["읍면동명"]) or extract_dong(fire_addr)
    fire_name_vars = name_variants(fire_name)
    fire_addr_key = compact_korean(fire_addr)

    for variant in fire_name_vars:
        ids = exact_gu_dong.get((fire_gu, fire_dong, variant), set())
        if len(ids) == 1:
            matched = registry.loc[next(iter(ids))]
            return MatchResult(
                lot_addr=matched["대지위치"],
                road_addr=matched["도로명대지위치"],
                reg_name=matched["건물명"],
                registry_pk=matched["관리건축물대장PK"],
                source_file=matched["source_file"],
                match_type="exact_name_gu_dong",
                match_score=100.0,
            )

    for variant in fire_name_vars:
        ids = exact_gu.get((fire_gu, variant), set())
        if len(ids) == 1:
            matched = registry.loc[next(iter(ids))]
            return MatchResult(
                lot_addr=matched["대지위치"],
                road_addr=matched["도로명대지위치"],
                reg_name=matched["건물명"],
                registry_pk=matched["관리건축물대장PK"],
                source_file=matched["source_file"],
                match_type="exact_name_gu",
                match_score=99.0,
            )

    candidates = []
    if fire_gu and fire_dong:
        candidates.extend(by_gu_dong.get((fire_gu, fire_dong), []))
    if not candidates and fire_gu:
        candidates.extend(by_gu.get(fire_gu, []))

    if not candidates:
        return MatchResult()

    best_idx = None
    best_score = -1.0
    best_name_score = 0.0
    second_score = -1.0

    for idx in candidates:
        reg_row = registry.loc[idx]
        score, name_score = score_candidate(
            reg_row,
            fire_name_vars,
            fire_addr_key,
            fire_gu,
            fire_dong,
        )
        if score > best_score:
            second_score = best_score
            best_score = score
            best_name_score = name_score
            best_idx = idx
        elif score > second_score:
            second_score = score

    if best_idx is None:
        return MatchResult()

    min_name = 0.76 if fire_dong else 0.82
    min_gap = 0.45 if fire_dong else 0.8
    if best_name_score < min_name:
        return MatchResult()
    if second_score >= 0 and (best_score - second_score) < min_gap:
        return MatchResult()

    matched = registry.loc[best_idx]
    return MatchResult(
        lot_addr=matched["대지위치"],
        road_addr=matched["도로명대지위치"],
        reg_name=matched["건물명"],
        registry_pk=matched["관리건축물대장PK"],
        source_file=matched["source_file"],
        match_type="fuzzy_name_address",
        match_score=round(best_score, 2),
    )


def search_kakao_keyword(session, query):
    if not query:
        return []
    try:
        response = session.get(
            "https://dapi.kakao.com/v2/local/search/keyword.json",
            params={"query": query, "size": 5},
            timeout=10,
        )
        response.raise_for_status()
        return response.json().get("documents", [])
    except Exception:
        return []


def fallback_from_kakao(fire_row, session):
    fire_name = clean_cell(fire_row["대상물명"])
    fire_gu = clean_cell(fire_row["시군구명"])
    fire_dong = clean_cell(fire_row["읍면동명"])
    fire_name_vars = name_variants(fire_name)

    queries = []
    for query in [
        f"{fire_name} {fire_gu}".strip(),
        f"{fire_name} {fire_dong}".strip(),
        fire_name,
    ]:
        if query and query not in queries:
            queries.append(query)

    best_doc = None
    best_score = 0.0
    for query in queries:
        for doc in search_kakao_keyword(session, query):
            place_name = clean_cell(doc.get("place_name", ""))
            address_name = clean_cell(doc.get("address_name", ""))
            road_name = clean_cell(doc.get("road_address_name", ""))
            score = best_name_similarity(fire_name_vars, name_variants(place_name)) * 10.0
            if fire_gu and fire_gu in f"{address_name} {road_name}":
                score += 1.5
            if fire_dong and fire_dong in f"{address_name} {road_name}":
                score += 1.0
            if score > best_score:
                best_score = score
                best_doc = doc
        time.sleep(0.04)

    if best_doc is None or best_score < 8.8:
        return MatchResult()

    return MatchResult(
        lot_addr=clean_cell(best_doc.get("address_name", "")),
        road_addr=clean_cell(best_doc.get("road_address_name", "")),
        reg_name=clean_cell(best_doc.get("place_name", "")),
        registry_pk="",
        source_file="kakao_keyword",
        match_type="kakao_keyword",
        match_score=round(best_score, 2),
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] 소방청 CSV 로드: {FIRE_PATH.name}")
    fire = read_fire_csv(FIRE_PATH)
    print(f"  행 수: {len(fire):,}")

    print("[2/5] 표제부 파일 탐색")
    registry_files = discover_registry_files(BASE_DIR)
    print(f"  표제부 파일 수: {len(registry_files)}")

    print("[3/5] 표제부 로드 및 인덱스 생성")
    registry = load_registry(registry_files)
    exact_gu_dong, exact_gu, by_gu_dong, by_gu = build_registry_indexes(registry)
    print(f"  표제부 행 수: {len(registry):,}")
    hospitality = load_hospitality_bridge(HOSPITALITY_PATH)
    hospitality_exact_gu_dong, hospitality_exact_gu, hospitality_by_gu_dong, hospitality_by_gu = (
        build_registry_indexes(hospitality) if not hospitality.empty else ({}, {}, {}, {})
    )
    if not hospitality.empty:
        print(f"  숙박업 보조 행 수: {len(hospitality):,}")

    session = requests.Session()
    session.headers.update({"Authorization": f"KakaoAK {KAKAO_KEY}"})

    print("[4/5] 주소 피처 매칭")
    results = []
    for idx, (_, fire_row) in enumerate(fire.iterrows(), start=1):
        result = match_from_registry(
            fire_row,
            registry,
            exact_gu_dong,
            exact_gu,
            by_gu_dong,
            by_gu,
        )
        if not result.match_type and not hospitality.empty:
            result = match_from_registry(
                fire_row,
                hospitality,
                hospitality_exact_gu_dong,
                hospitality_exact_gu,
                hospitality_by_gu_dong,
                hospitality_by_gu,
            )
            if result.match_type:
                result.match_type = f"hospitality_{result.match_type}"
        if not result.match_type:
            result = fallback_from_kakao(fire_row, session)
        results.append(result)
        if idx % 200 == 0:
            print(f"  진행: {idx:,}/{len(fire):,}")

    fire["지번주소_매칭"] = [result.lot_addr for result in results]
    fire["도로명주소_매칭"] = [result.road_addr for result in results]
    fire["매칭건물명"] = [result.reg_name for result in results]
    fire["매칭건축물대장PK"] = [result.registry_pk for result in results]
    fire["매칭소스"] = [result.source_file for result in results]
    fire["매칭방식"] = [result.match_type for result in results]
    fire["매칭점수"] = [result.match_score for result in results]

    matched = fire["매칭방식"].ne("")
    print(f"  매칭 성공: {matched.sum():,}/{len(fire):,}")
    print(f"  미매칭: {(~matched).sum():,}")
    print("  매칭 방식별 건수:")
    for key, value in fire["매칭방식"].value_counts().items():
        print(f"    {key}: {value:,}")

    print("[5/5] 결과 저장")
    fire.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    fire.loc[~matched].to_csv(UNMATCHED_PATH, index=False, encoding="utf-8-sig")

    print(f"  CSV 저장: {OUTPUT_PATH}")
    print(f"  미매칭 저장: {UNMATCHED_PATH}")


if __name__ == "__main__":
    main()
