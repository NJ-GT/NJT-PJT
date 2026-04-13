# -*- coding: utf-8 -*-
import os
import re
import time
from difflib import SequenceMatcher

import pandas as pd
import requests


BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
TARGET_PATH = os.path.join(BASE_DIR, "소방청_특정소방대상물소방시설정보서비스.csv")
MERGED_PATH = os.path.join(BASE_DIR, "등기부등본_소방청병합_결과.csv")
OTHER_FIRE_PATH = os.path.join(BASE_DIR, "소방청_특정소방대상물정보서비스.csv")
KAKAO_KEY = "96172db4c3b086f76853ed89242acefa"


COL_BUILDING_ID = "\uac74\ubb3c\uc77c\ub828\ubc88\ud638"
COL_BASE_ADDR = "\uae30\ubcf8\uc8fc\uc18c"
COL_NAME = "\ub300\uc0c1\ubb3c\uba85"
COL_GU = "\uc2dc\uad70\uad6c\uba85"
COL_X = "X\uc88c\ud45c"
COL_Y = "Y\uc88c\ud45c"
COL_ROAD = "\ub3c4\ub85c\uba85\uc8fc\uc18c"
COL_BUILDING_NAME = "\uac74\ubb3c\uba85"

MERGED_ROAD = "\ub3c4\ub85c\uba85\ub300\uc9c0\uc704\uce58"
MERGED_LOT = "\ub300\uc9c0\uc704\uce58"
MERGED_MATCH = "match_Method"


session = requests.Session()
session.headers.update({"Authorization": f"KakaoAK {KAKAO_KEY}"})

trans_cache = {}
search_cache = {}
coord_addr_cache = {}
address_search_cache = {}


def normalize_text(text):
    text = "" if pd.isna(text) else str(text)
    text = re.sub(r"\[[^\]]*\]", " ", text)
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"[^0-9A-Za-z\uac00-\ud7a3]", "", text)
    return text.lower().strip()


def clean_name(text):
    text = "" if pd.isna(text) else str(text).strip()
    text = re.sub(r"\[[^\]]*\]", " ", text)
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"\b(?:old|new)\b", " ", text, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_dong(addr):
    addr = "" if pd.isna(addr) else str(addr)
    parts = addr.split()
    return parts[-1] if parts else ""


def name_similarity(a, b):
    na = normalize_text(a)
    nb = normalize_text(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    if na in nb or nb in na:
        return 0.92
    return SequenceMatcher(None, na, nb).ratio()


def to_wgs84(x, y):
    key = (round(float(x), 2), round(float(y), 2))
    if key in trans_cache:
        return trans_cache[key]
    try:
        resp = session.get(
            "https://dapi.kakao.com/v2/local/geo/transcoord.json",
            params={
                "x": x,
                "y": y,
                "input_coord": "WTM",
                "output_coord": "WGS84",
            },
            timeout=10,
        )
        docs = resp.json().get("documents", [])
        if docs:
            value = (docs[0]["x"], docs[0]["y"])
            trans_cache[key] = value
            return value
    except Exception:
        pass
    trans_cache[key] = (None, None)
    return None, None


def reverse_coord_address(lon, lat):
    key = (round(float(lon), 6), round(float(lat), 6))
    if key in coord_addr_cache:
        return coord_addr_cache[key]
    try:
        resp = session.get(
            "https://dapi.kakao.com/v2/local/geo/coord2address.json",
            params={"x": lon, "y": lat, "input_coord": "WGS84"},
            timeout=10,
        )
        docs = resp.json().get("documents", [])
        if docs:
            doc = docs[0]
            road = ((doc.get("road_address") or {}).get("address_name") or "").strip()
            jibun = ((doc.get("address") or {}).get("address_name") or "").strip()
            coord_addr_cache[key] = (road, jibun)
            return road, jibun
    except Exception:
        pass
    coord_addr_cache[key] = ("", "")
    return "", ""


def kakao_address_search(query):
    query = str(query).strip()
    if not query:
        return []
    if query in address_search_cache:
        return address_search_cache[query]
    try:
        resp = session.get(
            "https://dapi.kakao.com/v2/local/search/address.json",
            params={"query": query},
            timeout=10,
        )
        docs = resp.json().get("documents", [])
        address_search_cache[query] = docs
        return docs
    except Exception:
        address_search_cache[query] = []
        return []


def kakao_keyword_search(query, lon, lat, radius=3000):
    cache_key = (query, round(float(lon), 5), round(float(lat), 5), radius)
    if cache_key in search_cache:
        return search_cache[cache_key]
    try:
        resp = session.get(
            "https://dapi.kakao.com/v2/local/search/keyword.json",
            params={
                "query": query,
                "x": lon,
                "y": lat,
                "radius": radius,
                "size": 5,
                "sort": "distance",
            },
            timeout=10,
        )
        docs = resp.json().get("documents", [])
        search_cache[cache_key] = docs
        return docs
    except Exception:
        search_cache[cache_key] = []
        return []


def build_queries(name, gu):
    cleaned = clean_name(name)
    compact = re.sub(r"\s+", "", cleaned)
    variants = []
    for candidate in [cleaned, compact]:
        if candidate:
            variants.append(f"{candidate} {gu}".strip())
            variants.append(candidate)
    deduped = []
    seen = set()
    for item in variants:
        key = item.strip()
        if key and key not in seen:
            seen.add(key)
            deduped.append(key)
    return deduped


def best_aux_keyword_road(row, aux_name):
    aux_name = clean_name(aux_name)
    if not aux_name:
        return None
    x = pd.to_numeric(row[COL_X], errors="coerce")
    y = pd.to_numeric(row[COL_Y], errors="coerce")
    if pd.isna(x) or pd.isna(y):
        return None
    lon, lat = to_wgs84(x, y)
    if not lon or not lat:
        return None

    best_doc = None
    best_score = -1.0
    for query in build_queries(aux_name, row[COL_GU]):
        for doc in kakao_keyword_search(query, lon, lat, radius=5000):
            road = doc.get("road_address_name", "").strip()
            if not road or not road_matches_gu(row, road):
                continue
            score = name_similarity(aux_name, doc.get("place_name", "")) * 5.0
            try:
                distance = float(doc.get("distance", 999999))
            except Exception:
                distance = 999999.0
            score += max(0.0, 2.0 - min(distance, 3000.0) / 1500.0)
            if score > best_score:
                best_score = score
                best_doc = doc
        time.sleep(0.03)

    if best_doc and best_score >= 5.3:
        return best_doc.get("road_address_name", "").strip()
    return None


def score_kakao_doc(row, doc):
    place_name = doc.get("place_name", "")
    address_name = doc.get("address_name", "")
    road_name = doc.get("road_address_name", "")
    gu = str(row[COL_GU]).strip()
    dong = extract_dong(row[COL_BASE_ADDR])
    score = 0.0
    score += name_similarity(row[COL_NAME], place_name) * 5.0
    if gu and gu in f"{address_name} {road_name}":
        score += 2.0
    if dong and dong in f"{address_name} {road_name}":
        score += 2.0
    try:
        distance = float(doc.get("distance", 999999))
    except Exception:
        distance = 999999.0
    score += max(0.0, 2.0 - min(distance, 2000.0) / 1000.0)
    return score


def road_matches_gu(row, road):
    gu = str(row[COL_GU]).strip()
    road = "" if pd.isna(road) else str(road).strip()
    return bool(gu and road and gu in road)


def best_kakao_road(row):
    x = pd.to_numeric(row[COL_X], errors="coerce")
    y = pd.to_numeric(row[COL_Y], errors="coerce")
    if pd.isna(x) or pd.isna(y):
        return None
    lon, lat = to_wgs84(x, y)
    if not lon or not lat:
        return None

    best_doc = None
    best_score = -1.0
    for query in build_queries(row[COL_NAME], row[COL_GU]):
        for doc in kakao_keyword_search(query, lon, lat):
            road = doc.get("road_address_name", "").strip()
            if not road:
                continue
            score = score_kakao_doc(row, doc)
            if score > best_score:
                best_score = score
                best_doc = doc
        time.sleep(0.03)

    if best_doc and best_score >= 4.6:
        road = best_doc.get("road_address_name", "").strip()
        if road_matches_gu(row, road):
            return road
    return None


def prepare_merged_candidates(merged):
    obj_col = next(c for c in merged.columns if "objNm" in c)
    x_col = next(c for c in merged.columns if "xCrdntVal" in c)
    y_col = next(c for c in merged.columns if "yCrdntVal" in c)

    subset = merged[[obj_col, MERGED_LOT, MERGED_ROAD, MERGED_MATCH, x_col, y_col]].copy()
    subset = subset.dropna(subset=[obj_col, MERGED_ROAD])
    subset[obj_col] = subset[obj_col].astype(str).str.strip()
    subset[MERGED_LOT] = subset[MERGED_LOT].fillna("").astype(str).str.strip()
    subset[MERGED_ROAD] = subset[MERGED_ROAD].fillna("").astype(str).str.strip()
    subset["_name_norm"] = subset[obj_col].apply(normalize_text)
    subset[x_col] = pd.to_numeric(subset[x_col], errors="coerce")
    subset[y_col] = pd.to_numeric(subset[y_col], errors="coerce")
    return subset, obj_col, x_col, y_col


def best_merged_road(row, merged_subset, obj_col, x_col, y_col):
    name_norm = normalize_text(row[COL_NAME])
    if not name_norm:
        return None

    candidates = merged_subset[merged_subset["_name_norm"] == name_norm].copy()
    if candidates.empty:
        sims = merged_subset[obj_col].apply(lambda x: name_similarity(row[COL_NAME], x))
        candidates = merged_subset[sims >= 0.9].copy()
        if candidates.empty:
            return None

    base_addr = str(row[COL_BASE_ADDR]).strip()
    x = pd.to_numeric(row[COL_X], errors="coerce")
    y = pd.to_numeric(row[COL_Y], errors="coerce")

    candidates["_addr_match"] = candidates[MERGED_LOT].apply(lambda x: base_addr and base_addr in x)
    if not pd.isna(x) and not pd.isna(y):
        candidates["_dist"] = ((candidates[x_col] - x) ** 2 + (candidates[y_col] - y) ** 2) ** 0.5
    else:
        candidates["_dist"] = 999999.0
    candidates["_sim"] = candidates[obj_col].apply(lambda v: name_similarity(row[COL_NAME], v))
    candidates = candidates.sort_values(
        by=["_addr_match", "_sim", "_dist"],
        ascending=[False, False, True],
    )
    best = candidates.iloc[0]
    if best["_addr_match"] or best["_sim"] >= 0.95:
        if road_matches_gu(row, best[MERGED_ROAD]):
            return best[MERGED_ROAD]
    return None


def best_nearest_same_gu_road(row, merged_subset, x_col, y_col, max_dist=70.0):
    x = pd.to_numeric(row[COL_X], errors="coerce")
    y = pd.to_numeric(row[COL_Y], errors="coerce")
    if pd.isna(x) or pd.isna(y):
        return None
    gu = str(row[COL_GU]).strip()
    candidates = merged_subset[merged_subset[MERGED_ROAD].astype(str).str.contains(gu, regex=False, na=False)].copy()
    if candidates.empty:
        return None
    candidates["_dist"] = ((candidates[x_col] - x) ** 2 + (candidates[y_col] - y) ** 2) ** 0.5
    best = candidates.sort_values("_dist").iloc[0]
    if pd.notna(best["_dist"]) and float(best["_dist"]) <= max_dist:
        return best[MERGED_ROAD]
    return None


def best_reverse_road(row):
    x = pd.to_numeric(row[COL_X], errors="coerce")
    y = pd.to_numeric(row[COL_Y], errors="coerce")
    if pd.isna(x) or pd.isna(y):
        return None
    lon, lat = to_wgs84(x, y)
    if not lon or not lat:
        return None

    road, jibun = reverse_coord_address(lon, lat)
    if road and road_matches_gu(row, road):
        return road

    if jibun:
        for doc in kakao_address_search(jibun):
            road2 = ((doc.get("road_address") or {}).get("address_name") or "").strip()
            if road2 and road_matches_gu(row, road2):
                return road2
    return None


def main():
    target = pd.read_csv(TARGET_PATH, encoding="utf-8-sig", low_memory=False)
    merged = pd.read_csv(MERGED_PATH, encoding="utf-8-sig", low_memory=False)
    other_fire = pd.read_csv(OTHER_FIRE_PATH, encoding="utf-8-sig", low_memory=False)

    for col in [COL_BUILDING_ID, COL_BASE_ADDR, COL_NAME, COL_GU]:
        target[col] = target[col].fillna("").astype(str).str.strip()
    for col in [COL_X, COL_Y]:
        target[col] = pd.to_numeric(target[col], errors="coerce")

    aux_building = (
        other_fire[[COL_BUILDING_ID, COL_BUILDING_NAME]]
        .drop_duplicates()
        .rename(columns={COL_BUILDING_NAME: "_aux_building_name"})
    )
    aux_building[COL_BUILDING_ID] = aux_building[COL_BUILDING_ID].fillna("").astype(str).str.strip()
    aux_building["_aux_building_name"] = aux_building["_aux_building_name"].fillna("").astype(str).str.strip()
    target = target.merge(aux_building, on=COL_BUILDING_ID, how="left")

    target[COL_ROAD] = ""

    merged_subset, obj_col, x_col, y_col = prepare_merged_candidates(merged)

    bass_addr_col = [c for c in merged.columns if "bassAdres" in c][0]
    direct_exact = merged[[bass_addr_col, obj_col, MERGED_ROAD]].dropna(subset=[MERGED_ROAD]).copy()
    direct_exact.columns = [COL_BASE_ADDR, COL_NAME, COL_ROAD]
    direct_exact[COL_BASE_ADDR] = direct_exact[COL_BASE_ADDR].fillna("").astype(str).str.strip()
    direct_exact[COL_NAME] = direct_exact[COL_NAME].fillna("").astype(str).str.strip()
    direct_exact[COL_ROAD] = direct_exact[COL_ROAD].fillna("").astype(str).str.strip()
    direct_exact = direct_exact.drop_duplicates()

    exact_map = {}
    for _, row in direct_exact.iterrows():
        key = (row[COL_BASE_ADDR], row[COL_NAME])
        exact_map.setdefault(key, []).append(row[COL_ROAD])

    exact_filled = 0
    kakao_filled = 0
    merged_filled = 0
    reverse_filled = 0
    aux_filled = 0
    nearest_filled = 0

    for idx, row in target.iterrows():
        if target.at[idx, COL_ROAD]:
            continue

        key = (row[COL_BASE_ADDR], row[COL_NAME])
        exact_roads = list(dict.fromkeys(exact_map.get(key, [])))
        if len(exact_roads) == 1 and road_matches_gu(row, exact_roads[0]):
            target.at[idx, COL_ROAD] = exact_roads[0]
            exact_filled += 1
            continue

        road = best_kakao_road(row)
        if road:
            target.at[idx, COL_ROAD] = road
            kakao_filled += 1
            continue

        road = best_merged_road(row, merged_subset, obj_col, x_col, y_col)
        if road:
            target.at[idx, COL_ROAD] = road
            merged_filled += 1
            continue

        road = best_reverse_road(row)
        if road:
            target.at[idx, COL_ROAD] = road
            reverse_filled += 1
            continue

        road = best_aux_keyword_road(row, row.get("_aux_building_name", ""))
        if road:
            target.at[idx, COL_ROAD] = road
            aux_filled += 1
            continue

        road = best_nearest_same_gu_road(row, merged_subset, x_col, y_col)
        if road:
            target.at[idx, COL_ROAD] = road
            nearest_filled += 1

        if (idx + 1) % 100 == 0:
            print(f"processed {idx + 1}/{len(target)}")

    target = target.drop(columns=["_aux_building_name"])
    target.to_csv(TARGET_PATH, index=False, encoding="utf-8-sig")

    filled = target[COL_ROAD].astype(str).str.strip().ne("").sum()
    print(f"rows={len(target)}")
    print(f"filled={filled}")
    print(f"exact={exact_filled}")
    print(f"kakao={kakao_filled}")
    print(f"merged={merged_filled}")
    print(f"reverse={reverse_filled}")
    print(f"aux={aux_filled}")
    print(f"nearest={nearest_filled}")
    print(f"missing={len(target) - filled}")
    print(
        target.loc[
            target[COL_ROAD].astype(str).str.strip().ne(""),
            [COL_BASE_ADDR, COL_NAME, COL_ROAD],
        ]
        .head(20)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
