# -*- coding: utf-8 -*-
import csv
import re
import unicodedata
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
FEATURE_PATH = BASE_DIR / "data" / "통합숙박시설표제부0415_피처선정본.csv"
FIRE_PATH = BASE_DIR / "data" / "소방청_특정소방대상물_주소피처.csv"
OUTPUT_PATH = BASE_DIR / "data" / "통합숙박시설표제부0415_피처선정본_소방청결합.csv"
NAME_ONLY_PATH = BASE_DIR / "data" / "통합숙박시설표제부0415_피처선정본_소방청_이름만일치.csv"
ADDR_ONLY_PATH = BASE_DIR / "data" / "통합숙박시설표제부0415_피처선정본_소방청_주소만일치.csv"
SPLIT_PATH = BASE_DIR / "data" / "통합숙박시설표제부0415_피처선정본_소방청_이름주소각자일치.csv"
AMBIGUOUS_PATH = BASE_DIR / "data" / "통합숙박시설표제부0415_피처선정본_소방청_복수후보.csv"


def clean_text(value):
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]
    text = text.replace('""', '"').strip()
    return re.sub(r"\s+", " ", text)


def normalize_text(value):
    return unicodedata.normalize("NFKC", clean_text(value)).strip()


def normalize_address(value):
    text = normalize_text(value).lower()
    text = text.replace("번지", "")
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"[^0-9a-z가-힣]", "", text)
    return text


def strip_company_prefix(text):
    for prefix in ["주식회사", "㈜", "(주)"]:
        if text.startswith(prefix):
            return text[len(prefix) :].strip()
    return text


def name_variants(text):
    raw = normalize_text(text).lower()
    pieces = {raw, re.sub(r"\([^)]*\)", " ", raw)}
    for group in re.findall(r"\(([^)]*)\)", raw):
        pieces.add(group)

    expanded = set()
    for piece in pieces:
        for part in re.split(r"[,/&·]", piece):
            expanded.add(part)

    variants = set()
    for part in expanded:
        part = strip_company_prefix(clean_text(part))
        key = re.sub(r"[^0-9a-z가-힣]", "", part)
        if len(key) >= 2:
            variants.add(key)
    return variants


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
        frame[column] = frame[column].map(clean_text)
    return frame


def candidate_preview(fire, indices):
    previews = []
    for idx in sorted(indices)[:5]:
        row = fire.loc[idx]
        previews.append(
            " / ".join(
                [
                    clean_text(row["대상물명"]),
                    clean_text(row["지번주소_매칭"]),
                    clean_text(row["도로명주소_매칭"]),
                ]
            )
        )
    return " | ".join(previews)


def build_fire_indexes(fire):
    fire = fire.copy()
    fire["name_variants"] = fire["대상물명"].map(name_variants)
    fire["addr_keys"] = fire.apply(
        lambda row: {
            key
            for key in [
                normalize_address(row["지번주소_매칭"]),
                normalize_address(row["도로명주소_매칭"]),
            ]
            if key
        },
        axis=1,
    )

    name_index = {}
    addr_index = {}
    for idx, row in fire.iterrows():
        for variant in row["name_variants"]:
            name_index.setdefault(variant, set()).add(idx)
        for addr_key in row["addr_keys"]:
            addr_index.setdefault(addr_key, set()).add(idx)
    return fire, name_index, addr_index


def collect_name_hits(name_index, variants):
    hits = set()
    for variant in variants:
        hits.update(name_index.get(variant, set()))
    return hits


def collect_addr_hits(addr_index, addr_keys):
    hits = set()
    for addr_key in addr_keys:
        hits.update(addr_index.get(addr_key, set()))
    return hits


def fill_match(features, idx, matched, status):
    features.at[idx, "소방청_매칭상태"] = status
    features.at[idx, "소방청_스프링클러(AV)설치여부"] = clean_text(
        matched["스프링클러(AV)설치여부"]
    )
    features.at[idx, "소방청_간이스프링클러(AV)설치여부"] = clean_text(
        matched["간이스프링클러(AV)설치여부"]
    )
    features.at[idx, "소방청_대상물명"] = clean_text(matched["대상물명"])
    features.at[idx, "소방청_지번주소_매칭"] = clean_text(matched["지번주소_매칭"])
    features.at[idx, "소방청_도로명주소_매칭"] = clean_text(matched["도로명주소_매칭"])
    features.at[idx, "소방청_매칭방식"] = clean_text(matched["매칭방식"])
    features.at[idx, "소방청_매칭점수"] = clean_text(matched["매칭점수"])


def main():
    features = pd.read_csv(FEATURE_PATH, encoding="utf-8-sig", low_memory=False)
    fire = read_fire_csv(FIRE_PATH)
    fire, name_index, addr_index = build_fire_indexes(fire)

    features["소방청_매칭상태"] = ""
    features["소방청_스프링클러(AV)설치여부"] = ""
    features["소방청_간이스프링클러(AV)설치여부"] = ""
    features["소방청_대상물명"] = ""
    features["소방청_지번주소_매칭"] = ""
    features["소방청_도로명주소_매칭"] = ""
    features["소방청_매칭방식"] = ""
    features["소방청_매칭점수"] = ""

    name_only_rows = []
    addr_only_rows = []
    split_rows = []
    ambiguous_rows = []

    for idx, feature_row in features.iterrows():
        business_name = clean_text(feature_row["사업장명"])
        business_variants = name_variants(business_name)
        feature_addr_keys = {
            key
            for key in [
                normalize_address(feature_row["대지위치"]),
                normalize_address(feature_row["도로명대지위치"]),
            ]
            if key
        }

        name_hits = collect_name_hits(name_index, business_variants)
        addr_hits = collect_addr_hits(addr_index, feature_addr_keys)
        both_hits = name_hits & addr_hits

        if len(both_hits) == 1:
            fire_idx = next(iter(both_hits))
            matched = fire.loc[fire_idx]
            fill_match(features, idx, matched, "이름+주소일치")
            continue

        if len(both_hits) > 1:
            features.at[idx, "소방청_매칭상태"] = "복수후보"
            ambiguous_rows.append(
                {
                    **feature_row.to_dict(),
                    "소방청_매칭상태": "복수후보",
                    "후보개수": len(both_hits),
                    "후보미리보기": candidate_preview(fire, both_hits),
                }
            )
            continue

        if name_hits and addr_hits:
            features.at[idx, "소방청_매칭상태"] = "이름주소각자일치"
            split_rows.append(
                {
                    **feature_row.to_dict(),
                    "소방청_매칭상태": "이름주소각자일치",
                    "이름후보개수": len(name_hits),
                    "주소후보개수": len(addr_hits),
                    "이름후보미리보기": candidate_preview(fire, name_hits),
                    "주소후보미리보기": candidate_preview(fire, addr_hits),
                }
            )
            continue

        if name_hits:
            features.at[idx, "소방청_매칭상태"] = "이름만일치"
            name_only_rows.append(
                {
                    **feature_row.to_dict(),
                    "소방청_매칭상태": "이름만일치",
                    "후보개수": len(name_hits),
                    "후보미리보기": candidate_preview(fire, name_hits),
                }
            )
            continue

        if len(addr_hits) == 1:
            fire_idx = next(iter(addr_hits))
            matched = fire.loc[fire_idx]
            fill_match(features, idx, matched, "주소만일치")
            continue

        if addr_hits:
            features.at[idx, "소방청_매칭상태"] = "주소만일치_복수후보"
            addr_only_rows.append(
                {
                    **feature_row.to_dict(),
                    "소방청_매칭상태": "주소만일치_복수후보",
                    "후보개수": len(addr_hits),
                    "후보미리보기": candidate_preview(fire, addr_hits),
                }
            )
            continue

        features.at[idx, "소방청_매칭상태"] = "불일치"

    features.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    pd.DataFrame(name_only_rows).to_csv(NAME_ONLY_PATH, index=False, encoding="utf-8-sig")
    pd.DataFrame(addr_only_rows).to_csv(ADDR_ONLY_PATH, index=False, encoding="utf-8-sig")
    pd.DataFrame(split_rows).to_csv(SPLIT_PATH, index=False, encoding="utf-8-sig")
    pd.DataFrame(ambiguous_rows).to_csv(AMBIGUOUS_PATH, index=False, encoding="utf-8-sig")

    status_counts = features["소방청_매칭상태"].value_counts().to_dict()
    print(f"feature_rows={len(features)}")
    for key in [
        "이름+주소일치",
        "이름만일치",
        "주소만일치",
        "주소만일치_복수후보",
        "이름주소각자일치",
        "복수후보",
        "불일치",
    ]:
        print(f"{key}={status_counts.get(key, 0)}")
    print(f"output={OUTPUT_PATH}")
    print(f"name_only={NAME_ONLY_PATH}")
    print(f"addr_only={ADDR_ONLY_PATH}")
    print(f"split_match={SPLIT_PATH}")
    print(f"ambiguous={AMBIGUOUS_PATH}")


if __name__ == "__main__":
    main()
