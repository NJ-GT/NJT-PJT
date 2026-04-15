# -*- coding: utf-8 -*-
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
REGISTRY_PATH = BASE_DIR / "등기부등본_표제부_강남.csv"
HOSPITALITY_PATH = BASE_DIR / "통합숙박시설표제부0414.csv"
OUTPUT_PATH = BASE_DIR / "data" / "강남표제부기준_통합숙박시설피처0414.csv"


def normalize_pk(series):
    return (
        series.fillna("")
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
    )


def collapse_unique(values):
    items = []
    seen = set()
    for value in values:
        text = "" if pd.isna(value) else str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        items.append(text)
    return " | ".join(items)


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    registry = pd.read_csv(REGISTRY_PATH, encoding="utf-8-sig", low_memory=False)
    hospitality = pd.read_csv(HOSPITALITY_PATH, encoding="utf-8-sig", low_memory=False)

    registry["registry_pk"] = normalize_pk(registry["관리건축물대장PK"])
    hospitality["registry_pk"] = normalize_pk(hospitality["selected_registry_pk"])

    matched_hospitality = hospitality[hospitality["registry_pk"].ne("")].copy()

    grouped = (
        matched_hospitality.groupby("registry_pk", dropna=False)
        .agg(
            hospitality_match_count=("registry_pk", "size"),
            hospitality_business_names=("사업장명", collapse_unique),
            hospitality_management_numbers=("관리번호", collapse_unique),
            hospitality_status_names=("영업상태명", collapse_unique),
            hospitality_jibun_addresses=("지번주소", collapse_unique),
            hospitality_road_addresses=("도로명주소", collapse_unique),
            hospitality_match_types=("match_type", collapse_unique),
            hospitality_selected_by=("selected_by", collapse_unique),
            hospitality_name_similarity_max=("selected_registry_name_similarity", "max"),
        )
        .reset_index()
    )

    result = registry.merge(grouped, on="registry_pk", how="left")
    result["has_hospitality_match"] = result["hospitality_match_count"].fillna(0).gt(0)

    result.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print(f"registry_rows={len(registry)}")
    print(f"matched_registry_rows={int(result['has_hospitality_match'].sum())}")
    print(f"hospitality_rows_used={len(matched_hospitality)}")
    print(f"output={OUTPUT_PATH}")


if __name__ == "__main__":
    main()
