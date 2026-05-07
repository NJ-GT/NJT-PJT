# -*- coding: utf-8 -*-
"""
[파일 설명]
강남구 등기부등본 표제부와 통합숙박시설 데이터를 관리건축물대장PK로 매칭하여
강남구 기준 숙박시설 피처를 생성하는 스크립트.

주요 역할:
  1. 등기부등본 표제부(강남)와 통합숙박시설 데이터를 PK로 조인한다.
  2. 같은 PK에 여러 사업장이 있는 경우 이름, 주소 등을 '|' 구분자로 합쳐 하나의 행으로 정리한다.
  3. 매칭 여부(has_hospitality_match)를 True/False 플래그로 추가한다.

입력: 등기부등본_표제부_강남.csv      (강남구 건물 등기 정보)
      통합숙박시설표제부0414.csv       (숙박시설 매칭 데이터)
출력: data/강남표제부기준_통합숙박시설피처0414.csv
"""

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent  # 프로젝트 루트 경로
REGISTRY_PATH = BASE_DIR / "등기부등본_표제부_강남.csv"       # 강남구 건물 등기부등본
HOSPITALITY_PATH = BASE_DIR / "통합숙박시설표제부0414.csv"    # 숙박시설 표제부
OUTPUT_PATH = BASE_DIR / "data" / "강남표제부기준_통합숙박시설피처0414.csv"  # 출력 파일


def normalize_pk(series):
    """
    PK 컬럼을 정규화한다. pandas가 float으로 읽어 '1234.0' 형태가 되는 것을 방지.
    예: '1234.0' → '1234'
    """
    return (
        series.fillna("")
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)  # 소수점 .0 제거
        .str.strip()
    )


def collapse_unique(values):
    """
    그룹 내 여러 값을 중복 없이 '|' 구분자로 합쳐 하나의 문자열로 반환한다.
    예: ['호텔A', '호텔A', '게스트하우스B'] → '호텔A | 게스트하우스B'
    """
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
