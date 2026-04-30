from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "새 폴더" / "날짜별_concat"
TOP9_PATH = DATA_DIR / "상위9개동_시간대별_25개월평균_방문생활인구수.csv"
LODGING_PATH = BASE_DIR.parent / "0424" / "data" / "cluster3_spatial_pipeline_fire_count_150m_0428" / "최최최종0428변수테이블.csv"
OUTPUT_PATH = DATA_DIR / "상위9개동_숙박시설수_매칭.csv"
NORMALIZED_OUTPUT_PATH = DATA_DIR / "상위9개동_숙박시설수_대표동명매칭.csv"

NORMALIZE_DONG = {
    "명동1가": "명동",
    "명동2가": "명동",
    "인현동1가": "인현동",
    "인현동2가": "인현동",
    "충무로1가": "충무로",
    "충무로2가": "충무로",
    "충무로3가": "충무로",
    "충무로4가": "충무로",
    "충무로5가": "충무로",
    "남산동1가": "남산동",
    "남산동2가": "남산동",
    "남산동3가": "남산동",
}


def main() -> None:
    top9 = pd.read_csv(TOP9_PATH, encoding="utf-8-sig")
    lodging = pd.read_csv(LODGING_PATH, encoding="utf-8-sig")

    lodging_counts = (
        lodging.groupby(["구", "동"], dropna=False)
        .size()
        .reset_index(name="숙박시설수")
    )

    out = top9.merge(
        lodging_counts,
        left_on=["추정_구", "추정_동"],
        right_on=["구", "동"],
        how="left",
    )
    out["숙박시설수"] = out["숙박시설수"].fillna(0).astype(int)
    out["숙박시설_매칭여부"] = out["숙박시설수"].gt(0)
    out = out.drop(columns=["구", "동"], errors="ignore")
    out.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print(OUTPUT_PATH)
    print(out[["추정_구", "추정_동", "숙박시설수", "숙박시설_매칭여부", "25개월평균_방문생활인구수"]].to_string(index=False))

    top9_norm = top9.copy()
    lodging_norm = lodging.copy()
    top9_norm["대표_동"] = top9_norm["추정_동"].replace(NORMALIZE_DONG)
    lodging_norm["대표_동"] = lodging_norm["동"].replace(NORMALIZE_DONG)

    normalized_counts = (
        lodging_norm.groupby(["구", "대표_동"], dropna=False)
        .size()
        .reset_index(name="대표동명_숙박시설수")
    )
    normalized_out = top9_norm.merge(
        normalized_counts,
        left_on=["추정_구", "대표_동"],
        right_on=["구", "대표_동"],
        how="left",
    )
    normalized_out["대표동명_숙박시설수"] = normalized_out["대표동명_숙박시설수"].fillna(0).astype(int)
    normalized_out["대표동명_매칭여부"] = normalized_out["대표동명_숙박시설수"].gt(0)
    normalized_out = normalized_out.drop(columns=["구"], errors="ignore")
    normalized_out.to_csv(NORMALIZED_OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print()
    print(NORMALIZED_OUTPUT_PATH)
    print(normalized_out[["추정_구", "추정_동", "대표_동", "대표동명_숙박시설수", "대표동명_매칭여부"]].to_string(index=False))


if __name__ == "__main__":
    main()
