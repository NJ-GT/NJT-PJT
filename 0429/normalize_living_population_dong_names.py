from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "새 폴더" / "날짜별_concat"
ROW_INPUT = DATA_DIR / "생활인구수_한글컬럼_방문4시간그룹_동추정.csv"
SUMMARY_INPUT = DATA_DIR / "동별_25개월평균_방문생활인구수_빠른추정.csv"
ROW_OUTPUT = DATA_DIR / "생활인구수_한글컬럼_방문4시간그룹_대표동명.csv"
SUMMARY_OUTPUT = DATA_DIR / "대표동별_25개월평균_방문생활인구수.csv"

TIME_COLS = [
    "방문생활인구수_00_03시",
    "방문생활인구수_04_07시",
    "방문생활인구수_08_11시",
    "방문생활인구수_12_15시",
    "방문생활인구수_16_19시",
    "방문생활인구수_20_23시",
]

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
    "을지로1가": "을지로",
    "을지로2가": "을지로",
    "을지로3가": "을지로",
    "을지로4가": "을지로",
    "을지로5가": "을지로",
    "을지로6가": "을지로",
    "을지로7가": "을지로",
}


def add_representative_dong(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["대표_동"] = out["추정_동"].replace(NORMALIZE_DONG)
    return out


def main() -> None:
    row_df = pd.read_csv(ROW_INPUT, encoding="utf-8-sig", dtype=str)
    row_df = add_representative_dong(row_df)
    row_df.to_csv(ROW_OUTPUT, index=False, encoding="utf-8-sig")

    for col in TIME_COLS:
        row_df[col] = pd.to_numeric(row_df[col], errors="coerce").fillna(0)

    monthly = (
        row_df.groupby(["파일기준년월", "추정_구", "대표_동"], as_index=False)[TIME_COLS]
        .sum()
    )
    summary = monthly.groupby(["추정_구", "대표_동"], as_index=False)[TIME_COLS].mean()
    summary["25개월평균_방문생활인구수"] = summary[TIME_COLS].sum(axis=1)
    summary = summary.sort_values("25개월평균_방문생활인구수", ascending=False)
    summary.to_csv(SUMMARY_OUTPUT, index=False, encoding="utf-8-sig")

    print(ROW_OUTPUT)
    print(SUMMARY_OUTPUT)
    print(summary.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
