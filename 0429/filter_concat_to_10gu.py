from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "새 폴더" / "날짜별_concat" / "ES1001AH00301MM_202403_202603_concat.csv"
OUTPUT_PATH = BASE_DIR / "새 폴더" / "날짜별_concat" / "ES1001AH00301MM_202403_202603_concat_10gu.csv"
SUMMARY_PATH = BASE_DIR / "새 폴더" / "날짜별_concat" / "ES1001AH00301MM_202403_202603_concat_10gu_summary.csv"

TARGET_GU = [
    "강남구",
    "강서구",
    "마포구",
    "서초구",
    "성동구",
    "송파구",
    "영등포구",
    "용산구",
    "종로구",
    "중구",
]


def main() -> None:
    df = pd.read_csv(INPUT_PATH, encoding="utf-8-sig", dtype=str)
    filtered = df[df["SIGNGU_NM"].isin(TARGET_GU)].copy()
    filtered.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    summary = (
        filtered.groupby(["source_ym", "SIGNGU_NM"], dropna=False)
        .size()
        .reset_index(name="rows")
        .sort_values(["source_ym", "SIGNGU_NM"])
    )
    summary.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")

    print(OUTPUT_PATH)
    print(SUMMARY_PATH)
    print(f"input_rows={len(df)} output_rows={len(filtered)}")
    print("gu_counts")
    print(filtered["SIGNGU_NM"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
