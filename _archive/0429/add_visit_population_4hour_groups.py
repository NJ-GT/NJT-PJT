from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "새 폴더" / "날짜별_concat" / "생활인구수_한글컬럼.csv"
OUTPUT_PATH = BASE_DIR / "새 폴더" / "날짜별_concat" / "생활인구수_한글컬럼_방문4시간그룹.csv"

GROUPS = {
    "방문생활인구수_00_03시": range(0, 4),
    "방문생활인구수_04_07시": range(4, 8),
    "방문생활인구수_08_11시": range(8, 12),
    "방문생활인구수_12_15시": range(12, 16),
    "방문생활인구수_16_19시": range(16, 20),
    "방문생활인구수_20_23시": range(20, 24),
}


def main() -> None:
    df = pd.read_csv(INPUT_PATH, encoding="utf-8-sig", dtype=str)

    for new_col, hours in GROUPS.items():
        hourly_cols = [f"{hour:02d}시_외국인_방문생활인구수" for hour in hours]
        missing = [col for col in hourly_cols if col not in df.columns]
        if missing:
            raise KeyError(f"Missing columns for {new_col}: {missing}")
        values = df[hourly_cols].apply(pd.to_numeric, errors="coerce")
        df[new_col] = values.sum(axis=1, min_count=1).round(4)

    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(OUTPUT_PATH)
    print(f"rows={len(df)} columns={len(df.columns)}")
    print(df[list(GROUPS)].head().to_string(index=False))


if __name__ == "__main__":
    main()
