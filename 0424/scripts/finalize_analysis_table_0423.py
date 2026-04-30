from __future__ import annotations

from pathlib import Path

import pandas as pd

from build_building_feature_probe import build_probe


ROOT = Path(__file__).resolve().parents[2]
PROBE_PATH = ROOT / "0424" / "분석" / "tables" / "분석변수_최종테이블0423_건물특성_probe.csv"
OUTPUT_PATH = ROOT / "0424" / "data" / "분석변수_최종테이블0423.csv"


def prepare_output_table(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()

    base_cols = ["구", "동", "숙소명", "승인연도", "주변건물수", "집중도", "단속위험도", "구조노후도", "도로폭위험도", "위도", "경도", "업종"]
    for col in base_cols:
        out[col] = df[col]

    out["건물용도명"] = df["건물용도명_통합"].fillna("미상").astype(str)
    out["총층수"] = pd.to_numeric(df["총층수_0층만보정"], errors="coerce").fillna(1).round().astype(int)
    out["연면적"] = pd.to_numeric(df["연면적_통합"], errors="coerce").fillna(0.0)

    # Sanity-check columns that must be fully populated for downstream models.
    required_non_null = ["건물용도명", "총층수", "연면적"]
    for col in required_non_null:
        if out[col].isna().any():
            raise ValueError(f"Required column still has nulls: {col}")

    return out


def main() -> None:
    if not PROBE_PATH.exists():
        build_probe()

    probe_df = pd.read_csv(PROBE_PATH, encoding="utf-8-sig")
    final_df = prepare_output_table(probe_df)
    final_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print(f"Saved: {OUTPUT_PATH}")
    print(f"Rows: {len(final_df)}")
    print(f"Columns: {len(final_df.columns)}")
    print(
        {
            "건물용도명_nulls": int(final_df["건물용도명"].isna().sum()),
            "총층수_nulls": int(final_df["총층수"].isna().sum()),
            "연면적_nulls": int(final_df["연면적"].isna().sum()),
        }
    )


if __name__ == "__main__":
    main()
