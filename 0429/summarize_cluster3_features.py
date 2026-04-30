from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR.parent / "0424" / "data" / "cluster3_spatial_pipeline_fire_count_150m_0428" / "최최최종0428변수테이블.csv"
OUTPUT_SUMMARY = BASE_DIR / "cluster3_feature_summary.csv"
OUTPUT_TOP_AREAS = BASE_DIR / "cluster3_top_areas_for_profile.csv"

FEATURES = [
    "최종_화재위험점수",
    "소방위험도_점수",
    "주변건물수",
    "집중도",
    "단속위험도",
    "구조노후도",
    "도로폭위험도",
    "최근접_소화용수_거리등급",
    "승인연도",
    "총층수",
    "연면적",
]


def main() -> None:
    df = pd.read_csv(INPUT_PATH, encoding="utf-8-sig")
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    summary = df.groupby("cluster")[FEATURES].agg(["count", "mean", "median", "std"])
    summary.columns = [f"{col}_{stat}" for col, stat in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(OUTPUT_SUMMARY, index=False, encoding="utf-8-sig")

    area_rows = []
    for cluster, sub in df.groupby("cluster"):
        top_dongs = (
            sub.groupby(["구", "동"])
            .size()
            .reset_index(name="시설수")
            .sort_values("시설수", ascending=False)
            .head(10)
        )
        for rank, row in enumerate(top_dongs.itertuples(index=False), start=1):
            area_rows.append({"cluster": cluster, "rank": rank, "구": row.구, "동": row.동, "시설수": row.시설수})
    pd.DataFrame(area_rows).to_csv(OUTPUT_TOP_AREAS, index=False, encoding="utf-8-sig")

    mean = df.groupby("cluster")[FEATURES].mean()
    overall = df[FEATURES].mean()
    rel = (mean - overall) / overall.abs() * 100

    print("cluster counts")
    print(df["cluster"].value_counts().sort_index().to_string())
    print("\nmeans")
    print(mean.round(3).to_string())
    print("\nrelative to overall (%)")
    print(rel.round(1).to_string())
    print("\ntop areas")
    print(pd.DataFrame(area_rows).head(30).to_string(index=False))


if __name__ == "__main__":
    main()
