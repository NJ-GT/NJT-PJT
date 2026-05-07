# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "0430" / "최종테이블0429.csv"


def main() -> None:
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()

    target_dongs = ["연남동", "서교동"]
    subset = df[df["구"].eq("마포구") & df["동"].isin(target_dongs)].copy()

    cols = [
        "최종위험점수_new",
        "구조노후도",
        "단속위험도",
        "도로폭위험도",
        "최근접_소화용수_거리등급",
        "소방위험도_점수",
        "승인연도",
        "연면적",
        "집중도",
        "주변건물수",
        "총층수",
    ]
    for col in cols:
        subset[col] = pd.to_numeric(subset[col], errors="coerce")

    print(f"rows: {len(subset):,}")
    print("\n[동별 건수 / 위험군 구성]")
    print(pd.crosstab(subset["동"], subset["cluster_label"]).to_string())

    print("\n[동별 주요 변수 describe]")
    for dong in target_dongs:
        part = subset[subset["동"].eq(dong)]
        print(f"\n=== {dong} n={len(part):,} ===")
        desc = part[cols].describe().T[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]
        print(desc.round(4).to_string())

    print("\n[동별 평균 요약]")
    print(subset.groupby("동")[cols].mean().round(4).reindex(target_dongs).to_string())


if __name__ == "__main__":
    main()
