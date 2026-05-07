# -*- coding: utf-8 -*-
"""Keep only the final six variables in full GWR/MGWR result CSV files.

Run this on the machine that still has the original full result CSVs:

    cd NJT-PJT
    python scripts/filter_full_results_to_6vars.py

It overwrites:
    data/full_gwr_mgwr/gwr_results_full.csv
    data/full_gwr_mgwr/mgwr_results_full.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE = Path(__file__).resolve().parents[1]
RESULT_DIR = BASE / "data" / "full_gwr_mgwr"
FILES = [
    RESULT_DIR / "gwr_results_full.csv",
    RESULT_DIR / "mgwr_results_full.csv",
]

KEEP_VARIABLES = {
    "구조노후도",
    "단속위험도",
    "도로폭위험도",
    "최근접_소화용수_거리등급",
    "소방위험도_점수",
    "집중도",
}

DROP_VARIABLES = {
    "승인연도",
    "연면적",
    "주변건물수",
    "총층수",
}

BASE_COLUMNS = {
    "구",
    "동",
    "숙소명",
    "위도",
    "경도",
    "x_5181",
    "y_5181",
    "최종위험점수_new",
    "local_R2",
    "bandwidth",
    "residual",
    "coef_intercept",
    "tval_intercept",
    "bw_intercept",
}

VARIABLE_PREFIXES = ("coef_", "tval_", "bw_", "z_", "contrib_")


def variable_name(column: str) -> str | None:
    for prefix in VARIABLE_PREFIXES:
        if column.startswith(prefix):
            return column.removeprefix(prefix)
    return None


def keep_column(column: str) -> bool:
    if column in BASE_COLUMNS:
        return True
    var = variable_name(column)
    if var is None:
        return False
    if var in DROP_VARIABLES:
        return False
    return var in KEEP_VARIABLES


def main() -> None:
    for path in FILES:
        df = pd.read_csv(path, encoding="utf-8-sig")
        keep_cols = [col for col in df.columns if keep_column(col)]
        if not any(col.startswith("coef_") for col in keep_cols):
            raise RuntimeError(
                f"{path} does not contain model coefficient columns. "
                "Use the original full result CSV, then run this script again."
            )
        filtered = df[keep_cols]
        filtered.to_csv(path, index=False, encoding="utf-8-sig")
        print(
            f"{path}: {len(df.columns)} -> {len(filtered.columns)} columns, rows={len(filtered):,}"
        )
        print(filtered.columns.tolist())


if __name__ == "__main__":
    main()
