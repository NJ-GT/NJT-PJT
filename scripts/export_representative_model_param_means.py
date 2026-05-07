# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE = Path(__file__).resolve().parents[1]
MGWR_PATH = BASE / "data" / "mgwr_local_params_low_high.csv"
GWR_PATH = BASE / "0424" / "data" / "cluster3_spatial_pipeline_fire_count_150m_0428" / "gwr_local_diagnostics_by_cluster.csv"
OUT_PATH = BASE / "0430" / "군집별_대표모형_params_평균계수.csv"

RISK_LABELS = {0: "저위험군", 1: "중위험군", 2: "고위험군"}
FEATURE_ORDER = [
    "intercept",
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
DISPLAY_ORDER = ["Intercept" if f == "intercept" else f for f in FEATURE_ORDER]


def append_param_rows(rows: list[dict], df: pd.DataFrame, cluster_id: int, model_name: str) -> None:
    sub = df[df["cluster"].astype(int).eq(cluster_id)].copy()
    for feature in FEATURE_ORDER:
        col = f"coef_{feature}"
        if col not in sub.columns:
            continue
        vals = pd.to_numeric(sub[col], errors="coerce").dropna()
        if vals.empty:
            continue
        rows.append(
            {
                "위험군": RISK_LABELS[cluster_id],
                "cluster": cluster_id,
                "대표모형": model_name,
                "변수": "Intercept" if feature == "intercept" else feature,
                "n_params": int(vals.shape[0]),
                "coef_mean": float(vals.mean()),
                "coef_abs_mean": float(vals.abs().mean()),
                "coef_std": float(vals.std(ddof=1)),
                "coef_min": float(vals.min()),
                "coef_max": float(vals.max()),
                "positive_ratio": float((vals > 0).mean()),
            }
        )


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    mgwr = pd.read_csv(MGWR_PATH, encoding="utf-8-sig")
    append_param_rows(rows, mgwr, 0, "MGWR")
    append_param_rows(rows, mgwr, 2, "MGWR")

    gwr = pd.read_csv(GWR_PATH, encoding="utf-8-sig")
    append_param_rows(rows, gwr, 1, "GWR")

    result = pd.DataFrame(rows)
    result["변수"] = pd.Categorical(result["변수"], categories=DISPLAY_ORDER, ordered=True)
    result = result.sort_values(["cluster", "변수"]).reset_index(drop=True)
    result["변수"] = result["변수"].astype(str)
    result.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    print(f"saved={OUT_PATH}")
    for label in ["저위험군", "중위험군", "고위험군"]:
        sub = result[result["위험군"].eq(label)]
        print(f"\n[{label}]")
        print(
            sub[
                [
                    "대표모형",
                    "변수",
                    "n_params",
                    "coef_mean",
                    "coef_abs_mean",
                    "coef_std",
                    "positive_ratio",
                ]
            ].to_string(index=False, float_format=lambda x: f"{x:.6f}")
        )


if __name__ == "__main__":
    main()
