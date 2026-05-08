# -*- coding: utf-8 -*-
"""
군집별 대표 공간모형(GWR/MGWR)의 변수 계수 평균을 한 표로 export.

규칙:
    - 저위험군(cluster 0): MGWR 대표
    - 중위험군(cluster 1): GWR 대표
    - 고위험군(cluster 2): MGWR 대표

각 변수에 대해 (n_params, coef_mean, coef_abs_mean, coef_std, min, max, positive_ratio) 산출.

산출:
    NJT-PJT/0430/군집별_대표모형_params_평균계수.csv
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


# scripts/ 기준 한 단계 위
BASE = Path(__file__).resolve().parents[1]
# MGWR 결과 (저/고위험군 사용)
MGWR_PATH = BASE / "data" / "mgwr_local_params_low_high.csv"
# GWR 결과 (중위험군 사용)
GWR_PATH = (
    BASE
    / "0424"
    / "data"
    / "cluster3_spatial_pipeline_fire_count_150m_0428"
    / "gwr_local_diagnostics_by_cluster.csv"
)
OUT_PATH = BASE / "0430" / "군집별_대표모형_params_평균계수.csv"

# 군집 라벨 매핑
RISK_LABELS = {0: "저위험군", 1: "중위험군", 2: "고위험군"}
# 표에 표시할 변수 순서 (intercept + 10변수)
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
# 표시용 — intercept를 'Intercept'로 변환
DISPLAY_ORDER = ["Intercept" if f == "intercept" else f for f in FEATURE_ORDER]


def append_param_rows(
    rows: list[dict], df: pd.DataFrame, cluster_id: int, model_name: str
) -> None:
    """주어진 cluster의 변수별 계수 통계량을 rows에 누적."""
    sub = df[df["cluster"].astype(int).eq(cluster_id)].copy()
    for feature in FEATURE_ORDER:
        col = f"coef_{feature}"
        # 입력 표에 컬럼이 없는 경우는 스킵
        if col not in sub.columns:
            continue
        # 결측 제거 후 통계량 산출
        vals = pd.to_numeric(sub[col], errors="coerce").dropna()
        if vals.empty:
            continue
        rows.append(
            {
                "위험군": RISK_LABELS[cluster_id],
                "cluster": cluster_id,
                "대표모형": model_name,
                # 표시용 라벨
                "변수": "Intercept" if feature == "intercept" else feature,
                "n_params": int(vals.shape[0]),
                "coef_mean": float(vals.mean()),
                "coef_abs_mean": float(vals.abs().mean()),
                # ddof=1: 표본 표준편차
                "coef_std": float(vals.std(ddof=1)),
                "coef_min": float(vals.min()),
                "coef_max": float(vals.max()),
                # 양수 비율 — 변수의 영향 방향성 안정성 점검
                "positive_ratio": float((vals > 0).mean()),
            }
        )


def main() -> None:
    """대표모형 매핑에 따라 군집별 결과를 모아 단일 CSV로 저장."""
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    # MGWR 결과 — 저위험(0)과 고위험(2)
    mgwr = pd.read_csv(MGWR_PATH, encoding="utf-8-sig")
    append_param_rows(rows, mgwr, 0, "MGWR")
    append_param_rows(rows, mgwr, 2, "MGWR")

    # GWR 결과 — 중위험(1)
    gwr = pd.read_csv(GWR_PATH, encoding="utf-8-sig")
    append_param_rows(rows, gwr, 1, "GWR")

    # 결과 DataFrame — 변수 순서를 카테고리로 강제
    result = pd.DataFrame(rows)
    result["변수"] = pd.Categorical(
        result["변수"], categories=DISPLAY_ORDER, ordered=True
    )
    result = result.sort_values(["cluster", "변수"]).reset_index(drop=True)
    # CSV 직전 카테고리는 다시 문자열로 (정렬 영향 X)
    result["변수"] = result["변수"].astype(str)
    result.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    # 콘솔 검증 출력 — 군집별 부분 표
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
