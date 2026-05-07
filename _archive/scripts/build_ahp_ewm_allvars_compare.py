# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr


BASE = Path(__file__).resolve().parents[1]
SRC = BASE / "0424" / "data" / "분석변수_최종테이블0428.csv"
OUT_DIR = BASE / "0424" / "data"

# 위도/경도는 위치 식별값이라 제외. 승인연도는 오래될수록 위험하므로 건물나이로 변환한다.
FEATURES = [
    "소방위험도_점수",
    "도로폭위험도",
    "구조노후도",
    "단속위험도",
    "주변건물수",
    "집중도",
    "건물나이_2026",
    "총층수",
    "연면적",
]

# 발표/실무용 판단 기준. 같은 군은 동등, 한 단계 차이는 3, 두 단계 차이는 5,
# 세 단계 차이는 7, 네 단계 이상은 9로 둔다.
PRIORITY = {
    "소방위험도_점수": 5,
    "도로폭위험도": 4,
    "구조노후도": 4,
    "단속위험도": 3,
    "주변건물수": 3,
    "집중도": 3,
    "건물나이_2026": 2,
    "총층수": 1,
    "연면적": 1,
}

SAATY_BY_DIFF = {
    0: 1,
    1: 3,
    2: 5,
    3: 7,
    4: 9,
}

RI = {
    1: 0.00,
    2: 0.00,
    3: 0.58,
    4: 0.90,
    5: 1.12,
    6: 1.24,
    7: 1.32,
    8: 1.41,
    9: 1.45,
    10: 1.49,
}


def pair_value(left: str, right: str) -> float:
    diff = PRIORITY[left] - PRIORITY[right]
    scale = SAATY_BY_DIFF[min(abs(diff), 4)]
    if diff >= 0:
        return float(scale)
    return 1.0 / float(scale)


def build_pairwise_matrix() -> pd.DataFrame:
    matrix = pd.DataFrame(index=FEATURES, columns=FEATURES, dtype=float)
    for left in FEATURES:
        for right in FEATURES:
            matrix.loc[left, right] = pair_value(left, right)
    return matrix


def ahp_weights(matrix: pd.DataFrame) -> tuple[pd.Series, float, float, float]:
    values = matrix.to_numpy(dtype=float)
    eigvals, eigvecs = np.linalg.eig(values)
    max_idx = int(np.argmax(eigvals.real))
    lambda_max = float(eigvals[max_idx].real)
    weights = eigvecs[:, max_idx].real
    weights = np.abs(weights)
    weights = weights / weights.sum()

    n = len(matrix)
    ci = (lambda_max - n) / (n - 1)
    cr = ci / RI[n] if RI[n] else 0.0
    return pd.Series(weights, index=matrix.index, name="AHP_weight"), lambda_max, ci, cr


def minmax_score(df: pd.DataFrame, weights: pd.Series, score_name: str) -> pd.Series:
    x = df[weights.index].apply(pd.to_numeric, errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan).fillna(x.median(numeric_only=True))
    z = (x - x.min()) / (x.max() - x.min()).replace(0, 1)
    return (z * weights).sum(axis=1).rename(score_name) * 100


def ewm_weights(df: pd.DataFrame) -> pd.Series:
    x = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan).fillna(x.median(numeric_only=True))
    z = (x - x.min()) / (x.max() - x.min()).replace(0, 1)
    p = z + 1e-12
    p = p.div(p.sum(axis=0), axis=1)
    entropy = -(1 / np.log(len(p))) * (p * np.log(p)).sum(axis=0)
    diversity = 1 - entropy
    return (diversity / diversity.sum()).rename("EWM_weight")


def main() -> None:
    df = pd.read_csv(SRC, encoding="utf-8-sig")
    df["건물나이_2026"] = 2026 - pd.to_numeric(df["승인연도"], errors="coerce")

    matrix = build_pairwise_matrix()
    weights, lambda_max, ci, cr = ahp_weights(matrix)
    ewm = ewm_weights(df)

    scored = df.copy()
    scored["건물나이_2026"] = df["건물나이_2026"]
    scored["AHP_전체변수"] = minmax_score(scored, weights, "AHP_전체변수")
    scored["EWM_전체변수"] = minmax_score(scored, ewm, "EWM_전체변수")
    scored["AHP순위"] = scored["AHP_전체변수"].rank(ascending=False, method="min").astype(int)
    scored["EWM순위"] = scored["EWM_전체변수"].rank(ascending=False, method="min").astype(int)
    scored["순위차_EWM-AHP"] = scored["EWM순위"] - scored["AHP순위"]

    comparison = pd.concat([weights, ewm], axis=1)
    comparison["weight_gap_EWM_minus_AHP"] = comparison["EWM_weight"] - comparison["AHP_weight"]
    comparison = comparison.sort_values("AHP_weight", ascending=False)

    rank_abs = scored["순위차_EWM-AHP"].abs()
    summary_rows = [
        ["n", len(scored)],
        ["pearson", float(scored["AHP_전체변수"].corr(scored["EWM_전체변수"], method="pearson"))],
        ["spearman", float(spearmanr(scored["AHP_전체변수"], scored["EWM_전체변수"]).correlation)],
        ["kendall", float(kendalltau(scored["AHP_전체변수"], scored["EWM_전체변수"]).statistic)],
        ["rank_abs_mean", float(rank_abs.mean())],
        ["rank_abs_median", float(rank_abs.median())],
        ["rank_abs_p90", float(rank_abs.quantile(0.9))],
        ["rank_abs_max", int(rank_abs.max())],
        ["lambda_max", lambda_max],
        ["CI", ci],
        ["CR", cr],
    ]
    for top_n in [30, 50, 100, 300]:
        top_a = set(scored.nsmallest(top_n, "AHP순위").index)
        top_e = set(scored.nsmallest(top_n, "EWM순위").index)
        summary_rows.append([f"top{top_n}_overlap_count", len(top_a & top_e)])
        summary_rows.append([f"top{top_n}_overlap_rate", len(top_a & top_e) / top_n])

    matrix.to_csv(OUT_DIR / "AHP_쌍대비교표_전체변수_0428.csv", encoding="utf-8-sig")
    comparison.to_csv(OUT_DIR / "AHP_EWM_가중치비교_전체변수_0428.csv", encoding="utf-8-sig")
    pd.DataFrame(summary_rows, columns=["metric", "value"]).to_csv(
        OUT_DIR / "AHP_EWM_차이요약_전체변수_0428.csv",
        index=False,
        encoding="utf-8-sig",
    )
    scored.to_csv(
        OUT_DIR / "분석변수_최종테이블0428_AHP_EWM_전체변수비교.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("AHP pairwise matrix saved")
    print(matrix.round(4).to_string())
    print("\nAHP/EWM weights")
    print(comparison.round(4).to_string())
    print("\nSummary")
    print(pd.DataFrame(summary_rows, columns=["metric", "value"]).to_string(index=False))


if __name__ == "__main__":
    main()
