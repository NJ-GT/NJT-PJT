# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from mgwr.gwr import MGWR
from mgwr.sel_bw import Sel_BW
from sklearn.preprocessing import StandardScaler


BASE = Path(__file__).resolve().parents[1]
PIPELINE_DIR = BASE / "0424" / "data" / "cluster3_spatial_pipeline_fire_count_150m_0428"
INPUT_SOURCE = PIPELINE_DIR / "최최최종0428변수테이블.csv"
METADATA_SOURCE = PIPELINE_DIR / "metadata.json"
OUTPUT_SOURCE = BASE / "data" / "mgwr_local_params_all.csv"
RISK_LABELS = {0: "저위험군", 1: "중위험군", 2: "고위험군"}
RANDOM_SEED = 42
SAMPLE_CAP = None


def read_metadata() -> tuple[str, list[str]]:
    metadata = json.loads(METADATA_SOURCE.read_text(encoding="utf-8"))
    return metadata["target"], metadata["features"]


def as_bandwidth_list(raw_bw, expected_len: int) -> list[float]:
    if isinstance(raw_bw, tuple):
        raw_bw = raw_bw[0]
    arr = np.asarray(raw_bw, dtype=float).reshape(-1)
    if len(arr) != expected_len:
        return [float("nan")] * expected_len
    return arr.tolist()


def fit_group(df: pd.DataFrame, cluster_id: int, target: str, features: list[str]) -> pd.DataFrame:
    group = (
        df[df["cluster"] == cluster_id]
        .dropna(subset=["x_5181", "y_5181", target, *features])
        .copy()
    )
    if group.empty:
        return pd.DataFrame()

    if SAMPLE_CAP is None:
        sample = group.reset_index(drop=True)
    else:
        sample_n = min(SAMPLE_CAP, len(group))
        sample = group.sample(n=sample_n, random_state=RANDOM_SEED).reset_index(drop=True)
    coords = sample[["x_5181", "y_5181"]].astype(float).to_numpy()
    y = sample[target].astype(float).to_numpy().reshape((-1, 1))
    scaler = StandardScaler()
    x = scaler.fit_transform(sample[features].astype(float).to_numpy())

    selector = Sel_BW(coords, y, x, multi=True, kernel="bisquare", fixed=False, n_jobs=1)
    selector.search(verbose=False)
    result = MGWR(coords, y, x, selector, kernel="bisquare", fixed=False, n_jobs=1).fit()
    bandwidths = as_bandwidth_list(selector.bw, len(features) + 1)

    out = sample[["구", "동", "숙소명", "위도", "경도", "x_5181", "y_5181", target, *features]].copy()
    out.insert(0, "cluster", cluster_id)
    out.insert(1, "cluster_label", RISK_LABELS[cluster_id])
    try:
        out["local_R2"] = np.asarray(result.localR2).reshape(-1)
    except NotImplementedError:
        out["local_R2"] = np.nan
    out["residual"] = np.asarray(result.resid_response).reshape(-1)

    terms = ["intercept", *features]
    for i, term in enumerate(terms):
        out[f"coef_{term}"] = result.params[:, i]
        if hasattr(result, "tvalues"):
            out[f"tval_{term}"] = result.tvalues[:, i]
        out[f"bw_{term}"] = bandwidths[i] if i < len(bandwidths) else np.nan

    # Spatial contribution varies even when MGWR coefficients are nearly global.
    # It answers: "how much did this variable contribute at this local point?"
    for i, feature in enumerate(features):
        out[f"z_{feature}"] = x[:, i]
        out[f"contrib_{feature}"] = out[f"coef_{feature}"] * out[f"z_{feature}"]
    return out


def main() -> None:
    target, features = read_metadata()
    df = pd.read_csv(INPUT_SOURCE, encoding="utf-8-sig")
    if target not in df.columns and "최종_화재위험점수" in df.columns:
        target = "최종_화재위험점수"
    for col in ["cluster", "x_5181", "y_5181", "위도", "경도", target, *features]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    outputs = []
    for cluster_id in RISK_LABELS:
        print(f"fitting MGWR: cluster={cluster_id} ({RISK_LABELS[cluster_id]})")
        outputs.append(fit_group(df, cluster_id, target, features))

    result = pd.concat([part for part in outputs if not part.empty], ignore_index=True)
    OUTPUT_SOURCE.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_SOURCE, index=False, encoding="utf-8-sig")
    print(f"saved {OUTPUT_SOURCE} rows={len(result):,}")


if __name__ == "__main__":
    main()
