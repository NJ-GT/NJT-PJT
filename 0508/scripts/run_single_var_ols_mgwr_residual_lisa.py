# -*- coding: utf-8 -*-
"""
단일변수 OLS + 군집별 MGWR 잔차 LISA 비교 지도 (변수마다 1세트씩 PNG 생성).

목적:
    - 핵심 위험변수 3개(구조노후도/도로폭위험도/단속위험도)를
      각각 단독으로 사용했을 때, OLS(전역) vs MGWR(군집별 국지) 의
      잔차 공간자기상관 차이를 시각화.

처리:
    변수 v 마다:
        1) 전체 데이터에 대해 단일변수 OLS 적합
        2) cluster 단위 MGWR 적합 후 결과 합치기
        3) 두 모델의 동 평균 잔차에 대해 Queen 가중치 LISA
        4) 좌(OLS)/우(MGWR) 비교 지도 PNG 저장

출력:
    - data/single_var_residual_lisa/{변수}_ols_single_var_results.csv
    - data/single_var_residual_lisa/{변수}_mgwr_single_var_results.csv
    - data/single_var_residual_lisa/{변수}_*_residual_lisa_by_dong.csv
    - data/single_var_residual_lisa/{변수}_single_var_ols_mgwr_residual_lisa.png
    - data/single_var_residual_lisa/single_var_residual_lisa_summary.csv
    - data/single_var_residual_lisa/run_metadata.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from esda.moran import Moran, Moran_Local
from libpysal.weights import Queen
from mgwr.gwr import MGWR
from mgwr.sel_bw import Sel_BW
from shapely.validation import make_valid
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


# 경로
BASE = Path(__file__).resolve().parents[1]
TABLE_PATH = next((BASE / "0430").glob("*0429.csv"))
BOUNDARY_PATH = BASE / "data" / "seoul_legal_dong_age_buckets_joined_0415.geojson"
OUT_DIR = BASE / "data" / "single_var_residual_lisa"

# 종속변수 + 단일 분석 변수 풀
TARGET = "최종위험점수_new"
VARIABLES = ["구조노후도", "도로폭위험도", "단속위험도"]
COORDS = ["x_5181", "y_5181"]
ID_COLS = [
    "구",
    "동",
    "숙소명",
    "cluster",
    "cluster_label",
    "위도",
    "경도",
    *COORDS,
    TARGET,
]

# 자치구 코드(EMD_CD 앞 5자리) → 자치구명
GU_BY_CODE = {
    "11110": "종로구",
    "11140": "중구",
    "11170": "용산구",
    "11200": "성동구",
    "11215": "광진구",
    "11230": "동대문구",
    "11260": "중랑구",
    "11290": "성북구",
    "11305": "강북구",
    "11320": "도봉구",
    "11350": "노원구",
    "11380": "은평구",
    "11410": "서대문구",
    "11440": "마포구",
    "11470": "양천구",
    "11500": "강서구",
    "11530": "구로구",
    "11545": "금천구",
    "11560": "영등포구",
    "11590": "동작구",
    "11620": "관악구",
    "11650": "서초구",
    "11680": "강남구",
    "11710": "송파구",
    "11740": "강동구",
}

# LISA 색상/라벨
LISA_COLORS = {
    "HH": "#d7191c",
    "LL": "#2c7bb6",
    "HL": "#fdae61",
    "LH": "#abd9e9",
    "Not Sig": "#d9dee7",
    "No Data": "#f2f4f7",
}
LISA_LABELS = {
    "HH": "HH: 양의 잔차 집중",
    "LL": "LL: 음의 잔차 집중",
    "HL": "HL: 국지적 양의 잔차",
    "LH": "LH: 국지적 음의 잔차",
    "Not Sig": "유의하지 않음",
    "No Data": "자료 없음",
}


def load_frame() -> pd.DataFrame:
    """마스터 CSV 로드 + 필수 컬럼 검증 + 결측 제거."""
    df = pd.read_csv(TABLE_PATH, encoding="utf-8-sig")
    required = [*ID_COLS, *VARIABLES]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    work = df[required].copy()
    # 좌표/타깃/변수 모두 숫자형 강제
    for col in [*COORDS, TARGET, *VARIABLES]:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    return work.dropna(subset=[*COORDS, TARGET, *VARIABLES]).reset_index(drop=True)


def load_boundary() -> gpd.GeoDataFrame:
    """법정동 경계 GDF — 자치구/동명 매칭 + 5179 좌표계 + 무효 지오메트리 보정."""
    gdf = gpd.read_file(BOUNDARY_PATH)
    gdf["구_매칭"] = gdf["EMD_CD"].astype(str).str[:5].map(GU_BY_CODE)
    if "구" in gdf.columns:
        gdf["구_매칭"] = gdf["구"].fillna(gdf["구_매칭"])
    gdf["동_매칭"] = gdf.get("법정동명", gdf["EMD_KOR_NM"]).fillna(gdf["EMD_KOR_NM"])
    gdf = gdf.to_crs(epsg=5179)
    gdf["geometry"] = gdf.geometry.apply(make_valid).buffer(0)
    return gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()


def run_ols(df: pd.DataFrame, variable: str) -> tuple[pd.DataFrame, dict]:
    """단일변수 OLS — 표준화 후 잔차/계수/R² 반환."""
    x = StandardScaler().fit_transform(df[[variable]].astype(float).to_numpy())
    y = df[TARGET].astype(float).to_numpy()
    model = LinearRegression().fit(x, y)
    pred = model.predict(x)
    out = df[ID_COLS].copy()
    out["prediction"] = pred
    out["residual"] = y - pred
    out[f"coef_{variable}"] = float(model.coef_[0])
    return out, {
        "model": "OLS",
        "variable": variable,
        "rows": int(len(out)),
        "R2": float(model.score(x, y)),
    }


def standardize_group(group: pd.DataFrame, variable: str):
    """MGWR 입력 셋 — 좌표/y(=(n,1))/표준화 X(=(n,1))."""
    coords = group[COORDS].astype(float).to_numpy()
    y = group[TARGET].astype(float).to_numpy().reshape((-1, 1))
    x = StandardScaler().fit_transform(group[[variable]].astype(float).to_numpy())
    return coords, y, x


def as_bandwidth_list(raw_bw, expected_len: int) -> list[float]:
    """Sel_BW.bw 의 다양한 반환을 list[float] 로 통일 — 길이 다르면 NaN 패딩."""
    if isinstance(raw_bw, tuple):
        raw_bw = raw_bw[0]
    arr = np.asarray(raw_bw, dtype=float).reshape(-1)
    if len(arr) != expected_len:
        return [float("nan")] * expected_len
    return arr.tolist()


def fit_mgwr_group(
    group: pd.DataFrame, variable: str, cluster_id, n_jobs: int
) -> tuple[pd.DataFrame, dict]:
    """단일 cluster 그룹에 대해 MGWR 적합 — 절편 + 변수 1개."""
    coords, y, x = standardize_group(group, variable)
    label = (
        group["cluster_label"].dropna().iloc[0]
        if group["cluster_label"].notna().any()
        else str(cluster_id)
    )
    print(f"[MGWR-1VAR] {variable} cluster={cluster_id} {label}, rows={len(group):,}")
    # multi=True — 절편/변수 BW 각각 탐색
    t0 = time.time()
    selector = Sel_BW(
        coords, y, x, multi=True, kernel="bisquare", fixed=False, n_jobs=n_jobs
    )
    selector.search(verbose=True)
    bandwidths = as_bandwidth_list(selector.bw, 2)  # 절편 + 1변수 = 2개
    print(
        f"[MGWR-1VAR] {variable} cluster={cluster_id} BW={bandwidths} search={time.time() - t0:.1f}s"
    )

    # 본 적합
    t0 = time.time()
    result = MGWR(
        coords, y, x, selector, kernel="bisquare", fixed=False, n_jobs=n_jobs
    ).fit()
    print(
        f"[MGWR-1VAR] {variable} cluster={cluster_id} fit={time.time() - t0:.1f}s R2={result.R2:.4f}"
    )

    out = group[ID_COLS].copy()
    out["residual"] = np.asarray(result.resid_response).reshape(-1)
    # 0번: 절편, 1번: 변수 — 행마다 다른 국지 계수
    out["coef_intercept"] = result.params[:, 0]
    out[f"coef_{variable}"] = result.params[:, 1]
    out["bw_intercept"] = bandwidths[0] if len(bandwidths) > 0 else np.nan
    out[f"bw_{variable}"] = bandwidths[1] if len(bandwidths) > 1 else np.nan
    return out, {
        "model": "MGWR",
        "variable": variable,
        "cluster": int(cluster_id),
        "cluster_label": str(label),
        "rows": int(len(out)),
        "bandwidths": bandwidths,
        "R2": float(result.R2),
        "adj_R2": float(result.adj_R2),
    }


def run_mgwr(
    df: pd.DataFrame, variable: str, n_jobs: int = 1
) -> tuple[pd.DataFrame, list[dict]]:
    """변수 1개에 대해 cluster 단위 MGWR 적합 후 결과를 한 DataFrame 으로 결합."""
    outputs = []
    metrics = []
    for cluster_id in sorted(df["cluster"].dropna().unique()):
        group = df[df["cluster"] == cluster_id].reset_index(drop=True)
        out, metric = fit_mgwr_group(group, variable, cluster_id, n_jobs)
        outputs.append(out)
        metrics.append(metric)
    return pd.concat(outputs, ignore_index=True), metrics


def lisa_category(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """LISA 4분면 + p값 → 라벨 — p≥0.05 면 'Not Sig'."""
    sig = p < 0.05
    cats = np.full(len(q), "Not Sig", dtype=object)
    cats[(q == 1) & sig] = "HH"
    cats[(q == 2) & sig] = "LH"
    cats[(q == 3) & sig] = "LL"
    cats[(q == 4) & sig] = "HL"
    return cats


def residual_lisa(
    boundary: gpd.GeoDataFrame, result: pd.DataFrame, model_name: str, variable: str
):
    """모델 잔차를 동 평균으로 집계 → Queen LISA → 결과 GDF/지표 반환."""
    # 동 단위 평균 잔차 + 표본 수
    dong = (
        result.groupby(["구", "동"], dropna=False)
        .agg(residual=("residual", "mean"), sample_count=("residual", "size"))
        .reset_index()
    )
    gdf = boundary.merge(
        dong, left_on=["구_매칭", "동_매칭"], right_on=["구", "동"], how="left"
    )
    data = gdf[gdf["residual"].notna()].copy().reset_index(drop=True)
    weights = Queen.from_dataframe(data, use_index=False)
    weights.transform = "r"
    y = data["residual"].to_numpy(dtype=float)
    moran = Moran(y, weights, permutations=999)
    local = Moran_Local(y, weights, permutations=999, seed=42)
    data["lisa_cat"] = lisa_category(local.p_sim, local.q)
    data["local_i"] = local.Is
    data["p_sim"] = local.p_sim
    data["quadrant"] = local.q

    # base GDF 에 LISA 컬럼 재결합 — '자료 없음'까지 포함
    out = gdf.merge(
        data[["구_매칭", "동_매칭", "lisa_cat", "local_i", "p_sim", "quadrant"]],
        on=["구_매칭", "동_매칭"],
        how="left",
    )
    out["lisa_cat"] = out["lisa_cat"].fillna("No Data")
    metrics = {
        "model": model_name,
        "variable": variable,
        "dong_count": int(len(data)),
        "global_moran_i": float(moran.I),
        "global_moran_p": float(moran.p_sim),
        "HH": int((data["lisa_cat"] == "HH").sum()),
        "LL": int((data["lisa_cat"] == "LL").sum()),
        "HL": int((data["lisa_cat"] == "HL").sum()),
        "LH": int((data["lisa_cat"] == "LH").sum()),
        "Not Sig": int((data["lisa_cat"] == "Not Sig").sum()),
    }
    return out, metrics


def plot_lisa(
    variable: str, model_maps: dict[str, gpd.GeoDataFrame], metrics: list[dict]
) -> Path:
    """변수 1개에 대한 OLS/MGWR LISA 비교 지도 PNG 저장."""
    matplotlib.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(20, 9.8), dpi=180)
    fig.patch.set_facecolor("#f7f9fc")
    metric_by_model = {m["model"]: m for m in metrics}
    # 자치구 외곽선 — 동 폴리곤 dissolve
    gu_boundary = next(iter(model_maps.values())).dissolve(
        by="구_매칭", as_index=False, method="unary", grid_size=0.05
    )

    for ax, (model_name, gdf) in zip(axes, model_maps.items()):
        ax.set_facecolor("#f7f9fc")
        gdf["plot_color"] = (
            gdf["lisa_cat"].map(LISA_COLORS).fillna(LISA_COLORS["No Data"])
        )
        gdf.plot(ax=ax, color=gdf["plot_color"], edgecolor="#c9d1dc", linewidth=0.18)
        gu_boundary.boundary.plot(ax=ax, color="#303744", linewidth=0.75, alpha=0.9)
        # 자치구 라벨
        for _, row in gu_boundary.iterrows():
            if row.geometry.is_empty:
                continue
            point = row.geometry.representative_point()
            ax.text(
                point.x,
                point.y,
                row["구_매칭"],
                ha="center",
                va="center",
                fontsize=5.8,
                weight="bold",
            )
        m = metric_by_model[model_name]
        ax.set_title(
            f"{model_name} {variable} 단일변수 잔차 LISA\nMoran's I={m['global_moran_i']:.3f}, p={m['global_moran_p']:.3f}",
            fontsize=16,
            weight="bold",
            loc="left",
        )
        ax.set_axis_off()

    handles = [
        mpatches.Patch(color=LISA_COLORS[key], label=LISA_LABELS[key])
        for key in ["HH", "LL", "HL", "LH", "Not Sig", "No Data"]
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=True,
        framealpha=0.96,
        fontsize=9,
    )
    fig.suptitle(
        f"{variable} 단일변수 OLS 잔차 LISA vs MGWR 잔차 LISA",
        fontsize=23,
        weight="bold",
        x=0.03,
        ha="left",
    )
    fig.text(
        0.03,
        0.04,
        f"사용 변수: {variable} 1개. HH는 이 변수 하나로 설명되지 않은 양의 잔차가 주변과 함께 몰린 지역.",
        fontsize=9,
        color="#667085",
    )
    out_path = OUT_DIR / f"{variable}_single_var_ols_mgwr_residual_lisa.png"
    fig.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


def main() -> None:
    """변수 3개 각각 OLS/MGWR/LISA 실행 후 비교지도와 메타 일괄 저장."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_frame()
    boundary = load_boundary()
    all_metrics = []

    print(f"[DATA] rows={len(df):,}, variables={VARIABLES}")
    for variable in VARIABLES:
        print(f"\n[VARIABLE] {variable}")
        # OLS / MGWR(군집별) 적합
        ols_out, ols_model_metric = run_ols(df, variable)
        mgwr_out, mgwr_model_metrics = run_mgwr(df, variable, n_jobs=1)

        # 행 단위 결과 CSV
        ols_out.to_csv(
            OUT_DIR / f"{variable}_ols_single_var_results.csv",
            index=False,
            encoding="utf-8-sig",
        )
        mgwr_out.to_csv(
            OUT_DIR / f"{variable}_mgwr_single_var_results.csv",
            index=False,
            encoding="utf-8-sig",
        )

        # LISA 계산 + 동 단위 CSV
        ols_map, ols_lisa = residual_lisa(boundary, ols_out, "OLS", variable)
        mgwr_map, mgwr_lisa = residual_lisa(boundary, mgwr_out, "MGWR", variable)
        ols_map.to_csv(
            OUT_DIR / f"{variable}_ols_single_var_residual_lisa_by_dong.csv",
            index=False,
            encoding="utf-8-sig",
        )
        mgwr_map.to_csv(
            OUT_DIR / f"{variable}_mgwr_single_var_residual_lisa_by_dong.csv",
            index=False,
            encoding="utf-8-sig",
        )

        path = plot_lisa(
            variable, {"OLS": ols_map, "MGWR": mgwr_map}, [ols_lisa, mgwr_lisa]
        )
        print(f"[SAVED] {path}")
        # 누적 — OLS 모형메트릭+OLS LISA, MGWR 군집별 메트릭, MGWR LISA
        all_metrics.extend(
            [{**ols_model_metric, **ols_lisa}, *mgwr_model_metrics, mgwr_lisa]
        )

    # 통합 요약 CSV/JSON
    pd.DataFrame(all_metrics).to_csv(
        OUT_DIR / "single_var_residual_lisa_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (OUT_DIR / "run_metadata.json").write_text(
        json.dumps(all_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("[DONE]")
    # 콘솔 요약 — LISA 지표(global_moran_i 가 있는 항목)만 표 출력
    print(
        pd.DataFrame([m for m in all_metrics if "global_moran_i" in m]).to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
