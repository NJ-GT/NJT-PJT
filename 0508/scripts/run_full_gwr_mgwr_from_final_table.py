# -*- coding: utf-8 -*-
"""
0430/최종테이블0429.csv 의 모든 유효 행으로 GWR 전체 + 군집별 MGWR 적합 후 변수별 비교 지도 출력.

설계 메모:
    - GWR 은 전 행에 한 번 적합 (전역 최적 BW 1개로 전 변수 공유).
    - MGWR 은 변수마다 다른 BW 를 추정해 비용이 크므로,
      안정성/속도를 위해 cluster (저/중/고 위험군) 단위로 분할 적합 후 결합.

사용:
    cd NJT-PJT
    python scripts/run_full_gwr_mgwr_from_final_table.py [옵션]

출력:
    data/full_gwr_mgwr/gwr_results_full.csv
    data/full_gwr_mgwr/mgwr_results_full.csv
    data/full_gwr_mgwr/run_metadata.json
    data/full_gwr_mgwr/maps/*.png   (변수별 GWR vs MGWR 상대강도 비교)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import geopandas as gpd
import matplotlib

# 헤드리스 PNG 저장
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from mgwr.gwr import GWR, MGWR
from mgwr.sel_bw import Sel_BW
from shapely.validation import make_valid
from sklearn.preprocessing import StandardScaler


# 기본 경로
BASE = Path(__file__).resolve().parents[1]
DEFAULT_TABLE = BASE / "0430" / "최종테이블0429.csv"
DEFAULT_BOUNDARY = BASE / "data" / "seoul_legal_dong_age_buckets_joined_0415.geojson"
DEFAULT_OUT = BASE / "data" / "full_gwr_mgwr"

# 종속변수 + 좌표
TARGET = "최종위험점수_new"
COORDS = ["x_5181", "y_5181"]   # GWR/MGWR 거리 계산용 평면 좌표
LAT_LON = ["위도", "경도"]      # 결과 CSV 보존용
# 6개 핵심 위험 변수
DEFAULT_FEATURES = [
    "구조노후도",
    "단속위험도",
    "도로폭위험도",
    "최근접_소화용수_거리등급",
    "소방위험도_점수",
    "집중도",
]

# 분석 대상 10개 자치구 — 법정동 코드 앞 5자리 매핑
GU_BY_CODE = {
    "11110": "종로구",
    "11140": "중구",
    "11170": "용산구",
    "11200": "성동구",
    "11440": "마포구",
    "11500": "강서구",
    "11560": "영등포구",
    "11650": "서초구",
    "11680": "강남구",
    "11710": "송파구",
}


def parse_args() -> argparse.Namespace:
    """CLI 인자 — 입력/출력 경로, 변수, 커널, GWR/MGWR 스킵, 병렬 작업 수."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--boundary", type=Path, default=DEFAULT_BOUNDARY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--target", default=TARGET)
    parser.add_argument("--features", nargs="*", default=DEFAULT_FEATURES)
    parser.add_argument(
        "--kernel", default="bisquare", choices=["bisquare", "gaussian"]
    )
    parser.add_argument("--skip-gwr", action="store_true")
    parser.add_argument("--skip-mgwr", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=1)
    return parser.parse_args()


def load_model_frame(table: Path, target: str, features: list[str]) -> pd.DataFrame:
    """마스터 CSV 로드 + 필수 컬럼 검증 + 결측 행 제거."""
    df = pd.read_csv(table, encoding="utf-8-sig")
    required = [
        "구",
        "동",
        "숙소명",
        "cluster",
        "cluster_label",
        *LAT_LON,
        *COORDS,
        target,
        *features,
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {table}: {missing}")

    work = df[required].copy()
    # 좌표/타깃/변수 모두 숫자형으로 강제 (비숫자는 NaN)
    for col in [*LAT_LON, *COORDS, target, *features]:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    # 좌표/타깃/변수 결측 행 제거
    work = work.dropna(subset=[*COORDS, target, *features]).reset_index(drop=True)
    if work.empty:
        raise ValueError("No valid rows after dropping missing model fields.")
    return work


def standardize_xy(df: pd.DataFrame, target: str, features: list[str]):
    """MGWR/GWR 입력 표준 형태 — 좌표/y(=(n,1))/표준화 X."""
    coords = df[COORDS].astype(float).to_numpy()
    y = df[target].astype(float).to_numpy().reshape((-1, 1))
    x = StandardScaler().fit_transform(df[features].astype(float).to_numpy())
    return coords, y, x


def run_gwr(
    df: pd.DataFrame,
    target: str,
    features: list[str],
    out_dir: Path,
    kernel: str,
    n_jobs: int,
):
    """전체 데이터에 대해 GWR 적합 + 행 단위 결과 CSV 저장."""
    coords, y, x = standardize_xy(df, target, features)
    print(f"[GWR] rows={len(df):,}, features={len(features)}")
    # 적응형 NN bandwidth — 황금비 탐색
    t0 = time.time()
    selector = Sel_BW(coords, y, x, kernel=kernel, fixed=False, n_jobs=n_jobs)
    bw = selector.search(search_method="golden_section")
    print(f"[GWR] selected BW={int(bw)} in {time.time() - t0:.1f}s")

    # 본 적합
    t0 = time.time()
    result = GWR(coords, y, x, bw=bw, kernel=kernel, fixed=False, n_jobs=n_jobs).fit()
    print(
        f"[GWR] fit done in {time.time() - t0:.1f}s, R2={result.R2:.4f}, adj_R2={result.adj_R2:.4f}"
    )

    # 행 단위 결과 — 식별/좌표/타깃 + 국지 R²/잔차/계수/(t값)
    out = df[["구", "동", "숙소명", *LAT_LON, *COORDS, target]].copy()
    out["local_R2"] = np.asarray(result.localR2).reshape(-1)
    out["bandwidth"] = int(bw)
    out["residual"] = np.asarray(result.resid_response).reshape(-1)
    out["coef_intercept"] = result.params[:, 0]
    if hasattr(result, "tvalues"):
        out["tval_intercept"] = result.tvalues[:, 0]
    # 변수마다 계수 + t값
    for i, feature in enumerate(features, start=1):
        out[f"coef_{feature}"] = result.params[:, i]
        if hasattr(result, "tvalues"):
            out[f"tval_{feature}"] = result.tvalues[:, i]
    path = out_dir / "gwr_results_full.csv"
    out.to_csv(path, index=False, encoding="utf-8-sig")
    return out, {
        "model": "GWR",
        "rows": len(out),
        "bandwidth": int(bw),
        "R2": float(result.R2),
        "adj_R2": float(result.adj_R2),
    }


def as_bandwidth_list(raw_bw, expected_len: int) -> list[float]:
    """Sel_BW.bw 의 다양한 형태를 list[float] 로 통일 — 길이 다르면 NaN 패딩."""
    if isinstance(raw_bw, tuple):
        raw_bw = raw_bw[0]
    arr = np.asarray(raw_bw, dtype=float).reshape(-1)
    if len(arr) != expected_len:
        return [float("nan")] * expected_len
    return arr.tolist()


def fit_one_mgwr_group(
    group: pd.DataFrame,
    cluster_id,
    target: str,
    features: list[str],
    kernel: str,
    n_jobs: int,
) -> tuple[pd.DataFrame, dict]:
    """단일 cluster 그룹에 대해 MGWR 적합 — 변수별 BW 탐색 후 fit."""
    coords, y, x = standardize_xy(group, target, features)
    label = (
        group["cluster_label"].dropna().iloc[0]
        if group["cluster_label"].notna().any()
        else str(cluster_id)
    )
    print(
        f"[MGWR] cluster={cluster_id} ({label}), rows={len(group):,}, features={len(features)}"
    )
    print("[MGWR] variable-specific bandwidth search runs inside this cluster.")
    # multi=True — 변수마다 다른 BW
    t0 = time.time()
    selector = Sel_BW(
        coords, y, x, multi=True, kernel=kernel, fixed=False, n_jobs=n_jobs
    )
    selector.search(verbose=True)
    # 절편 + features = features+1 개 BW
    bandwidths = as_bandwidth_list(selector.bw, len(features) + 1)
    print(
        f"[MGWR] cluster={cluster_id} selected BW={bandwidths} in {time.time() - t0:.1f}s"
    )

    # 본 적합
    t0 = time.time()
    result = MGWR(
        coords, y, x, selector, kernel=kernel, fixed=False, n_jobs=n_jobs
    ).fit()
    print(
        f"[MGWR] cluster={cluster_id} fit done in {time.time() - t0:.1f}s, "
        f"R2={result.R2:.4f}, adj_R2={result.adj_R2:.4f}"
    )

    # 결과 데이터프레임 — 식별/좌표/타깃 + 잔차/계수/BW/표준화 X/기여도
    out = group[
        ["구", "동", "숙소명", "cluster", "cluster_label", *LAT_LON, *COORDS, target]
    ].copy()
    # mgwr 일부 버전에서 localR2 가 미구현일 수 있어 안전 처리
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
    # 표준화 X 와 변수별 기여도(coef × z) 함께 저장 — 후속 시각화용
    for i, feature in enumerate(features):
        z = x[:, i]
        out[f"z_{feature}"] = z
        out[f"contrib_{feature}"] = out[f"coef_{feature}"] * z
    metrics = {
        "model": "MGWR",
        "cluster": int(cluster_id) if pd.notna(cluster_id) else None,
        "cluster_label": str(label),
        "rows": len(out),
        "bandwidths": bandwidths,
        "R2": float(result.R2),
        "adj_R2": float(result.adj_R2),
    }
    return out, metrics


def run_mgwr_clusterwise(
    df: pd.DataFrame,
    target: str,
    features: list[str],
    out_dir: Path,
    kernel: str,
    n_jobs: int,
):
    """cluster 단위 MGWR — 그룹별 적합 후 결과 합쳐 한 CSV로 저장."""
    print(
        "[MGWR] cluster-wise mode: fit one MGWR per risk cluster, using all rows in that cluster."
    )
    outputs = []
    metrics = []
    for cluster_id in sorted(df["cluster"].dropna().unique()):
        group = df[df["cluster"] == cluster_id].reset_index(drop=True)
        # 변수 + 절편 수보다 표본이 적으면 적합 불가 — 스킵
        if len(group) <= len(features) + 2:
            print(f"[MGWR] skip cluster={cluster_id}: too few rows ({len(group):,})")
            continue
        out, metric = fit_one_mgwr_group(
            group, cluster_id, target, features, kernel, n_jobs
        )
        outputs.append(out)
        metrics.append(metric)
    if not outputs:
        raise RuntimeError("MGWR produced no cluster outputs.")
    result = pd.concat(outputs, ignore_index=True)
    path = out_dir / "mgwr_results_full.csv"
    result.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[MGWR] saved cluster-wise combined results: {path} rows={len(result):,}")
    return result, metrics


def load_boundary(path: Path) -> gpd.GeoDataFrame:
    """법정동 경계 GDF 로드 + 자치구/동명 부착 + 5179 좌표계 + 무효 지오메트리 보정."""
    boundary = gpd.read_file(path)
    boundary["구"] = boundary["EMD_CD"].astype(str).str[:5].map(GU_BY_CODE)
    boundary["동"] = boundary["EMD_KOR_NM"]
    boundary = boundary.to_crs(epsg=5179)
    boundary["geometry"] = boundary.geometry.apply(make_valid).buffer(0)
    return boundary[boundary.geometry.notna() & ~boundary.geometry.is_empty].copy()


def rank_0_100(s: pd.Series) -> pd.Series:
    """순위 백분율(0~100) 변환 — 동일값 처리: 평균순위, 단일값/모두 동일이면 50."""
    out = pd.Series(np.nan, index=s.index, dtype=float)
    valid = s.dropna()
    if valid.empty:
        return out
    if valid.max() == valid.min():
        out.loc[valid.index] = 50.0
    else:
        out.loc[valid.index] = (
            100 * (valid.rank(method="average") - 1) / (len(valid) - 1)
        )
    return out


def fill_idw(
    gdf: gpd.GeoDataFrame, value_col: str, out_col: str, k: int = 5
) -> gpd.GeoDataFrame:
    """결측 동에 대해 인접 동 상위 k개의 IDW(역거리 가중) 값으로 채움."""
    gdf[out_col] = gdf[value_col]
    # 폴리곤 대표점 — 거리 계산 기준
    points = gdf.geometry.representative_point()
    gdf["_x"] = points.x
    gdf["_y"] = points.y
    valid = gdf[gdf[value_col].notna()]
    missing = gdf[gdf[value_col].isna()]
    if valid.empty:
        # 유효값이 전혀 없으면 모두 0
        gdf[out_col] = 0.0
        return gdf
    vx, vy, vv = (
        valid["_x"].to_numpy(),
        valid["_y"].to_numpy(),
        valid[value_col].to_numpy(dtype=float),
    )
    # 결측 동 각각에 대해 k 최근접 + 거리² 역수 가중평균
    for idx, row in missing.iterrows():
        dist = np.sqrt((vx - row["_x"]) ** 2 + (vy - row["_y"]) ** 2)
        order = np.argsort(dist)[: min(k, len(dist))]
        d = np.maximum(dist[order], 1.0)  # 0 나눗셈 방지
        weights = 1 / (d**2)
        gdf.at[idx, out_col] = float(np.sum(weights * vv[order]) / np.sum(weights))
    return gdf


def plot_variable_maps(
    boundary: gpd.GeoDataFrame,
    model_outputs: dict[str, pd.DataFrame],
    features: list[str],
    maps_dir: Path,
):
    """변수별로 GWR vs MGWR 상대강도 비교 지도 PNG 생성."""
    maps_dir.mkdir(parents=True, exist_ok=True)
    matplotlib.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    # 자치구 외곽선 — 동 폴리곤 dissolve
    gu_boundary = boundary.dissolve(
        by="구", as_index=False, method="unary", grid_size=0.05
    )
    cmap = matplotlib.colormaps["YlOrRd"]
    norm = Normalize(vmin=0, vmax=100)

    for feature in features:
        map_frames = {}
        for model_name, out in model_outputs.items():
            # GWR — 강도 = |coef × t값| (없으면 |coef|)
            if model_name == "GWR":
                raw_col = f"strength_{feature}"
                out = out.copy()
                t_col = f"tval_{feature}"
                if t_col in out.columns:
                    out[raw_col] = (out[f"coef_{feature}"] * out[t_col]).abs()
                else:
                    out[raw_col] = out[f"coef_{feature}"].abs()
            # MGWR — 강도 = |contrib| (= |coef × z|), 없으면 |coef|
            else:
                raw_col = f"strength_{feature}"
                out = out.copy()
                contrib = f"contrib_{feature}"
                if contrib in out.columns:
                    out[raw_col] = out[contrib].abs()
                else:
                    out[raw_col] = out[f"coef_{feature}"].abs()

            # (구, 동) 단위 평균 → 0~100 백분위 변환
            agg = (
                out.groupby(["구", "동"], dropna=False)
                .agg(raw_strength=(raw_col, "mean"))
                .reset_index()
            )
            agg["strength"] = rank_0_100(agg["raw_strength"])
            # 경계와 좌측 결합 + 결측은 IDW 보간
            gdf = boundary.merge(agg, on=["구", "동"], how="left")
            gdf = fill_idw(gdf, "strength", "strength_full")
            map_frames[model_name] = gdf

        # 좌(GWR) / 우(MGWR) 비교 지도
        fig, axes = plt.subplots(1, len(map_frames), figsize=(20, 11.5), dpi=180)
        if len(map_frames) == 1:
            axes = [axes]
        fig.patch.set_facecolor("#f7f9fc")
        for ax, (model_name, gdf) in zip(axes, map_frames.items()):
            ax.set_facecolor("#f7f9fc")
            gdf.plot(
                ax=ax,
                column="strength_full",
                cmap=cmap,
                norm=norm,
                edgecolor="#c9d1dc",
                linewidth=0.14,
            )
            gu_boundary.boundary.plot(ax=ax, color="#2f3642", linewidth=0.8, alpha=0.9)
            # 자치구 라벨
            for _, row in gu_boundary.iterrows():
                point = row.geometry.representative_point()
                ax.text(
                    point.x,
                    point.y,
                    row["구"],
                    ha="center",
                    va="center",
                    fontsize=6,
                    weight="bold",
                )
            ax.set_axis_off()
            ax.set_title(
                f"{model_name} {feature} 상대강도",
                fontsize=16,
                weight="bold",
                loc="left",
            )

        # 공통 컬러바 — 모형 내 상대강도 0~100
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.012)
        cbar.set_label("모형 내 상대강도 0-100", fontsize=9)
        fig.suptitle(
            f"{feature}: GWR 전체 vs MGWR 군집별 전체 비교",
            fontsize=22,
            weight="bold",
            x=0.03,
            ha="left",
        )
        fig.text(
            0.03,
            0.035,
            "직접값 없는 법정동은 인접 법정동 IDW로 전체 경계를 채움",
            fontsize=9,
            color="#667085",
        )
        fig.savefig(
            maps_dir / f"{feature}_gwr_mgwr_full.png",
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
        )
        plt.close(fig)


def main() -> None:
    """전체 파이프라인 실행 — 데이터 로드 → GWR → MGWR(클러스터) → 비교지도 → 메타 저장."""
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    maps_dir = args.out_dir / "maps"
    df = load_model_frame(args.table, args.target, args.features)
    print(f"[DATA] valid rows={len(df):,} from {args.table}")

    # 메타데이터 — 입력/설정 기록 + 모델별 메트릭 추가
    metadata = {
        "table": str(args.table),
        "target": args.target,
        "features": args.features,
        "rows": int(len(df)),
        "kernel": args.kernel,
    }
    outputs = {}
    metrics = []
    if not args.skip_gwr:
        gwr_out, gwr_metrics = run_gwr(
            df, args.target, args.features, args.out_dir, args.kernel, args.n_jobs
        )
        outputs["GWR"] = gwr_out
        metrics.append(gwr_metrics)
    if not args.skip_mgwr:
        mgwr_out, mgwr_metrics = run_mgwr_clusterwise(
            df,
            args.target,
            args.features,
            args.out_dir,
            args.kernel,
            args.n_jobs,
        )
        outputs["MGWR"] = mgwr_out
        metrics.extend(mgwr_metrics)

    metadata["metrics"] = metrics
    (args.out_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 경계 파일이 있고 결과가 있으면 지도 생성
    if args.boundary.exists() and outputs:
        boundary = load_boundary(args.boundary)
        plot_variable_maps(boundary, outputs, args.features, maps_dir)
        print(f"[MAP] saved maps to {maps_dir}")
    else:
        print("[MAP] boundary not found or no model output; skipped maps.")


if __name__ == "__main__":
    main()
