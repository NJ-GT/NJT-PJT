# -*- coding: utf-8 -*-
"""
OLS 잔차 LISA 지도와 MGWR 잔차 LISA 지도를 한 PNG에 좌우로 비교 시각화.

목적:
    - 동일한 6개 변수(구조노후도/단속위험도/도로폭위험도/소화용수거리등급/소방위험도/집중도)
      을 사용해 OLS 잔차를 새로 적합하고, 사전 계산된 MGWR 잔차와 함께 LISA 지도로 비교.
    - HH/LL/HL/LH 군집과 Moran's I 글로벌 통계로 모델별 잔차의 공간 구조를 진단.

입력:
    - 0430/*0429.csv                                    : OLS 적합용 마스터 테이블
    - data/full_gwr_mgwr/mgwr_results_full.csv         : 사전 계산된 MGWR 잔차 (구/동/residual)
    - data/seoul_legal_dong_age_buckets_joined_0415.geojson : 법정동 경계

출력:
    - data/full_gwr_mgwr/residual_lisa/ols_mgwr_residual_lisa_maps.png
    - data/full_gwr_mgwr/residual_lisa/{ols|mgwr}_residual_lisa_by_dong.csv
    - data/full_gwr_mgwr/residual_lisa/residual_lisa_summary.csv
"""

# 미래호환 — 인자 타입 힌트(Path | None 등) 사용을 위해
from __future__ import annotations

# 표준 라이브러리
from pathlib import Path

# 지오 처리 + matplotlib(헤드리스 백엔드)
import geopandas as gpd
import matplotlib

# Agg 백엔드 — 디스플레이 없는 환경에서도 PNG 저장 가능
matplotlib.use("Agg")
import matplotlib.patches as mpatches  # 범례용 색상 패치
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# 모란 글로벌/지역, Queen 인접성 가중치
from esda.moran import Moran, Moran_Local
from libpysal.weights import Queen
# 지오메트리 유효성 보정
from shapely.validation import make_valid
# OLS 와 표준화
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


# 경로 상수 — 스크립트 위치 기준 상대 경로
BASE = Path(__file__).resolve().parents[1]
# 0430 폴더 내 *0429.csv 첫 매칭 — 분석 마스터 테이블
TABLE_PATH = next((BASE / "0430").glob("*0429.csv"))
# MGWR 잔차 CSV (사전 계산본)
MGWR_PATH = BASE / "data" / "full_gwr_mgwr" / "mgwr_results_full.csv"
# 법정동 폴리곤 (시각화용 경계)
BOUNDARY_PATH = BASE / "data" / "seoul_legal_dong_age_buckets_joined_0415.geojson"
# 결과 출력 폴더
OUT_DIR = BASE / "data" / "full_gwr_mgwr" / "residual_lisa"

# 종속변수
TARGET = "최종위험점수_new"
# 설명변수 6개 — full GWR/MGWR 분석과 동일하게 맞춤
FEATURES = [
    "구조노후도",
    "단속위험도",
    "도로폭위험도",
    "최근접_소화용수_거리등급",
    "소방위험도_점수",
    "집중도",
]

# 법정동 코드 앞 5자리 → 자치구명 매핑 (서울시 25구)
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

# LISA 4분면 + 비유의/무자료 색상
LISA_COLORS = {
    "HH": "#d7191c",  # High-High (양의 잔차 군집) — 빨강
    "LL": "#2c7bb6",  # Low-Low  (음의 잔차 군집) — 파랑
    "HL": "#fdae61",  # High-Low (외톨이 양) — 주황
    "LH": "#abd9e9",  # Low-High (외톨이 음) — 하늘
    "Not Sig": "#d9dee7",  # 통계적 비유의 — 회색
    "No Data": "#f2f4f7",  # 자료 없음 — 매우 연한 회색
}
# 범례용 한글 라벨
LISA_LABELS = {
    "HH": "High-High: 양의 잔차 집중",
    "LL": "Low-Low: 음의 잔차 집중",
    "HL": "High-Low: 국지적 양의 잔차",
    "LH": "Low-High: 국지적 음의 잔차",
    "Not Sig": "유의하지 않음",
    "No Data": "자료 없음",
}


def load_boundary() -> gpd.GeoDataFrame:
    """법정동 경계 GeoDataFrame 로드 + 자치구/동명 매칭 컬럼 부착 + 5179 좌표계 변환."""
    gdf = gpd.read_file(BOUNDARY_PATH)
    # EMD_CD 앞 5자리로 자치구 매칭
    gdf["구_매칭"] = gdf["EMD_CD"].astype(str).str[:5].map(GU_BY_CODE)
    # 이미 '구' 컬럼이 있으면 우선 사용 (코드 매칭이 누락될 때 백업)
    if "구" in gdf.columns:
        gdf["구_매칭"] = gdf["구"].fillna(gdf["구_매칭"])
    # 법정동명 — 우선 '법정동명' 컬럼, 없으면 EMD_KOR_NM
    gdf["동_매칭"] = gdf.get("법정동명", gdf["EMD_KOR_NM"]).fillna(gdf["EMD_KOR_NM"])
    # 한국 중부 평면좌표(5179)로 변환 — 면적/거리 계산 정확도
    gdf = gdf.to_crs(epsg=5179)
    # 자기교차 등 무효 지오메트리 보정 + buffer(0) 으로 위상 정리
    gdf["geometry"] = gdf.geometry.apply(make_valid).buffer(0)
    # 비어있는 지오메트리 제거
    return gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()


def fit_ols_residuals() -> tuple[pd.DataFrame, dict]:
    """마스터 테이블로 OLS 적합 후 행 단위 잔차와 R² 반환."""
    df = pd.read_csv(TABLE_PATH, encoding="utf-8-sig")
    # 필수 컬럼 검증 — 누락 시 명시적 에러
    required = ["구", "동", TARGET, *FEATURES]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {TABLE_PATH}: {missing}")

    work = df[required].copy()
    # 숫자형 변환 — 비숫자는 NaN 처리
    for col in [TARGET, *FEATURES]:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    # 결측 행 제거 후 인덱스 재설정
    work = work.dropna(subset=[TARGET, *FEATURES]).reset_index(drop=True)
    # X 표준화 + OLS 적합
    x = StandardScaler().fit_transform(work[FEATURES].to_numpy(dtype=float))
    y = work[TARGET].to_numpy(dtype=float)
    model = LinearRegression().fit(x, y)
    pred = model.predict(x)
    # 잔차 = 관측값 - 예측값
    work["residual"] = y - pred
    metrics = {
        "model": "OLS",
        "rows": int(len(work)),
        "r2": float(model.score(x, y)),
    }
    return work[["구", "동", "residual"]], metrics


def load_mgwr_residuals() -> tuple[pd.DataFrame, dict]:
    """사전 계산된 MGWR 잔차 CSV를 로드 — 구/동/residual 3개 컬럼 검증."""
    mgwr = pd.read_csv(MGWR_PATH, encoding="utf-8-sig")
    required = ["구", "동", "residual"]
    missing = [c for c in required if c not in mgwr.columns]
    if missing:
        raise ValueError(f"Missing columns in {MGWR_PATH}: {missing}")
    return mgwr[required].copy(), {"model": "MGWR", "rows": int(len(mgwr))}


def aggregate_to_dong(residuals: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """행 단위 잔차를 (구, 동) 단위 평균 잔차 + 표본 수로 집계."""
    out = (
        residuals.groupby(["구", "동"], dropna=False)
        .agg(residual=("residual", "mean"), sample_count=("residual", "size"))
        .reset_index()
    )
    out["model"] = model_name
    return out


def classify_lisa(values: np.ndarray, p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    LISA 결과를 HH/LL/HL/LH/Not Sig 라벨로 분류.

    quadrant(q)는 esda 의 표준: 1=HH, 2=LH, 3=LL, 4=HL.
    p < 0.05 인 경우만 4분면 라벨, 그 외는 'Not Sig'.
    """
    sig = p < 0.05
    cats = np.full(len(values), "Not Sig", dtype=object)
    cats[(q == 1) & sig] = "HH"
    cats[(q == 2) & sig] = "LH"
    cats[(q == 3) & sig] = "LL"
    cats[(q == 4) & sig] = "HL"
    return cats


def run_lisa(
    boundary: gpd.GeoDataFrame, dong_values: pd.DataFrame, model_name: str
) -> tuple[gpd.GeoDataFrame, dict]:
    """
    동 단위 잔차를 경계 GDF에 결합 → Queen 가중치 → 글로벌/지역 모란 → LISA 분류 → 결과 GDF 반환.
    """
    # 경계와 잔차를 (구, 동) 키로 좌측 결합 — 잔차 없는 동은 NaN
    gdf = boundary.merge(
        dong_values, left_on=["구_매칭", "동_매칭"], right_on=["구", "동"], how="left"
    )
    # 잔차가 있는 동만 추려 LISA 계산 (없는 동은 'No Data' 처리)
    data = gdf[gdf["residual"].notna()].copy().reset_index(drop=True)
    if len(data) < 5:
        # 통계적으로 의미 있으려면 최소 표본 필요
        raise ValueError(
            f"Too few legal dongs with residuals for {model_name}: {len(data)}"
        )

    # Queen 인접성 — 변/꼭지점 공유 모두 이웃으로 인정
    weights = Queen.from_dataframe(data, use_index=False)
    weights.transform = "r"  # 행 표준화
    y = data["residual"].to_numpy(dtype=float)
    # 글로벌 모란 I — 999회 순열 검정
    moran = Moran(y, weights, permutations=999)
    # 지역 모란 — 동마다 LISA 통계
    local = Moran_Local(y, weights, permutations=999, seed=42)
    # 분류 결과/통계량을 데이터프레임에 부착
    data["lisa_cat"] = classify_lisa(y, local.p_sim, local.q)
    data["local_i"] = local.Is
    data["p_sim"] = local.p_sim
    data["quadrant"] = local.q

    # 원본 boundary GDF 에 LISA 결과 재결합 — '자료 없음' 동까지 모두 포함된 결과
    result = gdf.merge(
        data[
            [
                "구_매칭",
                "동_매칭",
                "lisa_cat",
                "local_i",
                "p_sim",
                "quadrant",
                "residual",
                "sample_count",
            ]
        ],
        on=["구_매칭", "동_매칭"],
        how="left",
        suffixes=("", "_lisa"),
    )
    # 결합 후 NaN 인 카테고리는 'No Data' 로 채움
    result["lisa_cat"] = result["lisa_cat"].fillna("No Data")
    # 모델별 요약 지표
    metrics = {
        "model": model_name,
        "dong_count": int(len(data)),
        "global_moran_i": float(moran.I),
        "global_moran_p": float(moran.p_sim),
        "hh_count": int((data["lisa_cat"] == "HH").sum()),
        "ll_count": int((data["lisa_cat"] == "LL").sum()),
        "hl_count": int((data["lisa_cat"] == "HL").sum()),
        "lh_count": int((data["lisa_cat"] == "LH").sum()),
        "not_sig_count": int((data["lisa_cat"] == "Not Sig").sum()),
    }
    return result, metrics


def plot_maps(model_maps: dict[str, gpd.GeoDataFrame], metrics: list[dict]) -> Path:
    """OLS / MGWR LISA 지도를 좌우로 그려 한 PNG 로 저장."""
    # 한글 폰트 설정 (Windows 우선, 폴백 DejaVu)
    matplotlib.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    # 1행 2열 — 가로 20인치, 고해상도 180dpi
    fig, axes = plt.subplots(1, 2, figsize=(20, 9.8), dpi=180)
    fig.patch.set_facecolor("#f7f9fc")

    # 자치구 경계 — 동 폴리곤 dissolve로 외곽선만 추출 (시각적 그룹핑)
    gu_boundary = next(iter(model_maps.values())).dissolve(
        by="구_매칭", as_index=False, method="unary", grid_size=0.05
    )
    # 모델명 → 지표 dict (제목 표시용)
    metric_by_model = {m["model"]: m for m in metrics}

    for ax, (model_name, gdf) in zip(axes, model_maps.items()):
        ax.set_facecolor("#f7f9fc")
        # LISA 카테고리 → 색상
        gdf["plot_color"] = (
            gdf["lisa_cat"].map(LISA_COLORS).fillna(LISA_COLORS["No Data"])
        )
        # 동 폴리곤 채색 + 얇은 경계선
        gdf.plot(ax=ax, color=gdf["plot_color"], edgecolor="#c9d1dc", linewidth=0.18)
        # 자치구 경계 굵게 덧그리기
        gu_boundary.boundary.plot(ax=ax, color="#303744", linewidth=0.75, alpha=0.9)
        # 자치구 라벨 — 대표점 좌표에 텍스트
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

        # 모델별 제목 — Moran's I 와 p-value 포함
        m = metric_by_model[model_name]
        ax.set_title(
            f"{model_name} 잔차 LISA\nMoran's I={m['global_moran_i']:.3f}, p={m['global_moran_p']:.3f}",
            fontsize=16,
            weight="bold",
            loc="left",
        )
        ax.set_axis_off()  # 축 눈금/테두리 제거 — 깔끔한 지도

    # 공통 범례 — 한 figure 하단 가운데
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
    # 메인 타이틀 + 좌측 정렬 + 하단 해설 캡션
    fig.suptitle(
        "OLS 잔차 LISA vs MGWR 잔차 LISA", fontsize=23, weight="bold", x=0.03, ha="left"
    )
    fig.text(
        0.03,
        0.04,
        "법정동 평균 잔차 기준. HH는 모델이 낮게 예측한 양의 잔차가 주변과 함께 몰린 곳, LL은 높게 예측한 음의 잔차가 주변과 함께 몰린 곳.",
        fontsize=9,
        color="#667085",
    )
    out_path = OUT_DIR / "ols_mgwr_residual_lisa_maps.png"
    fig.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


def main() -> None:
    """전체 파이프라인 실행: 경계 로드 → 두 모델 잔차 → LISA → 지도/CSV 저장."""
    # 출력 디렉터리 보장
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # 경계 1회 로드 후 두 모델에서 공유
    boundary = load_boundary()

    # OLS 적합 + MGWR 잔차 로드
    ols_resid, ols_metric = fit_ols_residuals()
    mgwr_resid, mgwr_metric = load_mgwr_residuals()
    # 동 단위 평균으로 집계
    model_inputs = {
        "OLS": aggregate_to_dong(ols_resid, "OLS"),
        "MGWR": aggregate_to_dong(mgwr_resid, "MGWR"),
    }

    model_maps = {}
    metrics = []
    # 모델별 LISA 실행 + 동단위 결과 CSV 저장
    for model_name, dong_values in model_inputs.items():
        gdf, lisa_metric = run_lisa(boundary, dong_values, model_name)
        model_maps[model_name] = gdf
        # OLS는 R²까지, MGWR는 행수만 — base_metric 차별
        base_metric = ols_metric if model_name == "OLS" else mgwr_metric
        metrics.append({**base_metric, **lisa_metric})
        # 동 단위 LISA 결과를 CSV 로 내보내 (외부 후처리용)
        export_cols = [
            "구_매칭",
            "동_매칭",
            "residual",
            "sample_count",
            "lisa_cat",
            "local_i",
            "p_sim",
            "quadrant",
        ]
        gdf[export_cols].to_csv(
            OUT_DIR / f"{model_name.lower()}_residual_lisa_by_dong.csv",
            index=False,
            encoding="utf-8-sig",
        )

    # 두 모델 요약 지표를 한 CSV로
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(
        OUT_DIR / "residual_lisa_summary.csv", index=False, encoding="utf-8-sig"
    )
    # 비교 지도 PNG 저장
    out_path = plot_maps(model_maps, metrics)
    print(f"Saved: {out_path}")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
