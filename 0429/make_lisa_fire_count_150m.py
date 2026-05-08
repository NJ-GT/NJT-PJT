# -*- coding: utf-8 -*-
"""
LISA(Local Indicators of Spatial Association) 분석 스크립트.

목적:
    K=2 클러스터링 산출물에서 fire_count_150m(반경 150m 화재건수)에 대한
    Local Moran's I (LISA)를 계산하고,
        - 결과 CSV (개별 시설별 lisa_I, lisa_p, lisa_q, lisa_type)
        - 글로벌 요약 CSV
        - 시설 단위 LISA 군집 지도(PNG, 경위도 산점도)
        - High-High 유형이 많이 분포한 상위 법정동 막대 그래프(PNG/CSV)
    를 한 번에 생성한다.

LISA 분류:
    q=1 -> High-High,  q=2 -> Low-High,
    q=3 -> Low-Low,    q=4 -> High-Low
    p<0.05 인 행만 유의 표기, 그 외는 'Not significant'
"""

from __future__ import annotations

from pathlib import Path

# matplotlib 백엔드 비-GUI 모드
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# 지역 모란 I (LISA) 구현
from esda.moran import Moran_Local
# K-최근접 공간 가중치
from libpysal.weights import KNN


# 프로젝트 루트 (0429 폴더 기준 한 단계 위)
ROOT = Path(__file__).resolve().parents[1]
# K=2 결과 폴더 (입력 CSV 위치)
K2_DIR = ROOT / "0429" / "cluster2_spatial_pipeline_fire_count_150m_0429"
# LISA 산출물을 모아둘 하위 폴더
OUT_DIR = K2_DIR / "lisa_fire_count_150m"
# 분석 대상 변수 (반경 150m 내 화재건수)
TARGET = "fire_count_150m"
# 공간 가중치 KNN의 k
KNN_K = 6
# 가짜순열 검정 횟수 (p값 안정성 ↑)
PERMUTATIONS = 999


def set_korean_font() -> None:
    """그래프에서 한글이 깨지지 않도록 윈도우 한글 폰트로 설정."""
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.family"] = "Malgun Gothic"


def read_data() -> pd.DataFrame:
    """K=2 산출 CSV에서 분석에 필요한 컬럼만 골라 읽고 결측을 제거."""
    # K2_DIR 안에서 *cluster_k2.csv 패턴 중 가장 큰 파일 사용
    csv_files = sorted(
        K2_DIR.glob("*cluster_k2.csv"), key=lambda p: p.stat().st_size, reverse=True
    )
    if not csv_files:
        raise FileNotFoundError(K2_DIR)
    # UTF-8 BOM 호환 로드
    df = pd.read_csv(csv_files[0], encoding="utf-8-sig")
    # 분석에 필요한 최소 컬럼 세트
    needed = [
        "구",
        "동",
        "숙소명",
        "경도",
        "위도",
        "x_5181",
        "y_5181",
        TARGET,
        "cluster_k2",
    ]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        # 누락이 있으면 이후 분석이 부정확해지므로 즉시 실패
        raise KeyError(f"Missing columns: {missing}")
    # 수치형 컬럼 일괄 변환
    for col in ["경도", "위도", "x_5181", "y_5181", TARGET, "cluster_k2"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # 좌표/타겟 결측 제거 (LISA 입력은 NaN 불가)
    return df.dropna(subset=["경도", "위도", "x_5181", "y_5181", TARGET]).reset_index(
        drop=True
    )


def classify_lisa(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """공간 가중치 + Local Moran 계산 후 유형(lisa_type) 컬럼 부여.

    반환:
        out: lisa_I/lisa_p/lisa_q/lisa_significant/lisa_type 컬럼이 추가된 df
        global_info: 전체 요약 통계 dict
    """
    # 미터 좌표(EPSG:5181)로 거리 기반 KNN 구성
    coords = df[["x_5181", "y_5181"]].to_numpy(dtype=float)
    values = df[TARGET].to_numpy(dtype=float)
    # n이 KNN_K+1보다 작으면 안전한 k로 자동 축소
    w = KNN.from_array(coords, k=min(KNN_K, len(df) - 1))
    # 행 표준화 (이웃 평균 해석 가능)
    w.transform = "r"
    # Local Moran's I 적합 (랜덤시드 고정으로 재현성 확보)
    lisa = Moran_Local(values, w, permutations=PERMUTATIONS, seed=42)

    out = df.copy()
    # 각 시설별 LISA 통계량과 유의성 부착
    out["lisa_I"] = lisa.Is
    out["lisa_p"] = lisa.p_sim
    out["lisa_q"] = lisa.q
    # 유의 임계 p<0.05
    out["lisa_significant"] = out["lisa_p"] < 0.05

    # 사분면 코드 -> 명칭 매핑
    labels = {1: "High-High", 2: "Low-High", 3: "Low-Low", 4: "High-Low"}
    # 기본값을 'Not significant'로 둔 뒤, 유의한 행만 라벨로 덮어쓰기
    out["lisa_type"] = "Not significant"
    sig = out["lisa_significant"]
    out.loc[sig, "lisa_type"] = (
        out.loc[sig, "lisa_q"].map(labels).fillna("Not significant")
    )

    # 글로벌 요약 (보고/리포트용 한 줄 통계)
    global_info = {
        "knn_k": KNN_K,
        "permutations": PERMUTATIONS,
        "n": int(len(out)),
        "target_mean": float(out[TARGET].mean()),
        "target_median": float(out[TARGET].median()),
        "high_high_n": int((out["lisa_type"] == "High-High").sum()),
        "low_low_n": int((out["lisa_type"] == "Low-Low").sum()),
        "high_low_n": int((out["lisa_type"] == "High-Low").sum()),
        "low_high_n": int((out["lisa_type"] == "Low-High").sum()),
        "not_significant_n": int((out["lisa_type"] == "Not significant").sum()),
    }
    return out, global_info


def plot_lisa_map(df: pd.DataFrame, out_path: Path) -> None:
    """경위도 좌표를 산점도로 그려 LISA 유형을 색/크기로 구분한 지도 생성."""
    # 유형별 색상 — 빨강 계열은 위험 군집 강조
    colors = {
        "High-High": "#d73027",
        "Low-Low": "#4575b4",
        "High-Low": "#fc8d59",
        "Low-High": "#91bfdb",
        "Not significant": "#c8cdd2",
    }
    # 유형별 점 크기 — High-High를 가장 강조
    sizes = {
        "High-High": 24,
        "Low-Low": 16,
        "High-Low": 22,
        "Low-High": 18,
        "Not significant": 8,
    }
    # 그리는 순서: 'Not significant'를 먼저 깔고, 강조 유형은 나중에(위에)
    order = ["Not significant", "Low-Low", "Low-High", "High-Low", "High-High"]

    fig, ax = plt.subplots(figsize=(11.5, 8.8), dpi=180)
    for typ in order:
        # 각 유형 부분집합만 산점도로 누적
        part = df[df["lisa_type"] == typ]
        if part.empty:
            continue
        ax.scatter(
            part["경도"],
            part["위도"],
            s=sizes[typ],
            color=colors[typ],
            # 비유의는 흐리게(0.28), 유의는 진하게(0.82)
            alpha=0.82 if typ != "Not significant" else 0.28,
            linewidths=0,
            label=f"{typ} (n={len(part):,})",
        )
    # 제목/축 라벨/그리드/배경
    ax.set_title("LISA 군집도: 150m 화재건수 기준", fontsize=19, weight="bold", pad=18)
    ax.set_xlabel("경도")
    ax.set_ylabel("위도")
    ax.grid(True, color="#e7ebf0", linewidth=0.8)
    ax.set_facecolor("#fbfcfe")
    # 범례 — 유형별 표본 수까지 함께 표기 (위에서 label에 포함됨)
    ax.legend(
        title="LISA 유형",
        loc="upper right",
        frameon=True,
        facecolor="white",
        edgecolor="#d7dde5",
        fontsize=9,
        title_fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_high_high_bar(df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    """High-High 유형이 많이 분포한 상위 법정동 막대 그래프 + 표 반환."""
    # High-High만 필터
    hh = df[df["lisa_type"] == "High-High"].copy()
    # (구, 동) 단위 집계: 시설수, 평균 150m 화재건수, 평균 LISA I, cluster_k2==1 비율
    top = (
        hh.groupby(["구", "동"])
        .agg(
            high_high_시설수=("숙소명", "count"),
            평균_150m화재건수=(TARGET, "mean"),
            평균_lisa_I=("lisa_I", "mean"),
            고위험군_cluster1_비율=("cluster_k2", lambda s: float((s == 1).mean())),
        )
        .reset_index()
        # 시설 수가 많고, 화재건수 평균이 큰 순으로 정렬
        .sort_values(["high_high_시설수", "평균_150m화재건수"], ascending=False)
    )

    # 상위 12개만 그래프화
    plot = top.head(12).copy()
    # x축 라벨용 "구 동" 결합
    plot["지역"] = plot["구"] + " " + plot["동"]
    fig, ax = plt.subplots(figsize=(11.5, 7.4), dpi=180)
    # 빨강 팔레트의 역순 — 위에서부터 진한 빨강이 되도록
    sns.barplot(data=plot, y="지역", x="high_high_시설수", palette="Reds_r", ax=ax)
    # 막대 옆에 화재건수 평균을 텍스트로 부착
    for i, row in plot.reset_index(drop=True).iterrows():
        ax.text(
            row["high_high_시설수"] + 0.5,
            i,
            f"평균 {row['평균_150m화재건수']:.1f}건",
            va="center",
            fontsize=9.5,
            color="#333333",
        )
    # 제목/축 스타일
    ax.set_title("LISA High-High 상위 법정동", fontsize=18, weight="bold", pad=16)
    ax.set_xlabel("High-High 시설 수")
    ax.set_ylabel("")
    ax.grid(axis="x", color="#e7ebf0", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    # 추후 CSV로도 저장하기 위해 정렬 결과 반환
    return top


def main() -> None:
    """LISA 결과 / 요약 / 지도 / 상위 동 그래프를 한 번에 생성."""
    set_korean_font()
    # 산출 폴더가 없으면 생성
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # 입력 데이터 로드 -> LISA 분류
    df = read_data()
    lisa_df, global_info = classify_lisa(df)
    # 개별 결과 CSV
    lisa_df.to_csv(
        OUT_DIR / "lisa_fire_count_150m_results.csv", index=False, encoding="utf-8-sig"
    )
    # 글로벌 요약 CSV (한 줄)
    pd.DataFrame([global_info]).to_csv(
        OUT_DIR / "lisa_fire_count_150m_summary.csv", index=False, encoding="utf-8-sig"
    )
    # 군집도 지도 PNG
    plot_lisa_map(lisa_df, OUT_DIR / "lisa_fire_count_150m_map.png")
    # 상위 동 PNG (+ 정렬된 표 반환)
    top = plot_high_high_bar(lisa_df, OUT_DIR / "lisa_high_high_top_dongs.png")
    # 상위 동 표 CSV
    top.to_csv(
        OUT_DIR / "lisa_high_high_top_dongs.csv", index=False, encoding="utf-8-sig"
    )
    # 콘솔 검증 출력
    print(OUT_DIR)
    print(pd.DataFrame([global_info]).to_string(index=False))
    print(top.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
