# -*- coding: utf-8 -*-
"""
K-Means 군집 파라미터 튜닝 과정 1장 요약 PNG 생성 스크립트.

목적:
    - 최종 저장된 cluster 라벨이 어떤 K/입력 공간에서 도출되었는지 검증/근거를 한 장에 정리.
    - K=2~8 범위에서 4개 지표(Inertia/Silhouette/CH/DB)를 측정해 K=3 채택의 타당성 시각화.

입력:
    - 0430/최종테이블0429.csv  (10개 변수 + cluster/cluster_label/최종위험점수_new 포함)

출력:
    - 0430/kmeans_군집파라미터_튜닝과정.png

처리 흐름:
    1) 한글 폰트 세팅 + 데이터 로드
    2) 두 입력 공간(점수 1D vs 10변수)에서 KMeans 적합 + 지표 측정
    3) 저장된 cluster 라벨과 ARI 비교 (재현성 점검)
    4) 4개 지표 라인플롯, 지표 표, 군집 규모 막대, 해석 노트를 GridSpec 으로 배치
"""
from __future__ import annotations  # 미래호환 — Path | None 등 타입 힌트

from pathlib import Path  # 경로 안전 처리

import matplotlib.font_manager as fm  # 한글 폰트 등록
import matplotlib

# 디스플레이 없는 환경 지원
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# KMeans + 4개 클러스터링 지표
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,        # 두 라벨링의 일치도
    calinski_harabasz_score,    # 군집 간 분산 / 군집 내 분산 (클수록 좋음)
    davies_bouldin_score,       # 군집 간 거리 대비 군집 내 산포 (작을수록 좋음)
    silhouette_score,           # -1 ~ 1, 클수록 분리/응집 양호
)
from sklearn.preprocessing import MinMaxScaler  # 변수별 범위 차이 보정


# 경로 — 스크립트 위치 기준
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "0430" / "최종테이블0429.csv"
OUT_PATH = ROOT / "0430" / "kmeans_군집파라미터_튜닝과정.png"


# 10변수 입력 공간(피처 기반 클러스터링용)
FEATURES = [
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


def set_korean_font() -> None:
    """Windows 한글 폰트(맑은 고딕/나눔고딕/노토산스)를 우선순위로 찾아 등록."""
    candidates = [
        r"C:\Windows\Fonts\malgun.ttf",
        r"C:\Windows\Fonts\NanumGothic.ttf",
        r"C:\Windows\Fonts\NotoSansKR-Regular.otf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            # 폰트 매니저에 추가 후 기본 폰트로 지정
            fm.fontManager.addfont(candidate)
            plt.rcParams["font.family"] = fm.FontProperties(fname=candidate).get_name()
            break
    # 음수 부호 깨짐 방지
    plt.rcParams["axes.unicode_minus"] = False


def kmeans_metrics(x: np.ndarray, k_range: range) -> pd.DataFrame:
    """K 후보별로 KMeans 학습 후 4개 지표를 한 번에 산출."""
    rows: list[dict] = []
    for k in k_range:
        # 동일 시드/초기화 횟수로 재현성 확보
        km = KMeans(n_clusters=k, random_state=42, n_init=10, init="k-means++")
        labels = km.fit_predict(x)
        rows.append(
            {
                "k": k,
                "inertia": float(km.inertia_),  # SSE — 작을수록 좋음
                "silhouette": float(silhouette_score(x, labels)),
                "calinski_harabasz": float(calinski_harabasz_score(x, labels)),
                "davies_bouldin": float(davies_bouldin_score(x, labels)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    """튜닝 + 시각화 메인 진입점."""
    set_korean_font()
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    # 컬럼명 좌우 공백 제거(엑셀 편집 흔적 방지)
    df.columns = df.columns.str.strip()

    # 입력 공간 1: 점수 1D — 결측은 0으로 채우고 numpy 배열로
    score_x = (
        df[["최종위험점수_new"]]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .to_numpy()
    )
    # 입력 공간 2: 10변수 — MinMax 정규화 (변수 단위 차이 제거)
    feature_x = df[FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0)
    feature_x = MinMaxScaler().fit_transform(feature_x)
    # CSV 에 저장된 기존 cluster 라벨 — 이번 튜닝과 비교 대조
    stored_labels = df["cluster"].astype(int).to_numpy()

    # 두 입력 공간에서 K=2~8 범위 지표 산출
    score_metrics = kmeans_metrics(score_x, range(2, 9))
    feature_metrics = kmeans_metrics(feature_x, range(2, 9))

    # K=3 라벨을 두 입력 공간에서 각각 도출 → 저장 cluster 와 ARI 비교
    score_k3_labels = KMeans(
        n_clusters=3, random_state=42, n_init=10, init="k-means++"
    ).fit_predict(score_x)
    feature_k3_labels = KMeans(
        n_clusters=3, random_state=42, n_init=10, init="k-means++"
    ).fit_predict(feature_x)
    ari_score = adjusted_rand_score(stored_labels, score_k3_labels)
    ari_feature = adjusted_rand_score(stored_labels, feature_k3_labels)

    # 저장된 cluster 라벨에 대해 점수 1D 기준의 지표 — 검증 노트용
    stored_score_metrics = {
        "silhouette": silhouette_score(score_x, stored_labels),
        "calinski_harabasz": calinski_harabasz_score(score_x, stored_labels),
        "davies_bouldin": davies_bouldin_score(score_x, stored_labels),
    }

    # 군집별(저·중·고위험) 시설 수와 위험점수 통계 — 평균 오름차순
    summary = (
        df.groupby(["cluster", "cluster_label"])["최종위험점수_new"]
        .agg(["count", "min", "mean", "max"])
        .reset_index()
        .sort_values("mean")
    )

    # ── Figure & GridSpec 구성 ──
    fig = plt.figure(figsize=(18, 10.2), dpi=180)
    fig.patch.set_facecolor("#f5f7fb")
    # 3행 4열 — 1행 타이틀/메모, 2행 4개 지표 라인플롯, 3행 표/막대/노트
    gs = fig.add_gridspec(
        3, 4, height_ratios=[0.92, 1.35, 1.15], hspace=0.42, wspace=0.28
    )

    # ── [1행] 제목 + 핵심 메모 ──
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis("off")  # 보이지 않는 캔버스 — 텍스트만 사용
    ax_title.text(
        0.012,
        0.88,
        "K-Means 군집 파라미터 및 튜닝 과정",
        fontsize=25,
        fontweight="bold",
        color="#101828",
        va="top",
    )
    ax_title.text(
        0.012,
        0.58,
        "최종 파일: 0430/최종테이블0429.csv  |  기준 점수: 최종위험점수_new  |  군집명: 평균 점수 기준 저위험군·중위험군·고위험군",
        fontsize=12.5,
        color="#475467",
        va="top",
    )

    # 사용 파라미터 글머리 — 발표/문서 자료용
    param_lines = [
        "사용 알고리즘: sklearn.cluster.KMeans",
        "최종 군집 수: n_clusters=3",
        "초기화: init='k-means++'",
        "반복 초기화: n_init=10",
        "재현성: random_state=42",
        "전처리 후보: 최종위험점수 1D 또는 10개 변수 MinMax 정규화",
    ]
    y = 0.28
    for line in param_lines:
        ax_title.text(0.018, y, f"• {line}", fontsize=12, color="#1d2939", va="top")
        y -= 0.12

    # 검증 메모 박스 — 저장 cluster 와 ARI 비교 결과 강조
    ax_title.text(
        0.62,
        0.30,
        f"검증 메모\n저장 cluster vs 점수 1D KMeans ARI = {ari_score:.3f}\n저장 cluster vs 10변수 KMeans ARI = {ari_feature:.3f}\n=> 최종 cluster 생성 스크립트/입력공간 확인 필요",
        fontsize=12,
        color="#7a2e0e",
        bbox=dict(
            boxstyle="round,pad=0.55,rounding_size=0.12",
            facecolor="#fff4e5",
            edgecolor="#ffd8a8",
        ),
        va="top",
    )

    def lineplot(ax, data, ycol, title, ylabel, better_note, color, mark_k3=True):
        """
        지표 1개를 K별 라인 + K=3 위치 강조 마커로 그리는 헬퍼.

        better_note : '클수록 좋음' / '작을수록 좋음' 등 해석 가이드 (좌하단 표기)
        """
        ax.plot(data["k"], data[ycol], marker="o", linewidth=2.4, color=color)
        if mark_k3:
            # K=3 지점만 큰 점 + 주석으로 강조 — 채택값임을 시각화
            k3 = data.loc[data["k"].eq(3), ycol].iloc[0]
            ax.scatter([3], [k3], s=120, color="#101828", zorder=5)
            ax.annotate(
                f"K=3\n{k3:,.3f}",
                xy=(3, k3),
                xytext=(3.25, k3),
                fontsize=10,
                color="#101828",
                arrowprops=dict(arrowstyle="-", color="#667085"),
            )
        ax.set_title(title, fontsize=13, fontweight="bold", color="#101828")
        ax.set_xlabel("K 후보")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.26)
        ax.set_facecolor("#ffffff")
        ax.text(
            0.02,
            0.04,
            better_note,
            transform=ax.transAxes,
            fontsize=9.5,
            color="#667085",
        )

    # ── [2행] 4개 지표 라인플롯 ──
    ax1 = fig.add_subplot(gs[1, 0])
    lineplot(
        ax1,
        score_metrics,
        "inertia",
        "Elbow: SSE / Inertia",
        "작을수록 좋음",
        "급격한 감소 후 완만해지는 지점 확인",
        "#2563eb",
    )

    ax2 = fig.add_subplot(gs[1, 1])
    lineplot(
        ax2,
        score_metrics,
        "silhouette",
        "Silhouette Score",
        "클수록 좋음",
        "분리도와 응집도 균형",
        "#059669",
    )

    ax3 = fig.add_subplot(gs[1, 2])
    lineplot(
        ax3,
        score_metrics,
        "calinski_harabasz",
        "Calinski-Harabasz",
        "클수록 좋음",
        "군집 간 분산 / 군집 내 분산",
        "#dc6803",
    )

    ax4 = fig.add_subplot(gs[1, 3])
    lineplot(
        ax4,
        score_metrics,
        "davies_bouldin",
        "Davies-Bouldin",
        "작을수록 좋음",
        "군집 간 겹침이 작을수록 양호",
        "#c11574",
    )

    # ── [3행 좌] 지표 요약 표 ──
    ax_table = fig.add_subplot(gs[2, 0:2])
    ax_table.axis("off")
    table_df = score_metrics.copy()
    # 모든 수치 컬럼을 천단위 콤마+소수3자리 문자열로 포맷팅
    for col in ["inertia", "silhouette", "calinski_harabasz", "davies_bouldin"]:
        table_df[col] = table_df[col].map(lambda v: f"{v:,.3f}")
    cell_text = table_df[
        ["k", "inertia", "silhouette", "calinski_harabasz", "davies_bouldin"]
    ].values
    table = ax_table.table(
        cellText=cell_text,
        colLabels=["K", "Inertia", "Silhouette", "CH", "DB"],
        loc="center",
        cellLoc="center",
        colColours=["#e7eefb"] * 5,
    )
    # 폰트 크기/스케일 수동 조절 — 가독성
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1, 1.45)
    # 헤더는 굵게, K=3 행은 노란 배경으로 강조
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#d0d5dd")
        if r == 0:
            cell.set_text_props(weight="bold", color="#101828")
        elif int(table_df.iloc[r - 1]["k"]) == 3:
            cell.set_facecolor("#fff4cc")
    ax_table.set_title(
        "K 후보별 튜닝 지표: 최종위험점수_new 1D 기준",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )

    # ── [3행 중] 군집 규모 막대 ──
    ax_bar = fig.add_subplot(gs[2, 2])
    colors = ["#60a5fa", "#fbbf24", "#ef4444"]  # 저(파)/중(노)/고(빨) 위험군
    ax_bar.bar(
        summary["cluster_label"],
        summary["count"],
        color=colors,
        edgecolor="#ffffff",
        linewidth=1.2,
    )
    # 막대 위 시설 수 텍스트
    for i, row in enumerate(summary.itertuples()):
        ax_bar.text(
            i,
            row.count + 35,
            f"{row.count:,}",
            ha="center",
            fontsize=11,
            fontweight="bold",
            color="#101828",
        )
    ax_bar.set_title("최종 저장 군집 규모", fontsize=13, fontweight="bold")
    ax_bar.set_ylabel("시설 수")
    ax_bar.grid(axis="y", alpha=0.25)
    ax_bar.set_facecolor("#ffffff")

    # ── [3행 우] 해석 포인트 노트 박스 ──
    ax_note = fig.add_subplot(gs[2, 3])
    ax_note.axis("off")
    note = (
        "해석 포인트\n"
        "1. 최종 산출물은 K=3 체계로 저·중·고 위험군을 사용.\n"
        "2. KMeans 기본 파라미터는 n_clusters=3, random_state=42, n_init=10.\n"
        "3. 점수 1개만으로 재군집하면 저장 cluster와 완전 일치하지 않음.\n"
        "4. 따라서 최종 발표에는 'K=3 사용'과 함께 최종 생성 코드의 입력공간을 명시해야 함.\n"
        f"5. 저장 cluster 기준 점수 1D 지표: Sil={stored_score_metrics['silhouette']:.3f}, "
        f"CH={stored_score_metrics['calinski_harabasz']:.1f}, DB={stored_score_metrics['davies_bouldin']:.3f}"
    )
    ax_note.text(
        0,
        1,
        note,
        fontsize=11.4,
        linespacing=1.62,
        color="#1d2939",
        va="top",
        bbox=dict(
            boxstyle="round,pad=0.72,rounding_size=0.12",
            facecolor="#ffffff",
            edgecolor="#d0d5dd",
        ),
    )

    # 하단 footer — PNG 출처/주의사항
    fig.text(
        0.012,
        0.015,
        "주의: 이 PNG는 현재 저장된 최종 CSV와 저장소 내 확인 가능한 KMeans 계열 스크립트를 기준으로 재계산한 근거 자료입니다.",
        fontsize=10.5,
        color="#667085",
    )
    fig.savefig(OUT_PATH, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(OUT_PATH)


if __name__ == "__main__":
    main()
