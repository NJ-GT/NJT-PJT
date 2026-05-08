# -*- coding: utf-8 -*-
"""
군집(저/중/고)별 '구조노후도' 와 '단속위험도' 두 지표를 박스플롯 1×2 패널로 시각화한다.

목적:
    - 두 지표의 분포 차이를 한 그림에서 비교한다.
    - 패널마다 군집별 표본수, 평균, 중앙값 라벨을 박스 위에 표시한다.

입력:
    - NJT-PJT/0430/최종테이블0429.csv  (cluster_label, 구조노후도, 단속위험도)

출력:
    - NJT-PJT/0430/군집별_구조노후도_단속위험도_boxplot.png

처리 흐름:
    1) 한글 폰트 등록
    2) CSV 로딩 → 분석 컬럼 3개 추출 후 결측 제거
    3) 1×2 박스플롯 (좌: 구조노후도 / 우: 단속위험도)
    4) 각 패널에 stripplot 으로 표본 점 + 통계 라벨 부착
    5) 제목/서브타이틀/축 정리 후 PNG 저장
"""
from __future__ import annotations

# 경로 처리 표준 라이브러리
from pathlib import Path

# Matplotlib: GUI 없이 PNG 저장만
import matplotlib

matplotlib.use("Agg")
# 한글 폰트 등록
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
# 데이터/시각화
import pandas as pd
import seaborn as sns


# ── 경로 상수 ─────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "0430" / "최종테이블0429.csv"
OUT_PATH = ROOT / "0430" / "군집별_구조노후도_단속위험도_boxplot.png"

# 군집 정렬 순서
ORDER = ["저위험군", "중위험군", "고위험군"]
# 군집별 색
PALETTE = {
    "저위험군": "#60A5FA",
    "중위험군": "#FBBF24",
    "고위험군": "#EF4444",
}


def set_korean_font() -> str:
    """시스템 한글 폰트 자동 탐색 후 Matplotlib 에 등록."""
    candidates = [
        r"C:\Windows\Fonts\malgun.ttf",
        r"C:\Windows\Fonts\NanumGothic.ttf",
        r"C:\Windows\Fonts\NotoSansKR-Regular.otf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            fm.fontManager.addfont(candidate)
            font_name = fm.FontProperties(fname=candidate).get_name()
            plt.rcParams["font.family"] = font_name
            return font_name
    # 폰트 없으면 기본 폴백
    plt.rcParams["axes.unicode_minus"] = False
    return "sans-serif"


def main() -> None:
    """메인 시각화 루틴."""
    # 한글 폰트 적용 + seaborn 테마
    font_name = set_korean_font()
    sns.set_theme(style="whitegrid", rc={"font.family": font_name})
    plt.rcParams["font.family"] = font_name
    plt.rcParams["axes.unicode_minus"] = False

    # CSV 로딩
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()

    # 필요 컬럼만 추출 (원본 보존)
    use = df[["cluster_label", "구조노후도", "단속위험도"]].copy()
    # 정렬 가능한 카테고리로 변환
    use["cluster_label"] = pd.Categorical(
        use["cluster_label"], categories=ORDER, ordered=True
    )
    # 두 지표를 숫자형으로 강제 변환
    for col in ["구조노후도", "단속위험도"]:
        use[col] = pd.to_numeric(use[col], errors="coerce")
    # 필요한 모든 컬럼이 채워진 행만 사용
    use = use.dropna(subset=["cluster_label", "구조노후도", "단속위험도"])

    # 군집별 두 지표 통계 (n, mean, median)
    summary = (
        use.groupby("cluster_label", observed=True)[["구조노후도", "단속위험도"]]
        .agg(["count", "mean", "median"])
        .reindex(ORDER)
    )

    # ── Figure 생성 (1×2 패널) ────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 7.2), dpi=180)
    fig.patch.set_facecolor("#f6f8fb")

    # 패널별 (컬럼명, 제목, 서브타이틀) 사양
    plot_specs = [
        ("구조노후도", "구조노후도", "건물 노후 위험 지표"),
        ("단속위험도", "단속위험도", "불법주정차/단속 위험 지표"),
    ]

    # 각 패널을 동일한 스타일로 그린다
    for ax, (col, title, subtitle) in zip(axes, plot_specs):
        ax.set_facecolor("#ffffff")
        # 박스플롯
        sns.boxplot(
            data=use,
            x="cluster_label",
            y=col,
            order=ORDER,
            palette=PALETTE,
            width=0.55,
            linewidth=1.35,
            fliersize=2.0,
            ax=ax,
        )
        # 표본 점 (최대 900개)
        sns.stripplot(
            data=use.sample(min(len(use), 900), random_state=42),
            x="cluster_label",
            y=col,
            order=ORDER,
            color="#111827",
            alpha=0.13,
            size=2.2,
            jitter=0.22,
            ax=ax,
        )
        # 패널 제목 + 서브타이틀
        ax.set_title(title, fontsize=18, fontweight="bold", color="#111827", pad=18)
        ax.text(
            0.5,
            1.01,
            subtitle,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=10.5,
            color="#667085",
        )
        # 축 라벨 / 눈금
        ax.set_xlabel("")
        ax.set_ylabel("원본 지표값", fontsize=12, color="#344054")
        ax.tick_params(axis="x", labelsize=12)
        ax.tick_params(axis="y", labelsize=10.5)
        # y 그리드만 옅게
        ax.grid(axis="y", alpha=0.25)
        ax.grid(axis="x", visible=False)
        for spine in ax.spines.values():
            spine.set_color("#d0d5dd")

        # 통계 라벨 위치 (98.5% 분위수)
        ymax = use[col].quantile(0.985)
        for idx, label in enumerate(ORDER):
            row = summary.loc[label, col]
            ax.text(
                idx,
                ymax,
                f"n={int(row['count']):,}\n평균 {row['mean']:.3f}\n중앙 {row['median']:.3f}",
                ha="center",
                va="top",
                fontsize=9.2,
                color="#1f2937",
                bbox=dict(
                    boxstyle="round,pad=0.28",
                    facecolor="#ffffff",
                    edgecolor="#e5e7eb",
                    alpha=0.88,
                ),
            )

    # 전체 제목 + 서브타이틀 (패널 위쪽)
    fig.suptitle(
        "군집별 구조노후도 · 단속위험도 분포",
        fontsize=24,
        fontweight="bold",
        color="#101828",
        y=0.98,
    )
    fig.text(
        0.5,
        0.925,
        "최종테이블0429 기준 | box=사분위범위(IQR), 중앙선=중앙값, 점=표본 일부",
        ha="center",
        fontsize=11.5,
        color="#667085",
    )
    # 여백 조정 후 저장
    fig.tight_layout(rect=[0.03, 0.04, 0.97, 0.89])
    fig.savefig(OUT_PATH, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(OUT_PATH)


if __name__ == "__main__":
    main()
