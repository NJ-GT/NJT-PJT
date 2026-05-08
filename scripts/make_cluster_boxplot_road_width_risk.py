# -*- coding: utf-8 -*-
"""
군집(저/중/고위험)별 도로폭위험도 분포를 박스플롯으로 시각화하는 스크립트.

목적:
    - 0430/최종테이블0429.csv 의 cluster_label × 도로폭위험도 분포를 비교한다.
    - 군집별 표본수(n), 평균, 중앙값을 박스 위에 라벨로 표기한다.

입력:
    - NJT-PJT/0430/최종테이블0429.csv (cluster_label, 도로폭위험도 컬럼 필요)

출력:
    - NJT-PJT/0430/군집별_도로폭위험도_boxplot.png

처리 흐름:
    1) 한글 폰트 등록 후 seaborn 테마 설정
    2) CSV 로드 → 분석에 필요한 두 컬럼만 추출, 결측 제거
    3) seaborn boxplot + stripplot 으로 분포 + 표본 점 시각화
    4) 군집별 n/평균/중앙값을 박스 상단에 텍스트 박스로 표시
    5) 제목·서브타이틀·축 스타일 정리 후 PNG 저장
"""
from __future__ import annotations

# 경로 처리를 위한 표준 라이브러리
from pathlib import Path

# Matplotlib 백엔드를 GUI 없이 파일로만 출력하도록 설정
import matplotlib

matplotlib.use("Agg")  # 비대화형 백엔드 (PNG 저장 전용)
# 한글 폰트 등록을 위한 모듈
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
# 데이터 처리/시각화 라이브러리
import pandas as pd
import seaborn as sns


# ── 경로 상수 ─────────────────────────────────────────────────────────
# 이 스크립트의 한 단계 위(NJT-PJT) 디렉터리를 프로젝트 루트로 잡는다
ROOT = Path(__file__).resolve().parents[1]
# 입력 CSV (군집 라벨 + 도로폭위험도)
DATA_PATH = ROOT / "0430" / "최종테이블0429.csv"
# 출력 PNG 경로
OUT_PATH = ROOT / "0430" / "군집별_도로폭위험도_boxplot.png"

# 군집 정렬 순서 (저→중→고)
ORDER = ["저위험군", "중위험군", "고위험군"]
# 군집별 박스 색 팔레트 (저=파랑, 중=주황, 고=빨강)
PALETTE = {
    "저위험군": "#60A5FA",
    "중위험군": "#FBBF24",
    "고위험군": "#EF4444",
}


def set_korean_font() -> str:
    """시스템에 설치된 한글 폰트를 Matplotlib 에 등록하고 폰트명을 반환한다."""
    # 우선순위 후보: 맑은고딕 → 나눔고딕 → 노토산스
    candidates = [
        r"C:\Windows\Fonts\malgun.ttf",
        r"C:\Windows\Fonts\NanumGothic.ttf",
        r"C:\Windows\Fonts\NotoSansKR-Regular.otf",
    ]
    for candidate in candidates:
        # 해당 경로의 폰트 파일이 존재하면 등록 후 반환
        if Path(candidate).exists():
            fm.fontManager.addfont(candidate)
            font_name = fm.FontProperties(fname=candidate).get_name()
            plt.rcParams["font.family"] = font_name
            return font_name
    # 한글 폰트가 없으면 기본 sans-serif 폴백
    return "sans-serif"


def main() -> None:
    """박스플롯 본체. 데이터 로딩 → 시각화 → 저장."""
    # 한글 폰트 적용
    font_name = set_korean_font()
    # seaborn 테마(흰 배경 + 그리드) + 폰트 일치
    sns.set_theme(style="whitegrid", rc={"font.family": font_name})
    plt.rcParams["font.family"] = font_name
    # 마이너스 부호가 깨지지 않도록 설정
    plt.rcParams["axes.unicode_minus"] = False

    # CSV 로딩 (BOM 포함 utf-8-sig)
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    # 컬럼명 양옆 공백 제거
    df.columns = df.columns.str.strip()

    # 분석에 필요한 두 컬럼만 추출 (원본 보존)
    use = df[["cluster_label", "도로폭위험도"]].copy()
    # 군집 라벨을 정해진 순서의 카테고리로 변환 (저→중→고 정렬용)
    use["cluster_label"] = pd.Categorical(
        use["cluster_label"], categories=ORDER, ordered=True
    )
    # 도로폭위험도를 숫자형으로 강제 변환 (변환 실패 시 NaN)
    use["도로폭위험도"] = pd.to_numeric(use["도로폭위험도"], errors="coerce")
    # NaN 행 제거
    use = use.dropna()

    # 군집별 표본수 / 평균 / 중앙값 요약
    summary = (
        use.groupby("cluster_label", observed=True)["도로폭위험도"]
        .agg(["count", "mean", "median"])
        .reindex(ORDER)
    )

    # ── Figure 생성 ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10.8, 7.4), dpi=180)
    # 캔버스/축 배경색
    fig.patch.set_facecolor("#f6f8fb")
    ax.set_facecolor("#ffffff")

    # 박스플롯 (군집별 도로폭위험도 분포)
    sns.boxplot(
        data=use,
        x="cluster_label",
        y="도로폭위험도",
        order=ORDER,
        hue="cluster_label",  # 색상 매핑용 (legend=False 로 범례 숨김)
        palette=PALETTE,
        width=0.52,
        linewidth=1.45,
        fliersize=2.3,  # 이상치 점 크기
        legend=False,
        ax=ax,
    )
    # 표본 점(stripplot) — 너무 많으면 가독성 떨어지므로 최대 1000개 샘플링
    sns.stripplot(
        data=use.sample(min(len(use), 1000), random_state=42),
        x="cluster_label",
        y="도로폭위험도",
        order=ORDER,
        color="#111827",
        alpha=0.13,
        size=2.2,
        jitter=0.22,
        ax=ax,
    )

    # 텍스트 라벨 위치(상단 98.5% 분위수 → 이상치를 약간 여유로 넘어가는 지점)
    ymax = use["도로폭위험도"].quantile(0.985)
    # 군집별로 통계 라벨을 박스 위에 표시
    for idx, label in enumerate(ORDER):
        row = summary.loc[label]
        ax.text(
            idx,
            ymax,
            f"n={int(row['count']):,}\n평균 {row['mean']:.3f}\n중앙 {row['median']:.3f}",
            ha="center",
            va="top",
            fontsize=10.4,
            color="#1f2937",
            bbox=dict(
                boxstyle="round,pad=0.32",
                facecolor="#ffffff",
                edgecolor="#e5e7eb",
                alpha=0.9,
            ),
        )

    # 제목·서브타이틀
    ax.set_title(
        "군집별 도로폭위험도 분포",
        fontsize=24,
        fontweight="bold",
        color="#101828",
        pad=25,
    )
    ax.text(
        0.5,
        1.015,
        "도로폭이 좁을수록 위험도가 높게 반영된 지표 | box=IQR, 중앙선=중앙값, 점=표본 일부",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=11.5,
        color="#667085",
    )
    # x축 라벨 비움(군집명만 노출), y축은 원본 지표값
    ax.set_xlabel("")
    ax.set_ylabel("원본 지표값", fontsize=13, color="#344054")
    ax.tick_params(axis="x", labelsize=13)
    ax.tick_params(axis="y", labelsize=11)
    # y 그리드만 옅게, x 그리드 제거
    ax.grid(axis="y", alpha=0.25)
    ax.grid(axis="x", visible=False)
    # 축 테두리 색
    for spine in ax.spines.values():
        spine.set_color("#d0d5dd")

    # 여백 조정 후 저장
    fig.tight_layout(rect=[0.03, 0.03, 0.97, 0.93])
    fig.savefig(OUT_PATH, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(OUT_PATH)


if __name__ == "__main__":
    # 스크립트 직접 실행 시 main() 호출
    main()
