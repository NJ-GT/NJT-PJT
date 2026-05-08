# -*- coding: utf-8 -*-
"""
방법론 자산 생성 스크립트.

목적:
    - 핵심 변수 테이블에 대해 VIF(다중공선성) 진단 결과 CSV/PNG 생성
    - PPT용 "분석 방법론" 슬라이드(이미지) 생성
    - 슬라이드 핵심 문구를 정리한 노트 텍스트 파일 생성

산출물:
    OUT_DIR/vif_check_cluster3_fire_count_150m.{csv,png}
    OUT_DIR/ppt_methodology_pipeline_revised_0429.png
    OUT_DIR/ppt_methodology_pipeline_revised_0429_notes.txt
"""
from pathlib import Path

# matplotlib 백엔드 강제 (PNG 저장만 필요 — 비-GUI)
import matplotlib

matplotlib.use("Agg")
# 폰트 매니저 (한글 폰트 등록용)
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# 슬라이드 카드/배너용 도형
from matplotlib.patches import FancyBboxPatch, Rectangle
# VIF 계산을 위한 1대-나머지 회귀 (R² 추출)
from sklearn.linear_model import LinearRegression
# 변수 표준화 — 스케일 차이 정규화
from sklearn.preprocessing import StandardScaler


# 프로젝트의 두 단계 위 (NJT-PJT/0429/ -> 깃 루트/) — 입력 경로 구성을 위한 기준
BASE = Path(__file__).resolve().parents[2]
# 산출물 저장 폴더
OUT_DIR = BASE / "NJT-PJT" / "0429"
# 입력 변수 테이블 폴더
SRC_DIR = (
    BASE
    / "NJT-PJT"
    / "0424"
    / "data"
    / "cluster3_spatial_pipeline_fire_count_150m_0428"
)


def setup_font() -> None:
    """윈도우의 맑은 고딕 폰트를 등록해 한글 깨짐 방지."""
    font_path = Path("C:/Windows/Fonts/malgun.ttf")
    if font_path.exists():
        # FontManager에 직접 추가하여 matplotlib에서 사용 가능하게 등록
        fm.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = fm.FontProperties(fname=str(font_path)).get_name()
    # 음수 부호 깨짐 방지
    plt.rcParams["axes.unicode_minus"] = False


def compute_vif(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """선형회귀 기반 VIF 계산.

    각 변수 j에 대해 나머지 변수들로 j를 회귀했을 때의 R²를 이용해
    VIF_j = 1 / (1 - R²) 로 계산한다. 표준화된 입력에서 수행.
    """
    # 작업용 사본
    work = df[features].copy()
    # 변수 모두 수치형 변환 (문자/혼합 타입 안전)
    for col in features:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    # 결측이 있는 행은 제외 (회귀 적합 안정)
    work = work.dropna(subset=features).reset_index(drop=True)

    # 표준화 — VIF는 스케일 영향이 거의 없지만, 수치 안정성 확보용
    x_scaled = StandardScaler().fit_transform(work[features].to_numpy(dtype=float))
    rows = []
    for j, col in enumerate(features):
        # 종속: j번째 변수
        y = x_scaled[:, j]
        # 독립: 나머지 변수
        x_other = np.delete(x_scaled, j, axis=1)
        r2 = LinearRegression().fit(x_other, y).score(x_other, y)
        # 완전 공선이면(r2≈1) 무한대로 마킹
        vif = np.inf if r2 >= 0.999999 else 1.0 / (1.0 - r2)
        rows.append({"variable": col, "vif": float(vif)})
    # 큰 VIF가 위에 오도록 정렬해 반환
    return pd.DataFrame(rows).sort_values("vif", ascending=False).reset_index(drop=True)


def save_vif_png(vif_df: pd.DataFrame, n_rows: int, out_png: Path) -> None:
    """VIF 결과를 가로 막대그래프 PNG로 저장 (5/10 기준선 포함)."""
    fig, ax = plt.subplots(figsize=(12, 7), dpi=180)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    # 제목/부제목 영역 확보
    fig.subplots_adjust(top=0.82, left=0.22, right=0.96, bottom=0.12)

    # 시각화는 작은 값을 아래에 두기 위해 오름차순 정렬
    plot_df = vif_df.sort_values("vif", ascending=True)
    # 5 미만 파랑, 10 미만 주황, 그 이상 빨강 — 단계별 경고 색
    colors = [
        "#2563EB" if v < 5 else "#F97316" if v < 10 else "#DC2626"
        for v in plot_df["vif"]
    ]
    ax.barh(plot_df["variable"], plot_df["vif"], color=colors, height=0.62)
    # 주의/심각 임계 기준선
    ax.axvline(5, color="#F59E0B", linewidth=1.4, linestyle="--")
    ax.axvline(10, color="#DC2626", linewidth=1.4, linestyle="--")
    # 기준선 옆 라벨
    ax.text(5.05, len(plot_df) - 0.55, "주의 기준 5", color="#92400E", fontsize=9)
    ax.text(10.05, len(plot_df) - 0.55, "심각 기준 10", color="#991B1B", fontsize=9)

    # 막대 끝에 수치 표기
    for y, v in enumerate(plot_df["vif"]):
        ax.text(
            v + 0.03,
            y,
            f"{v:.2f}",
            va="center",
            fontsize=10,
            color="#111827",
            weight="bold",
        )

    # x축 범위 — 최대 VIF에 맞춰 자동 확장 (최소 10.8 보장)
    max_vif = float(vif_df["vif"].max())
    ax.set_xlim(0, max(10.8, max_vif + 0.8))
    ax.set_xlabel("VIF", fontsize=11, color="#374151")
    # 그림 좌상단 큰 제목 (axes 외 좌표)
    fig.text(
        0.22, 0.94, "다중공선성 점검: VIF", fontsize=18, weight="bold", color="#111827"
    )
    # 부제목 — 입력 테이블 정보, n, 최대 VIF
    fig.text(
        0.22,
        0.905,
        f"기준 테이블: 최최최종0428변수테이블.csv · 설명변수 10개 · n={n_rows:,} · 최대 VIF={max_vif:.2f}",
        fontsize=10.5,
        color="#4B5563",
    )
    # x축 격자 + 축 테두리 제거 — 미니멀 스타일
    ax.grid(axis="x", color="#E5E7EB")
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def save_methodology_slide(
    out_png: Path,
    n_total: int,
    matched_150: int,
    max_vif: float,
    ols_moran_min: float,
    ols_moran_max: float,
    best_slm: float,
    best_gwr: float,
    best_mgwr: float,
) -> None:
    """PPT 한 장 분량의 '분석 방법론' 슬라이드 PNG 생성."""
    # 16:9 비율, 슬라이드 형식
    fig = plt.figure(figsize=(16, 9), dpi=180)
    fig.patch.set_facecolor("#F8FAFC")
    # 한 장 슬라이드라 axes를 풀-블리드로 두고 모두 텍스트로 그린다
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    # 좌상단 챕터 칩
    ax.text(
        0.035,
        0.93,
        "분석 방법론",
        fontsize=12,
        color="#2563EB",
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.35", fc="#EFF6FF", ec="#BFDBFE"),
    )
    # 큰 제목
    ax.text(
        0.035,
        0.84,
        "데이터 → 변수검증 → 군집화 → 공간통계 → 정책 제안 파이프라인",
        fontsize=27,
        weight="bold",
        color="#111827",
    )
    # 제목 밑줄 강조
    ax.add_patch(
        Rectangle((0.035, 0.79), 0.045, 0.006, color="#2563EB", transform=ax.transAxes)
    )

    # 5단계 카드 정의 (스텝 라벨, 제목, 본문)
    cards = [
        (
            "STEP 1",
            "분석 테이블 확정",
            "최최최종0428변수테이블.csv\n4,246개 숙박시설 · 10개 설명변수\n외부 매칭: fire_count_150m",
        ),
        (
            "STEP 2",
            "타깃/반경 선택",
            "100m 3,209개 / 150m 3,794개 / 200m 4,049개\n150m = 매칭률 89.4% + 변별력 균형",
        ),
        (
            "STEP 3",
            "VIF 다중공선성 점검",
            f"승인연도·소방위험도·주변건물수 등 10변수\n최대 VIF {max_vif:.2f} · 기준 5 미만",
        ),
        (
            "STEP 4",
            "K-Means 군집화",
            "표준화 후 K=3 군집 사용\ncluster 0/1/2 = 시설 위험요인 조합별 유형",
        ),
        (
            "STEP 5",
            "공간통계·공간회귀",
            "KNN k=12 row-standardized\nOLS+Moran → SLM/SEM → GWR/MGWR",
        ),
    ]
    # 카드별 좌측 x 좌표 (수동으로 정렬)
    card_x = [0.035, 0.225, 0.415, 0.605, 0.795]
    for x, (step, title, body) in zip(card_x, cards):
        # 각 카드는 둥근 모서리 박스 (axes 좌표계 기준)
        box = FancyBboxPatch(
            (x, 0.56),
            0.165,
            0.17,
            boxstyle="round,pad=0.012,rounding_size=0.012",
            linewidth=1,
            edgecolor="#DDE5EF",
            facecolor="white",
            transform=ax.transAxes,
        )
        ax.add_patch(box)
        # STEP 라벨
        ax.text(
            x + 0.012,
            0.705,
            step,
            fontsize=8.5,
            color="#2563EB",
            weight="bold",
            transform=ax.transAxes,
        )
        # 카드 제목
        ax.text(
            x + 0.012,
            0.675,
            title,
            fontsize=11.5,
            color="#111827",
            weight="bold",
            transform=ax.transAxes,
        )
        # 본문 (여러 줄)
        ax.text(
            x + 0.012,
            0.635,
            body,
            fontsize=8.7,
            color="#475569",
            linespacing=1.35,
            transform=ax.transAxes,
            va="top",
        )
        # 카드 사이에 흐름 화살표 — 마지막 카드 다음엔 생략
        if x != card_x[-1]:
            ax.text(
                x + 0.178,
                0.635,
                "→",
                fontsize=20,
                color="#CBD5E1",
                transform=ax.transAxes,
            )

    # 푸른 강조 배너 — 핵심 수치를 한 줄에 보여줌
    band = FancyBboxPatch(
        (0.035, 0.40),
        0.93,
        0.105,
        boxstyle="round,pad=0.01,rounding_size=0.014",
        linewidth=0,
        facecolor="#2563EB",
        alpha=0.97,
        transform=ax.transAxes,
    )
    ax.add_patch(band)

    # 배너에 들어갈 8개 숫자 지표 (값, 라벨)
    metrics = [
        (f"{n_total:,}", "분석 시설"),
        (f"{matched_150:,}", "150m 화재 매칭"),
        (f"{max_vif:.2f}", "VIF 최대"),
        ("3개", "K-Means 군집"),
        (f"{ols_moran_min:.3f}~{ols_moran_max:.3f}", "OLS 잔차 Moran's I"),
        (f"{best_slm:.3f}", "SLM 최고 fit"),
        (f"{best_gwr:.3f}", "GWR 최고 R²"),
        (f"{best_mgwr:.3f}", "MGWR 최고 R²"),
    ]
    # x좌표를 균등 분할해 8칸으로 배치
    for x, (num, label) in zip(np.linspace(0.065, 0.91, len(metrics)), metrics):
        # 큰 숫자
        ax.text(
            x,
            0.462,
            num,
            fontsize=18,
            color="white",
            weight="bold",
            ha="center",
            transform=ax.transAxes,
        )
        # 라벨
        ax.text(
            x,
            0.426,
            label,
            fontsize=8.5,
            color="#DBEAFE",
            ha="center",
            transform=ax.transAxes,
        )

    # 하단 3칼럼 — 사용 테이블 / 군집화 기준 / 공간통계 기준
    sections = [
        (
            "사용 테이블",
            [
                "기본: cluster3_spatial_pipeline_fire_count_150m_0428/최최최종0428변수테이블.csv",
                "타깃: team_pipeline_scored_dataset.csv의 fire_count_150m을 숙소명+좌표로 결합",
                "좌표: x_5181, y_5181 / 위도, 경도 모두 보유",
            ],
        ),
        (
            "군집화 기준",
            [
                "K-Means K=3, random_state=42, n_init=10 계열 산출물 사용",
                "입력: 승인연도, 소방위험도, 주변건물수, 집중도, 단속위험도, 구조노후도 등",
                "해석: 위치 군집이 아니라 시설 위험요인 조합별 유형",
            ],
        ),
        (
            "공간통계 기준",
            [
                "공간가중치: KNN k=min(12,n-1), row-standardized",
                "진단: OLS 잔차 Moran's I, permutations=199",
                "모델: SLM(공간시차), SEM(공간오차), GWR/MGWR(adaptive bisquare)",
            ],
        ),
    ]
    for x, (title, bullets) in zip([0.045, 0.36, 0.675], sections):
        # 섹션 제목
        ax.text(
            x,
            0.325,
            title,
            fontsize=13,
            color="#111827",
            weight="bold",
            transform=ax.transAxes,
        )
        # 제목 밑줄 강조 라인
        ax.add_patch(
            Rectangle((x, 0.305), 0.055, 0.004, color="#2563EB", transform=ax.transAxes)
        )
        y = 0.275
        # 불릿 본문 — 한 줄씩 아래로
        for bullet in bullets:
            ax.text(
                x,
                y,
                "• " + bullet,
                fontsize=8.9,
                color="#334155",
                transform=ax.transAxes,
                va="top",
            )
            y -= 0.055

    # 하단 주의사항 (계산량 한계로 GWR/MGWR은 표본 추출 사용)
    ax.text(
        0.035,
        0.035,
        "주의: GWR/MGWR은 계산량 때문에 각 군집에서 표본추출(GWR 최대 700, MGWR 최대 220)하여 보조적 공간 비정상성 검증으로 해석",
        fontsize=9.3,
        color="#64748B",
        transform=ax.transAxes,
    )

    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """VIF + 슬라이드 + 노트 텍스트 일괄 생성."""
    setup_font()

    # SRC_DIR에서 100KB 이상의 CSV 중 가장 큰 파일을 베이스로 사용
    data_path = max(
        [p for p in SRC_DIR.glob("*.csv") if p.stat().st_size > 100000],
        key=lambda p: p.stat().st_size,
    )
    df = pd.read_csv(data_path, encoding="utf-8-sig", low_memory=False)
    cols = list(df.columns)
    # 미리 합의된 변수 인덱스 위치 — 컬럼 순서가 안정적이라는 가정 하에 사용
    features = [cols[i] for i in [3, 4, 5, 6, 7, 8, 9, 14, 17, 18]]

    # VIF 산출 + CSV/PNG 저장
    vif_df = compute_vif(df, features)
    vif_csv = OUT_DIR / "vif_check_cluster3_fire_count_150m.csv"
    vif_png = OUT_DIR / "vif_check_cluster3_fire_count_150m.png"
    vif_df.to_csv(vif_csv, index=False, encoding="utf-8-sig")
    save_vif_png(vif_df, len(df), vif_png)

    # 클러스터별 공간모델 요약 — OLS/SLM/GWR/MGWR 행 분리
    summary = pd.read_csv(
        SRC_DIR / "spatial_model_summary_by_cluster.csv", encoding="utf-8-sig"
    )
    ols = summary[summary["model"].eq("OLS")]
    slm = summary[summary["model"].eq("SLM")]
    gwr = summary[summary["model"].eq("GWR")]
    mgwr = summary[summary["model"].eq("MGWR")]

    # 150m 반경 매칭 건수 — 외부 요약 CSV가 있으면 거기서 추출, 없으면 기본값
    radius_csv = OUT_DIR / "fire_count_radius_match_100_150_200_summary.csv"
    matched_150 = 3794
    if radius_csv.exists():
        radius_df = pd.read_csv(radius_csv, encoding="utf-8-sig")
        matched_150 = int(
            radius_df.loc[radius_df["radius_m"].eq(150), "matched_count"].iloc[0]
        )

    # 슬라이드 이미지 생성
    slide_png = OUT_DIR / "ppt_methodology_pipeline_revised_0429.png"
    save_methodology_slide(
        slide_png,
        n_total=len(df),
        matched_150=matched_150,
        max_vif=float(vif_df["vif"].max()),
        ols_moran_min=float(ols["resid_moran_I"].min()),
        ols_moran_max=float(ols["resid_moran_I"].max()),
        best_slm=float(slm["fit"].max()),
        best_gwr=float(gwr["fit"].max()),
        best_mgwr=float(mgwr["fit"].max()),
    )

    # 슬라이드 핵심 문구 노트 (PPT 작성 보조용)
    notes = OUT_DIR / "ppt_methodology_pipeline_revised_0429_notes.txt"
    notes.write_text(
        "\n".join(
            [
                "수정 슬라이드 핵심 문구",
                "- 사용 테이블: 최최최종0428변수테이블.csv (4,246개 숙박시설) + team_pipeline_scored_dataset.csv의 fire_count_150m 결합",
                f"- VIF: 10개 설명변수 기준 최대 VIF {float(vif_df['vif'].max()):.2f}, 기준 5 미만으로 다중공선성 문제 낮음",
                "- 군집화: 표준화한 위험요인 조합 기준 K-Means K=3, 위치 기반 군집이 아님",
                "- 공간통계: KNN k=12 행표준화 공간가중치, OLS 잔차 Moran, SLM/SEM, GWR/MGWR 사용",
                "- GWR/MGWR: 계산량 때문에 군집별 표본상한 GWR 700, MGWR 220 적용",
            ]
        ),
        encoding="utf-8",
    )

    # 산출 경로 출력 + VIF 표 미리보기
    print(vif_png)
    print(vif_csv)
    print(slide_png)
    print(notes)
    print(vif_df.to_string(index=False))


if __name__ == "__main__":
    main()
