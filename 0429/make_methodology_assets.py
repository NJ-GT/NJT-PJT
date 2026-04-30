# -*- coding: utf-8 -*-
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Rectangle
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


BASE = Path(__file__).resolve().parents[2]
OUT_DIR = BASE / "NJT-PJT" / "0429"
SRC_DIR = BASE / "NJT-PJT" / "0424" / "data" / "cluster3_spatial_pipeline_fire_count_150m_0428"


def setup_font() -> None:
    font_path = Path("C:/Windows/Fonts/malgun.ttf")
    if font_path.exists():
        fm.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = fm.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams["axes.unicode_minus"] = False


def compute_vif(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    work = df[features].copy()
    for col in features:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna(subset=features).reset_index(drop=True)

    x_scaled = StandardScaler().fit_transform(work[features].to_numpy(dtype=float))
    rows = []
    for j, col in enumerate(features):
        y = x_scaled[:, j]
        x_other = np.delete(x_scaled, j, axis=1)
        r2 = LinearRegression().fit(x_other, y).score(x_other, y)
        vif = np.inf if r2 >= 0.999999 else 1.0 / (1.0 - r2)
        rows.append({"variable": col, "vif": float(vif)})
    return pd.DataFrame(rows).sort_values("vif", ascending=False).reset_index(drop=True)


def save_vif_png(vif_df: pd.DataFrame, n_rows: int, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 7), dpi=180)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    fig.subplots_adjust(top=0.82, left=0.22, right=0.96, bottom=0.12)

    plot_df = vif_df.sort_values("vif", ascending=True)
    colors = ["#2563EB" if v < 5 else "#F97316" if v < 10 else "#DC2626" for v in plot_df["vif"]]
    ax.barh(plot_df["variable"], plot_df["vif"], color=colors, height=0.62)
    ax.axvline(5, color="#F59E0B", linewidth=1.4, linestyle="--")
    ax.axvline(10, color="#DC2626", linewidth=1.4, linestyle="--")
    ax.text(5.05, len(plot_df) - 0.55, "주의 기준 5", color="#92400E", fontsize=9)
    ax.text(10.05, len(plot_df) - 0.55, "심각 기준 10", color="#991B1B", fontsize=9)

    for y, v in enumerate(plot_df["vif"]):
        ax.text(v + 0.03, y, f"{v:.2f}", va="center", fontsize=10, color="#111827", weight="bold")

    max_vif = float(vif_df["vif"].max())
    ax.set_xlim(0, max(10.8, max_vif + 0.8))
    ax.set_xlabel("VIF", fontsize=11, color="#374151")
    fig.text(0.22, 0.94, "다중공선성 점검: VIF", fontsize=18, weight="bold", color="#111827")
    fig.text(
        0.22,
        0.905,
        f"기준 테이블: 최최최종0428변수테이블.csv · 설명변수 10개 · n={n_rows:,} · 최대 VIF={max_vif:.2f}",
        fontsize=10.5,
        color="#4B5563",
    )
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
    fig = plt.figure(figsize=(16, 9), dpi=180)
    fig.patch.set_facecolor("#F8FAFC")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    ax.text(
        0.035,
        0.93,
        "분석 방법론",
        fontsize=12,
        color="#2563EB",
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.35", fc="#EFF6FF", ec="#BFDBFE"),
    )
    ax.text(
        0.035,
        0.84,
        "데이터 → 변수검증 → 군집화 → 공간통계 → 정책 제안 파이프라인",
        fontsize=27,
        weight="bold",
        color="#111827",
    )
    ax.add_patch(Rectangle((0.035, 0.79), 0.045, 0.006, color="#2563EB", transform=ax.transAxes))

    cards = [
        ("STEP 1", "분석 테이블 확정", "최최최종0428변수테이블.csv\n4,246개 숙박시설 · 10개 설명변수\n외부 매칭: fire_count_150m"),
        ("STEP 2", "타깃/반경 선택", "100m 3,209개 / 150m 3,794개 / 200m 4,049개\n150m = 매칭률 89.4% + 변별력 균형"),
        ("STEP 3", "VIF 다중공선성 점검", f"승인연도·소방위험도·주변건물수 등 10변수\n최대 VIF {max_vif:.2f} · 기준 5 미만"),
        ("STEP 4", "K-Means 군집화", "표준화 후 K=3 군집 사용\ncluster 0/1/2 = 시설 위험요인 조합별 유형"),
        ("STEP 5", "공간통계·공간회귀", "KNN k=12 row-standardized\nOLS+Moran → SLM/SEM → GWR/MGWR"),
    ]
    card_x = [0.035, 0.225, 0.415, 0.605, 0.795]
    for x, (step, title, body) in zip(card_x, cards):
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
        ax.text(x + 0.012, 0.705, step, fontsize=8.5, color="#2563EB", weight="bold", transform=ax.transAxes)
        ax.text(x + 0.012, 0.675, title, fontsize=11.5, color="#111827", weight="bold", transform=ax.transAxes)
        ax.text(x + 0.012, 0.635, body, fontsize=8.7, color="#475569", linespacing=1.35, transform=ax.transAxes, va="top")
        if x != card_x[-1]:
            ax.text(x + 0.178, 0.635, "→", fontsize=20, color="#CBD5E1", transform=ax.transAxes)

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
    for x, (num, label) in zip(np.linspace(0.065, 0.91, len(metrics)), metrics):
        ax.text(x, 0.462, num, fontsize=18, color="white", weight="bold", ha="center", transform=ax.transAxes)
        ax.text(x, 0.426, label, fontsize=8.5, color="#DBEAFE", ha="center", transform=ax.transAxes)

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
        ax.text(x, 0.325, title, fontsize=13, color="#111827", weight="bold", transform=ax.transAxes)
        ax.add_patch(Rectangle((x, 0.305), 0.055, 0.004, color="#2563EB", transform=ax.transAxes))
        y = 0.275
        for bullet in bullets:
            ax.text(x, y, "• " + bullet, fontsize=8.9, color="#334155", transform=ax.transAxes, va="top")
            y -= 0.055

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
    setup_font()

    data_path = max([p for p in SRC_DIR.glob("*.csv") if p.stat().st_size > 100000], key=lambda p: p.stat().st_size)
    df = pd.read_csv(data_path, encoding="utf-8-sig", low_memory=False)
    cols = list(df.columns)
    features = [cols[i] for i in [3, 4, 5, 6, 7, 8, 9, 14, 17, 18]]

    vif_df = compute_vif(df, features)
    vif_csv = OUT_DIR / "vif_check_cluster3_fire_count_150m.csv"
    vif_png = OUT_DIR / "vif_check_cluster3_fire_count_150m.png"
    vif_df.to_csv(vif_csv, index=False, encoding="utf-8-sig")
    save_vif_png(vif_df, len(df), vif_png)

    summary = pd.read_csv(SRC_DIR / "spatial_model_summary_by_cluster.csv", encoding="utf-8-sig")
    ols = summary[summary["model"].eq("OLS")]
    slm = summary[summary["model"].eq("SLM")]
    gwr = summary[summary["model"].eq("GWR")]
    mgwr = summary[summary["model"].eq("MGWR")]

    radius_csv = OUT_DIR / "fire_count_radius_match_100_150_200_summary.csv"
    matched_150 = 3794
    if radius_csv.exists():
        radius_df = pd.read_csv(radius_csv, encoding="utf-8-sig")
        matched_150 = int(radius_df.loc[radius_df["radius_m"].eq(150), "matched_count"].iloc[0])

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

    print(vif_png)
    print(vif_csv)
    print(slide_png)
    print(notes)
    print(vif_df.to_string(index=False))


if __name__ == "__main__":
    main()
