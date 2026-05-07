# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

FEATURE_PATH = DATA_DIR / "분석변수_테이블_주변노후도_50m100m.csv"
FEATURE_VALID_PATH = DATA_DIR / "분석변수_테이블_주변노후도_50m100m_연면적유효.csv"
VIF_CORE_PATH = DATA_DIR / "주변노후건물_50m100m_vif_핵심후보.csv"
VIF_RECOMMENDED_PATH = DATA_DIR / "주변노후건물_50m100m_vif_추천변수.csv"

OUT_CORR_HEATMAP = DATA_DIR / "주변노후건물_50m100m_corr_heatmap.png"
OUT_VIF_CORE = DATA_DIR / "주변노후건물_50m100m_vif_핵심후보.png"
OUT_VIF_RECOMMENDED = DATA_DIR / "주변노후건물_50m100m_vif_추천변수.png"
OUT_VALID_SUMMARY = DATA_DIR / "주변노후건물_50m100m_연면적유효_요약.png"
OUT_VALID_VIF = DATA_DIR / "주변노후건물_50m100m_연면적유효_vif_추천변수.png"

CORE_COLUMNS = [
    "주변건물수_50m",
    "주변건물수_100m",
    "주변_노후건물비율_30년이상_50m",
    "주변_노후건물비율_30년이상_100m",
    "주변_초노후건물비율_50년이상_50m",
    "주변_초노후건물비율_50년이상_100m",
    "주변_평균건물연한_50m",
    "주변_평균건물연한_100m",
    "주변_노후연면적비율_30년이상_50m",
    "주변_노후연면적비율_30년이상_100m",
]

SHORT_LABELS = {
    "주변건물수_50m": "건물수 50m",
    "주변건물수_100m": "건물수 100m",
    "주변_노후건물비율_30년이상_50m": "30년+ 비율 50m",
    "주변_노후건물비율_30년이상_100m": "30년+ 비율 100m",
    "주변_초노후건물비율_50년이상_50m": "50년+ 비율 50m",
    "주변_초노후건물비율_50년이상_100m": "50년+ 비율 100m",
    "주변_평균건물연한_50m": "평균연한 50m",
    "주변_평균건물연한_100m": "평균연한 100m",
    "주변_노후연면적비율_30년이상_50m": "30년+ 연면적비 50m",
    "주변_노후연면적비율_30년이상_100m": "30년+ 연면적비 100m",
    "소방접근성_점수": "소방접근성",
    "노후도_점수": "자체 노후도",
    "반경_50m_건물수": "기존 건물수 50m",
    "집중도(%)": "집중도",
    "로그_주변대비_상대위험도_고유단속지점_50m": "단속 로그위험",
    "공식도로폭m": "공식도로폭",
}


def setup_font() -> None:
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False


def draw_corr_heatmap() -> None:
    df = pd.read_csv(FEATURE_PATH, encoding="utf-8-sig")
    data = df[CORE_COLUMNS].rename(columns=SHORT_LABELS)
    corr = data.corr(numeric_only=True)

    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(
        corr,
        vmin=-1,
        vmax=1,
        center=0,
        cmap="RdBu_r",
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        linecolor="#e5e7eb",
        square=True,
        cbar_kws={"shrink": 0.75, "label": "Pearson r"},
        annot_kws={"fontsize": 9},
    )
    ax.set_title("주변 노후건물 50m/100m 핵심 후보 상관 히트맵", fontsize=15, pad=16)
    ax.tick_params(axis="x", labelrotation=45)
    ax.tick_params(axis="y", labelrotation=0)
    plt.tight_layout()
    plt.savefig(OUT_CORR_HEATMAP, dpi=180, bbox_inches="tight")
    plt.close()


def draw_vif_table(csv_path: Path, out_path: Path, title: str) -> None:
    vif = pd.read_csv(csv_path, encoding="utf-8-sig")
    vif = vif.copy()
    vif["변수"] = vif["변수"].map(lambda value: SHORT_LABELS.get(value, value))
    vif["VIF"] = vif["VIF"].map(lambda value: "inf" if pd.isna(value) else f"{float(value):.2f}")
    display = vif[["변수", "VIF", "판정", "사용행수"]]

    row_count = len(display)
    fig_height = max(4.8, 0.45 * row_count + 1.8)
    fig, ax = plt.subplots(figsize=(11.5, fig_height))
    ax.axis("off")
    table = ax.table(
        cellText=display.values,
        colLabels=display.columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
        colWidths=[0.46, 0.12, 0.22, 0.12],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.35)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#d1d5db")
        if row == 0:
            cell.set_facecolor("#111827")
            cell.set_text_props(color="white", weight="bold")
        elif "높음" in str(display.iloc[row - 1]["판정"]) or "무한대" in str(display.iloc[row - 1]["판정"]):
            cell.set_facecolor("#fee2e2")
        elif "주의" in str(display.iloc[row - 1]["판정"]):
            cell.set_facecolor("#fef3c7")
        else:
            cell.set_facecolor("#ecfdf5")

    ax.set_title(title, fontsize=15, pad=16)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def draw_floor_area_valid_summary() -> None:
    full = pd.read_csv(FEATURE_PATH, encoding="utf-8-sig")
    valid = pd.read_csv(FEATURE_VALID_PATH, encoding="utf-8-sig")
    excluded = len(full) - len(valid)
    excluded_pct = excluded / len(full) * 100 if len(full) else 0

    values = [len(valid), excluded]
    labels = ["연면적 유효", "연면적 계산 불가 제외"]
    colors = ["#059669", "#dc2626"]

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    bars = ax.bar(labels, values, color=colors, width=0.55)
    ax.set_title("주변 노후연면적비율 모델용 데이터 필터링 결과", fontsize=15, pad=16)
    ax.set_ylabel("행 수")
    ax.set_ylim(0, max(values) * 1.25)
    ax.grid(axis="y", alpha=0.25)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.03,
            f"{value:,}행",
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
        )

    ax.text(
        0.5,
        0.88,
        f"전체 {len(full):,}행 중 {len(valid):,}행 사용, {excluded:,}행 제외 ({excluded_pct:.1f}%)",
        transform=ax.transAxes,
        ha="center",
        fontsize=13,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#f8fafc", "edgecolor": "#cbd5e1"},
    )
    ax.text(
        0.5,
        0.78,
        "제외 기준: 50m와 100m의 30년 이상 노후연면적비율이 모두 계산 가능해야 함",
        transform=ax.transAxes,
        ha="center",
        fontsize=11,
        color="#374151",
    )

    plt.tight_layout()
    plt.savefig(OUT_VALID_SUMMARY, dpi=180, bbox_inches="tight")
    plt.close()


def main() -> None:
    setup_font()
    draw_corr_heatmap()
    draw_vif_table(VIF_CORE_PATH, OUT_VIF_CORE, "주변 노후건물 핵심 후보 VIF")
    draw_vif_table(VIF_RECOMMENDED_PATH, OUT_VIF_RECOMMENDED, "추천 변수 조합 VIF")
    draw_floor_area_valid_summary()
    draw_vif_table(VIF_RECOMMENDED_PATH, OUT_VALID_VIF, "연면적 유효 데이터 기준 추천 변수 VIF")
    print(OUT_CORR_HEATMAP)
    print(OUT_VIF_CORE)
    print(OUT_VIF_RECOMMENDED)
    print(OUT_VALID_SUMMARY)
    print(OUT_VALID_VIF)


if __name__ == "__main__":
    main()
