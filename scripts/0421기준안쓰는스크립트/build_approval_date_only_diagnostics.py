# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

SOURCE = DATA_DIR / "분석변수_테이블_주변노후도_50m100m.csv"
OUT_DATA = DATA_DIR / "분석변수_테이블_주변노후도_50m100m_사용승인일기준_전체행.csv"
OUT_VIF = DATA_DIR / "주변노후건물_50m100m_사용승인일기준_전체행_vif.csv"
OUT_SUMMARY = DATA_DIR / "주변노후건물_50m100m_사용승인일기준_전체행_요약.png"
OUT_VIF_PNG = DATA_DIR / "주변노후건물_50m100m_사용승인일기준_전체행_vif.png"
OUT_CORR = DATA_DIR / "주변노후건물_50m100m_사용승인일기준_전체행_corr.png"

RATIO_COLUMNS = [
    "주변_노후건물비율_30년이상_50m",
    "주변_노후건물비율_30년이상_100m",
    "주변_초노후건물비율_50년이상_50m",
    "주변_초노후건물비율_50년이상_100m",
]

VIF_COLUMNS = [
    "소방접근성_점수",
    "노후도_점수",
    "반경_50m_건물수",
    "집중도(%)",
    "로그_주변대비_상대위험도_고유단속지점_50m",
    "공식도로폭m",
    "주변_노후건물비율_30년이상_50m",
    "주변_노후건물비율_30년이상_100m",
]

CORR_COLUMNS = [
    "주변건물수_50m",
    "주변건물수_100m",
    "주변_사용승인일유효건물수_50m",
    "주변_사용승인일유효건물수_100m",
    "주변_노후건물수_30년이상_50m",
    "주변_노후건물수_30년이상_100m",
    "주변_노후건물비율_30년이상_50m",
    "주변_노후건물비율_30년이상_100m",
    "주변_초노후건물비율_50년이상_50m",
    "주변_초노후건물비율_50년이상_100m",
]

SHORT_LABELS = {
    "소방접근성_점수": "소방접근성",
    "노후도_점수": "자체 노후도",
    "반경_50m_건물수": "기존 건물수 50m",
    "집중도(%)": "집중도",
    "로그_주변대비_상대위험도_고유단속지점_50m": "단속 로그위험",
    "공식도로폭m": "공식도로폭",
    "주변건물수_50m": "건물수 50m",
    "주변건물수_100m": "건물수 100m",
    "주변_사용승인일유효건물수_50m": "승인일 유효수 50m",
    "주변_사용승인일유효건물수_100m": "승인일 유효수 100m",
    "주변_노후건물수_30년이상_50m": "30년+ 수 50m",
    "주변_노후건물수_30년이상_100m": "30년+ 수 100m",
    "주변_노후건물비율_30년이상_50m": "30년+ 비율 50m",
    "주변_노후건물비율_30년이상_100m": "30년+ 비율 100m",
    "주변_초노후건물비율_50년이상_50m": "50년+ 비율 50m",
    "주변_초노후건물비율_50년이상_100m": "50년+ 비율 100m",
}


def setup_font() -> None:
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False


def judgement(value: float) -> str:
    if value > 10:
        return "높음(>10)"
    if value > 5:
        return "주의(5~10)"
    return "양호"


def calculate_vif(df: pd.DataFrame) -> pd.DataFrame:
    data = df[VIF_COLUMNS].replace([np.inf, -np.inf], np.nan).dropna()
    x = (data - data.mean()) / data.std(ddof=0)
    x.insert(0, "const", 1.0)
    out = pd.DataFrame(
        {
            "변수": VIF_COLUMNS,
            "VIF": [variance_inflation_factor(x.values, i + 1) for i in range(len(VIF_COLUMNS))],
            "사용행수": len(x),
        }
    ).sort_values("VIF", ascending=False)
    out["판정"] = out["VIF"].map(judgement)
    return out[["변수", "VIF", "판정", "사용행수"]]


def draw_summary(df: pd.DataFrame, raw_missing_50m: int, raw_missing_100m: int) -> None:
    labels = ["전체 사용행", "50m 유효주변 0건", "100m 유효주변 0건"]
    values = [len(df), raw_missing_50m, raw_missing_100m]
    colors = ["#059669", "#f59e0b", "#f97316"]

    fig, ax = plt.subplots(figsize=(11.5, 6.4))
    bars = ax.bar(labels, values, color=colors, width=0.55)
    ax.set_title("사용승인일 기준 주변 노후건물 변수: 전체행 유지", fontsize=15, pad=16)
    ax.set_ylabel("행 수")
    ax.set_ylim(0, max(values) * 1.22)
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
        0.86,
        "주변 유효건물이 0개인 행은 제외하지 않고 30년+/50년+ 비율을 0으로 처리",
        transform=ax.transAxes,
        ha="center",
        fontsize=12,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#f8fafc", "edgecolor": "#cbd5e1"},
    )
    ax.text(
        0.5,
        0.77,
        "대상 숙박시설 사용승인일 결측 때문이 아니라, 주변 인접건물의 유효 승인일이 없는 경우를 표시한 것",
        transform=ax.transAxes,
        ha="center",
        fontsize=11,
        color="#374151",
    )

    plt.tight_layout()
    plt.savefig(OUT_SUMMARY, dpi=180, bbox_inches="tight")
    plt.close()


def draw_vif(vif: pd.DataFrame) -> None:
    display = vif.copy()
    display["변수"] = display["변수"].map(lambda value: SHORT_LABELS.get(value, value))
    display["VIF"] = display["VIF"].map(lambda value: f"{float(value):.2f}")

    fig, ax = plt.subplots(figsize=(11.5, 5.5))
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
    table.scale(1.0, 1.4)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#d1d5db")
        if row == 0:
            cell.set_facecolor("#111827")
            cell.set_text_props(color="white", weight="bold")
        elif "주의" in str(display.iloc[row - 1]["판정"]):
            cell.set_facecolor("#fef3c7")
        elif "높음" in str(display.iloc[row - 1]["판정"]):
            cell.set_facecolor("#fee2e2")
        else:
            cell.set_facecolor("#ecfdf5")
    ax.set_title("사용승인일 기준 전체행 VIF", fontsize=15, pad=16)
    plt.tight_layout()
    plt.savefig(OUT_VIF_PNG, dpi=180, bbox_inches="tight")
    plt.close()


def draw_corr(df: pd.DataFrame) -> None:
    data = df[CORR_COLUMNS].rename(columns=SHORT_LABELS)
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
    ax.set_title("사용승인일 기준 주변 노후건물 변수 상관 히트맵", fontsize=15, pad=16)
    ax.tick_params(axis="x", labelrotation=45)
    ax.tick_params(axis="y", labelrotation=0)
    plt.tight_layout()
    plt.savefig(OUT_CORR, dpi=180, bbox_inches="tight")
    plt.close()


def main() -> None:
    setup_font()
    df = pd.read_csv(SOURCE, encoding="utf-8-sig")

    raw_missing_50m = int(df["주변_노후건물비율_30년이상_50m"].isna().sum())
    raw_missing_100m = int(df["주변_노후건물비율_30년이상_100m"].isna().sum())

    df[RATIO_COLUMNS] = df[RATIO_COLUMNS].fillna(0.0)
    df.to_csv(OUT_DATA, index=False, encoding="utf-8-sig")

    vif = calculate_vif(df)
    vif.to_csv(OUT_VIF, index=False, encoding="utf-8-sig")

    draw_summary(df, raw_missing_50m, raw_missing_100m)
    draw_vif(vif)
    draw_corr(df)

    print(OUT_DATA)
    print(OUT_VIF)
    print(OUT_SUMMARY)
    print(OUT_VIF_PNG)
    print(OUT_CORR)


if __name__ == "__main__":
    main()
