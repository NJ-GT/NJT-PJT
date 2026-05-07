# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "0430" / "최종테이블0429.csv"
OUT_DIR = ROOT / "0430"

TARGET_DONGS = ["연남동", "서교동"]
RISK_ORDER = ["저위험군", "중위험군", "고위험군"]
RISK_COLORS = {"저위험군": "#60A5FA", "중위험군": "#FBBF24", "고위험군": "#EF4444"}
MAIN_COLORS = {"연남동": "#2563EB", "서교동": "#F97316"}
DESCRIBE_COLS = [
    "최종위험점수_new",
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
DISPLAY_NAMES = {
    "최종위험점수_new": "최종위험점수",
    "구조노후도": "구조노후도",
    "단속위험도": "단속위험도",
    "도로폭위험도": "도로폭위험도",
    "최근접_소화용수_거리등급": "소화용수 거리등급",
    "소방위험도_점수": "소방위험도",
    "승인연도": "승인연도",
    "연면적": "연면적",
    "집중도": "집중도",
    "주변건물수": "주변건물수",
    "총층수": "총층수",
}


def set_korean_font() -> str:
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
    return "sans-serif"


def make_card(df: pd.DataFrame, dong: str) -> Path:
    part = df[df["동"].eq(dong)].copy()
    desc = (
        part[DESCRIBE_COLS]
        .describe()
        .T[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]
    )
    desc = desc.rename(index=DISPLAY_NAMES)
    table_df = desc.copy()
    table_df["count"] = table_df["count"].map(lambda v: f"{int(v):,}")
    for col in ["mean", "std", "min", "25%", "50%", "75%", "max"]:
        table_df[col] = table_df[col].map(lambda v: f"{v:,.3f}")

    risk_counts = part["cluster_label"].value_counts().reindex(RISK_ORDER, fill_value=0)
    score = part["최종위험점수_new"]
    color = MAIN_COLORS[dong]

    fig = plt.figure(figsize=(14.8, 10.2), dpi=180)
    fig.patch.set_facecolor("#f6f8fb")
    gs = fig.add_gridspec(
        3, 4, height_ratios=[0.55, 1.1, 1.85], hspace=0.42, wspace=0.35
    )

    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis("off")
    ax_title.text(
        0.01,
        0.84,
        f"마포구 {dong} describe",
        fontsize=25,
        fontweight="bold",
        color="#101828",
        va="top",
    )
    ax_title.text(
        0.01,
        0.40,
        "최종테이블0429 기준 | 주요 위험 변수의 분포 요약",
        fontsize=12.5,
        color="#667085",
        va="top",
    )
    ax_title.text(
        0.67,
        0.80,
        f"시설 수 {len(part):,}개\n평균위험도 {score.mean():.2f}점\n중앙값 {score.median():.2f}점\n최고위험도 {score.max():.2f}점",
        fontsize=12.2,
        color="#1d2939",
        va="top",
        bbox=dict(
            boxstyle="round,pad=0.55,rounding_size=0.12",
            facecolor="#ffffff",
            edgecolor="#d0d5dd",
        ),
    )

    ax_pie = fig.add_subplot(gs[1, 0])
    wedges, _, autotexts = ax_pie.pie(
        risk_counts.values,
        colors=[RISK_COLORS[k] for k in RISK_ORDER],
        startangle=92,
        autopct=lambda pct: f"{pct:.0f}%" if pct > 5 else "",
        wedgeprops=dict(edgecolor="white", linewidth=1.2),
        textprops=dict(color="#111827", fontsize=10, fontweight="bold"),
    )
    ax_pie.set_title("위험군 비율", fontsize=14, fontweight="bold")
    ax_pie.legend(
        wedges,
        [f"{k} {int(risk_counts[k])}개" for k in RISK_ORDER],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.25),
        ncol=1,
        fontsize=9,
    )

    ax_hist = fig.add_subplot(gs[1, 1:3])
    sns.histplot(
        part["최종위험점수_new"],
        bins=18,
        kde=True,
        color=color,
        edgecolor="white",
        alpha=0.82,
        ax=ax_hist,
    )
    ax_hist.axvline(
        score.mean(),
        color="#111827",
        linewidth=1.5,
        linestyle="--",
        label=f"평균 {score.mean():.2f}",
    )
    ax_hist.axvline(
        score.median(),
        color="#475467",
        linewidth=1.5,
        linestyle=":",
        label=f"중앙 {score.median():.2f}",
    )
    ax_hist.set_title("최종위험점수 분포", fontsize=14, fontweight="bold")
    ax_hist.set_xlabel("최종위험점수_new")
    ax_hist.set_ylabel("시설 수")
    ax_hist.legend(fontsize=9)
    ax_hist.set_facecolor("#ffffff")

    ax_bar = fig.add_subplot(gs[1, 3])
    top_means = part[
        ["구조노후도", "단속위험도", "도로폭위험도", "소방위험도_점수"]
    ].mean()
    bars = ax_bar.barh(
        [DISPLAY_NAMES[i] for i in top_means.index],
        top_means.values,
        color=["#7DD3FC", "#FDBA74", "#A7F3D0", "#FCA5A5"],
        edgecolor="white",
    )
    for bar, val in zip(bars, top_means.values):
        ax_bar.text(
            val + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}",
            va="center",
            fontsize=9.5,
        )
    ax_bar.set_title("핵심 위험 변수 평균", fontsize=14, fontweight="bold")
    ax_bar.set_xlabel("평균")
    ax_bar.set_facecolor("#ffffff")

    ax_table = fig.add_subplot(gs[2, :])
    ax_table.axis("off")
    table = ax_table.table(
        cellText=table_df.reset_index().values,
        colLabels=["변수", "count", "mean", "std", "min", "25%", "50%", "75%", "max"],
        cellLoc="center",
        loc="center",
        colColours=["#e7eefb"] * 9,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.4)
    table.scale(1, 1.36)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#d0d5dd")
        if row == 0:
            cell.set_text_props(weight="bold", color="#101828")
        elif col == 0:
            cell.set_text_props(weight="bold", color="#1d2939")
            cell.set_facecolor("#f8fafc")
        elif row % 2 == 0:
            cell.set_facecolor("#fbfdff")
    ax_table.set_title("주요 변수 describe", fontsize=16, fontweight="bold", pad=12)

    fig.text(
        0.018,
        0.018,
        "box/describe 값은 원본 지표 기준입니다. 위험점수 산출 단계에서는 변수 정규화와 가중치가 별도로 적용됩니다.",
        fontsize=10.5,
        color="#667085",
    )

    out = OUT_DIR / f"마포구_{dong}_describe.png"
    fig.savefig(out, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


def main() -> None:
    font_name = set_korean_font()
    sns.set_theme(style="whitegrid", rc={"font.family": font_name})
    plt.rcParams["font.family"] = font_name
    plt.rcParams["axes.unicode_minus"] = False

    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    df = df[df["구"].eq("마포구") & df["동"].isin(TARGET_DONGS)].copy()
    for col in DESCRIBE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for dong in TARGET_DONGS:
        print(make_card(df, dong))


if __name__ == "__main__":
    main()
