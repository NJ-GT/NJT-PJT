# -*- coding: utf-8 -*-
"""Visualize injured-fire origin hierarchy as a polished static PNG."""

from __future__ import annotations

import math
from pathlib import Path
from textwrap import shorten

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


SCRIPT_DIR = Path(__file__).resolve().parent
APRIL_DIR = SCRIPT_DIR.parent
PROJECT_DIR = APRIL_DIR.parent
SOURCE_CSV = PROJECT_DIR / "data" / "화재출동" / "화재출동_2021_2024.csv"
OUT_DIR = APRIL_DIR / "data" / "fire_dispatch_viz"
OUT_PATH = OUT_DIR / "07_injury_fire_origin_hierarchy.png"

COL_L1 = "발화장소_대분류"
COL_L2 = "발화장소_중분류"
COL_L3 = "발화장소_소분류"
INJURY_COL = "인명피해계"
FIG_DPI = 220

BASE_COLORS = [
    "#0f766e",
    "#2563eb",
    "#dc2626",
    "#7c3aed",
    "#ea580c",
    "#0891b2",
    "#65a30d",
    "#be185d",
    "#4f46e5",
    "#b45309",
    "#374151",
    "#0f172a",
]


def setup_style() -> None:
    font_names = {f.name for f in fm.fontManager.ttflist}
    selected_font = "Malgun Gothic"
    for font in ("Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR"):
        if font in font_names:
            selected_font = font
            break

    sns.set_theme(style="white")
    plt.rcParams["font.family"] = selected_font
    plt.rcParams["font.sans-serif"] = [selected_font, "DejaVu Sans", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "#fcfcfb"
    plt.rcParams["axes.facecolor"] = "#fcfcfb"
    plt.rcParams["savefig.facecolor"] = "#fcfcfb"


def read_csv_robust(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "cp949", "euc-kr", "utf-8"):
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, low_memory=False)


def clean_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = [str(col).strip() for col in cleaned.columns]
    for col in cleaned.select_dtypes(include=["object"]).columns:
        cleaned[col] = (
            cleaned[col]
            .astype("string")
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
            .replace("", pd.NA)
        )
    return cleaned


def prepare_data(path: Path) -> pd.DataFrame:
    df = clean_text_columns(read_csv_robust(path))
    df[INJURY_COL] = pd.to_numeric(df[INJURY_COL], errors="coerce")
    filtered = df[df[INJURY_COL] >= 1].copy()

    for col in (COL_L1, COL_L2, COL_L3):
        filtered[col] = filtered[col].fillna("미상")

    return filtered.reset_index(drop=True)


def blend_with_white(color: str, blend: float) -> tuple[float, float, float]:
    rgb = np.array(mcolors.to_rgb(color))
    return tuple(rgb + (1 - rgb) * blend)


def build_hierarchy(df: pd.DataFrame) -> dict[str, object]:
    level1 = (
        df.groupby(COL_L1)
        .size()
        .sort_values(ascending=False)
        .rename("count")
        .reset_index()
    )
    level1["share"] = level1["count"] / level1["count"].sum()
    level1["base_color"] = BASE_COLORS[: len(level1)]
    l1_color_map = dict(zip(level1[COL_L1], level1["base_color"]))

    level2_rows: list[dict[str, object]] = []
    level3_rows: list[dict[str, object]] = []

    for l1_name in level1[COL_L1]:
        level2 = (
            df[df[COL_L1] == l1_name]
            .groupby(COL_L2)
            .size()
            .sort_values(ascending=False)
            .rename("count")
            .reset_index()
        )

        for idx2, row2 in level2.reset_index(drop=True).iterrows():
            child_color = blend_with_white(l1_color_map[l1_name], min(0.18 + idx2 * 0.08, 0.62))
            level2_rows.append(
                {
                    "l1": l1_name,
                    "l2": row2[COL_L2],
                    "count": int(row2["count"]),
                    "share": float(row2["count"]) / len(df),
                    "color": child_color,
                }
            )

            level3 = (
                df[(df[COL_L1] == l1_name) & (df[COL_L2] == row2[COL_L2])]
                .groupby(COL_L3)
                .size()
                .sort_values(ascending=False)
                .rename("count")
                .reset_index()
            )
            for idx3, row3 in level3.reset_index(drop=True).iterrows():
                leaf_color = blend_with_white(
                    mcolors.to_hex(child_color),
                    min(0.10 + idx3 * 0.04, 0.55),
                )
                level3_rows.append(
                    {
                        "l1": l1_name,
                        "l2": row2[COL_L2],
                        "l3": row3[COL_L3],
                        "count": int(row3["count"]),
                        "share": float(row3["count"]) / len(df),
                        "color": leaf_color,
                    }
                )

    level2_df = pd.DataFrame(level2_rows)
    level3_df = pd.DataFrame(level3_rows)
    return {
        "total": len(df),
        "level1": level1,
        "level2": level2_df,
        "level3": level3_df,
    }


def format_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def short_label(text: str, width: int = 22) -> str:
    return shorten(str(text), width=width, placeholder="…")


def add_ring_annotations(
    ax: plt.Axes,
    wedges: list,
    labels: list[str],
    values: list[int],
    total: int,
    radius: float,
    threshold: float,
    font_size: float,
) -> None:
    for wedge, label, value in zip(wedges, labels, values):
        share = value / total
        if share < threshold:
            continue

        angle = (wedge.theta1 + wedge.theta2) / 2
        angle_rad = math.radians(angle)
        x = math.cos(angle_rad) * radius
        y = math.sin(angle_rad) * radius
        text = f"{short_label(label, 20)}\n{share * 100:.1f}%"
        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=font_size,
            color="#0f172a",
            fontweight="bold" if share >= threshold * 1.8 else "normal",
        )


def draw_summary_panel(
    ax: plt.Axes,
    summary_df: pd.DataFrame,
    label_col: str,
    count_col: str,
    title: str,
    bar_color: str,
    total: int,
    top_n: int,
) -> None:
    top = summary_df.nlargest(top_n, count_col).iloc[::-1].copy()
    top["share"] = top[count_col] / total
    y = np.arange(len(top))
    ax.barh(y, top["share"], color=bar_color, height=0.66, alpha=0.92)
    ax.set_yticks(y)
    ax.set_yticklabels([short_label(v, 28) for v in top[label_col]], fontsize=10)
    ax.set_xlim(0, max(top["share"].max() * 1.18, 0.12))
    ax.tick_params(axis="x", labelsize=9, colors="#6b7280")
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#e5e7eb", linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.set_title(title, loc="left", fontsize=12.5, fontweight="bold", pad=10)

    for idx, (_, row) in enumerate(top.iterrows()):
        ax.text(
            row["share"] + 0.004,
            idx,
            f"{row[count_col]:,}건  {row['share'] * 100:.1f}%",
            va="center",
            ha="left",
            fontsize=9.5,
            color="#334155",
        )


def create_figure(data: dict[str, object], out_path: Path) -> Path:
    total = int(data["total"])
    level1 = data["level1"]
    level2 = data["level2"]
    level3 = data["level3"]

    fig = plt.figure(figsize=(18, 10), dpi=FIG_DPI)
    gs = fig.add_gridspec(
        nrows=3,
        ncols=2,
        width_ratios=[1.55, 1],
        height_ratios=[1, 1, 1],
        wspace=0.08,
        hspace=0.32,
    )

    ax_main = fig.add_subplot(gs[:, 0])
    ax_l1 = fig.add_subplot(gs[0, 1])
    ax_l2 = fig.add_subplot(gs[1, 1])
    ax_l3 = fig.add_subplot(gs[2, 1])

    fig.text(
        0.055,
        0.955,
        "인명피해 화재의 발화장소 계층 비율",
        fontsize=24,
        fontweight="bold",
        color="#0f172a",
        ha="left",
    )
    fig.text(
        0.055,
        0.918,
        "대분류 → 중분류 → 소분류 순서로 구성한 3중 도넛. 오른쪽 패널은 각 레벨의 상위 비중을 정리했습니다.",
        fontsize=11.5,
        color="#475569",
        ha="left",
    )
    fig.text(
        0.055,
        0.892,
        f"대상: 2021~2024 화재출동 데이터 중 인명피해계 1명 이상 · 총 {total:,}건",
        fontsize=10.8,
        color="#64748b",
        ha="left",
    )

    radius1, width = 1.06, 0.23
    radius2 = radius1 + width
    radius3 = radius2 + width

    values1 = level1["count"].tolist()
    labels1 = level1[COL_L1].tolist()
    colors1 = level1["base_color"].tolist()

    values2 = level2["count"].tolist()
    labels2 = level2["l2"].tolist()
    colors2 = level2["color"].tolist()

    values3 = level3["count"].tolist()
    labels3 = level3["l3"].tolist()
    colors3 = level3["color"].tolist()

    wedges1, _ = ax_main.pie(
        values1,
        radius=radius1,
        startangle=90,
        colors=colors1,
        labels=None,
        wedgeprops=dict(width=width, edgecolor="#fcfcfb", linewidth=2),
    )
    wedges2, _ = ax_main.pie(
        values2,
        radius=radius2,
        startangle=90,
        colors=colors2,
        labels=None,
        wedgeprops=dict(width=width, edgecolor="#fcfcfb", linewidth=1.4),
    )
    wedges3, _ = ax_main.pie(
        values3,
        radius=radius3,
        startangle=90,
        colors=colors3,
        labels=None,
        wedgeprops=dict(width=width, edgecolor="#fcfcfb", linewidth=0.8),
    )

    add_ring_annotations(ax_main, wedges1, labels1, values1, total, radius1 - width / 2, 0.045, 10.8)
    add_ring_annotations(ax_main, wedges2, labels2, values2, total, radius2 - width / 2, 0.055, 8.3)
    add_ring_annotations(ax_main, wedges3, labels3, values3, total, radius3 + 0.10, 0.028, 7.5)

    ax_main.text(
        0,
        0.06,
        f"{total:,}건",
        ha="center",
        va="center",
        fontsize=25,
        fontweight="bold",
        color="#0f172a",
    )
    ax_main.text(
        0,
        -0.11,
        "인명피해 화재",
        ha="center",
        va="center",
        fontsize=11.5,
        color="#475569",
    )
    ax_main.text(
        0,
        -0.27,
        "Inner  대분류\nMiddle  중분류\nOuter  소분류",
        ha="center",
        va="center",
        fontsize=10,
        color="#64748b",
        linespacing=1.5,
    )
    ax_main.set(aspect="equal")
    ax_main.set_xlim(-1.7, 1.7)
    ax_main.set_ylim(-1.55, 1.55)
    ax_main.axis("off")

    draw_summary_panel(
        ax_l1,
        level1[[COL_L1, "count"]].rename(columns={COL_L1: "label"}),
        "label",
        "count",
        "대분류 상위 비중",
        "#0f766e",
        total,
        top_n=min(8, len(level1)),
    )
    draw_summary_panel(
        ax_l2,
        level2[["l2", "count"]]
        .groupby("l2", as_index=False)
        .sum()
        .rename(columns={"l2": "label"}),
        "label",
        "count",
        "중분류 상위 비중",
        "#2563eb",
        total,
        top_n=10,
    )
    draw_summary_panel(
        ax_l3,
        level3[["l3", "count"]]
        .groupby("l3", as_index=False)
        .sum()
        .rename(columns={"l3": "label"}),
        "label",
        "count",
        "소분류 상위 비중",
        "#dc2626",
        total,
        top_n=12,
    )

    fig.text(
        0.055,
        0.045,
        "주: 비율은 '화재 건수' 기준입니다. 라벨이 과밀해지는 것을 막기 위해 도넛의 세부 라벨은 일정 비중 이상 구간만 표기했습니다.",
        fontsize=9.6,
        color="#64748b",
        ha="left",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    setup_style()
    df = prepare_data(SOURCE_CSV)
    hierarchy = build_hierarchy(df)
    out = create_figure(hierarchy, OUT_PATH)
    print(f"[완료] {out}")


if __name__ == "__main__":
    main()
