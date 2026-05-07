from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_INPUT = ROOT / "0424" / "data" / "분석변수_최종테이블0423.csv"
CORE_INPUT = ROOT / "data" / "핵심서울0424.csv"
OUTPUT_DIR = ROOT / "0424" / "분석"
TABLE_DIR = OUTPUT_DIR / "tables"
FIG_DIR = OUTPUT_DIR / "figures"

GRADE_ORDER = ["안전", "보통", "위험"]
GRADE_COLORS = {
    "안전": "#4C956C",
    "보통": "#F4A259",
    "위험": "#D1495B",
}
GROUP_ORDER = ["기존숙박군", "외국인관광도시민박업"]
GROUP_COLORS = {
    "기존숙박군": "#2A6F97",
    "외국인관광도시민박업": "#8C5E58",
}


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    analysis = pd.read_csv(ANALYSIS_INPUT, encoding="utf-8-sig")
    core = pd.read_csv(CORE_INPUT, encoding="utf-8-sig")
    return analysis, core


def add_ahp_score(analysis: pd.DataFrame, core: pd.DataFrame) -> pd.DataFrame:
    base = analysis.copy()
    key_cols = ["업소명", "위도", "경도", "위험점수_AHP"]
    core_key = core[key_cols].copy()

    for frame, name_col in [(base, "숙소명"), (core_key, "업소명")]:
        frame[name_col] = frame[name_col].astype(str).str.strip()
        frame["위도_key"] = pd.to_numeric(frame["위도"], errors="coerce").round(8)
        frame["경도_key"] = pd.to_numeric(frame["경도"], errors="coerce").round(8)

    core_key = core_key[["업소명", "위도_key", "경도_key", "위험점수_AHP"]]
    core_key = core_key.drop_duplicates(subset=["업소명", "위도_key", "경도_key"], keep="first")
    merged = base.merge(
        core_key.rename(columns={"업소명": "숙소명"}),
        on=["숙소명", "위도_key", "경도_key"],
        how="left",
    )
    merged["AHP_매칭방식"] = np.where(merged["위험점수_AHP"].notna(), "exact", "missing")

    unmatched = merged["위험점수_AHP"].isna()
    if unmatched.any():
        name_counts = core_key["업소명"].value_counts()
        unique_names = name_counts[name_counts.eq(1)].index
        fallback_map = (
            core_key.loc[core_key["업소명"].isin(unique_names), ["업소명", "위험점수_AHP"]]
            .drop_duplicates(subset=["업소명"])
            .set_index("업소명")["위험점수_AHP"]
        )
        filled = merged.loc[unmatched, "숙소명"].map(fallback_map)
        fill_mask = unmatched & filled.notna()
        merged.loc[fill_mask, "위험점수_AHP"] = filled.loc[fill_mask]
        merged.loc[fill_mask, "AHP_매칭방식"] = "name_fallback"

    merged["위험점수_AHP"] = pd.to_numeric(merged["위험점수_AHP"], errors="coerce")
    merged = merged.drop(columns=["위도_key", "경도_key"])
    return merged


def build_group_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["업종그룹"] = np.where(
        out["업종"].eq("외국인관광도시민박업"),
        "외국인관광도시민박업",
        "기존숙박군",
    )
    return out


def compute_cutoffs(series: pd.Series) -> tuple[float, float]:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        raise ValueError("Cannot compute cutoffs from an empty score series.")

    lower = float(numeric.quantile(1 / 3))
    upper = float(numeric.quantile(2 / 3))

    if lower >= upper:
        ranked = numeric.rank(method="first")
        lower = float(numeric.loc[ranked <= ranked.quantile(1 / 3)].max())
        upper = float(numeric.loc[ranked <= ranked.quantile(2 / 3)].max())

    return lower, upper


def assign_grade(series: pd.Series, lower: float, upper: float) -> pd.Categorical:
    numeric = pd.to_numeric(series, errors="coerce")
    labels = pd.Series(pd.NA, index=series.index, dtype="object")
    labels.loc[numeric.notna() & (numeric <= lower)] = "안전"
    labels.loc[numeric.notna() & (numeric > lower) & (numeric <= upper)] = "보통"
    labels.loc[numeric.notna() & (numeric > upper)] = "위험"
    return pd.Categorical(labels, categories=GRADE_ORDER, ordered=True)


def build_grade_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    rows: list[dict[str, object]] = []

    common_lower, common_upper = compute_cutoffs(out["위험점수_AHP"])
    out["공통_3등급"] = assign_grade(out["위험점수_AHP"], common_lower, common_upper)
    rows.append(
        {
            "기준": "공통",
            "업종그룹": "전체",
            "하한_안전상한": round(common_lower, 2),
            "상한_보통상한": round(common_upper, 2),
            "행수": int(out["위험점수_AHP"].notna().sum()),
        }
    )

    group_col = pd.Series(pd.NA, index=out.index, dtype="object")
    for group_name, group_df in out.groupby("업종그룹"):
        lower, upper = compute_cutoffs(group_df["위험점수_AHP"])
        group_col.loc[group_df.index] = assign_grade(group_df["위험점수_AHP"], lower, upper).astype(object)
        rows.append(
            {
                "기준": "업종군별",
                "업종그룹": group_name,
                "하한_안전상한": round(lower, 2),
                "상한_보통상한": round(upper, 2),
                "행수": int(group_df["위험점수_AHP"].notna().sum()),
            }
        )

    out["업종군별_3등급"] = pd.Categorical(group_col, categories=GRADE_ORDER, ordered=True)
    cutoff_df = pd.DataFrame(rows)
    return out, cutoff_df


def build_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for 기준, grade_col in [("공통", "공통_3등급"), ("업종군별", "업종군별_3등급")]:
        counts = (
            df.groupby(["업종그룹", grade_col], observed=False)
            .size()
            .rename("숙소수")
            .reset_index()
            .rename(columns={grade_col: "등급"})
        )
        counts["비율"] = counts.groupby("업종그룹")["숙소수"].transform(lambda s: (s / s.sum() * 100).round(2))
        counts.insert(0, "기준", 기준)
        records.append(counts)

    summary = pd.concat(records, ignore_index=True)
    summary["등급"] = pd.Categorical(summary["등급"], categories=GRADE_ORDER, ordered=True)
    return summary.sort_values(["기준", "업종그룹", "등급"]).reset_index(drop=True)


def add_percent_labels(ax: plt.Axes, values: list[float], bottoms: list[float], labels: list[str]) -> None:
    for idx, (value, bottom, label) in enumerate(zip(values, bottoms, labels)):
        if value < 7:
            continue
        ax.text(
            idx,
            bottom + value / 2,
            label,
            ha="center",
            va="center",
            fontsize=10,
            color="white",
            fontweight="bold",
        )


def plot_stacked_distribution(
    ax: plt.Axes,
    summary: pd.DataFrame,
    title: str,
    show_legend: bool = True,
) -> None:
    x = np.arange(len(GROUP_ORDER))
    bottoms = np.zeros(len(GROUP_ORDER))

    for grade in GRADE_ORDER:
        vals = [
            summary.loc[
                summary["업종그룹"].eq(group) & summary["등급"].eq(grade),
                "비율",
            ].iloc[0]
            for group in GROUP_ORDER
        ]
        ax.bar(
            x,
            vals,
            bottom=bottoms,
            color=GRADE_COLORS[grade],
            edgecolor="white",
            width=0.58,
            label=grade,
        )
        add_percent_labels(ax, vals, bottoms.tolist(), [f"{v:.1f}%" for v in vals])
        bottoms += np.array(vals)

    ax.set_xticks(x, GROUP_ORDER)
    ax.set_ylim(0, 100)
    ax.set_ylabel("비율(%)")
    ax.set_title(title, fontsize=15, fontweight="bold", loc="left")
    ax.grid(alpha=0.15, axis="y")
    if show_legend:
        ax.legend(frameon=False, ncols=3, bbox_to_anchor=(1.0, 1.12), loc="upper right")


def build_cutoff_text(cutoff_df: pd.DataFrame, mode: str) -> list[str]:
    if mode == "공통":
        row = cutoff_df.loc[cutoff_df["기준"].eq("공통")].iloc[0]
        return [
            "공통 분위수 기준",
            f"안전: {row['하한_안전상한']:.2f} 이하",
            f"보통: {row['하한_안전상한']:.2f} 초과 ~ {row['상한_보통상한']:.2f} 이하",
            f"위험: {row['상한_보통상한']:.2f} 초과",
        ]

    text_lines: list[str] = []
    for group_name in GROUP_ORDER:
        row = cutoff_df.loc[
            cutoff_df["기준"].eq("업종군별") & cutoff_df["업종그룹"].eq(group_name)
        ].iloc[0]
        text_lines.extend(
            [
                f"{group_name} 내부 기준",
                f"안전: {row['하한_안전상한']:.2f} 이하",
                f"보통: {row['하한_안전상한']:.2f} 초과 ~ {row['상한_보통상한']:.2f} 이하",
                f"위험: {row['상한_보통상한']:.2f} 초과",
                "",
            ]
        )
    return text_lines[:-1]


def plot_common_only(df: pd.DataFrame, cutoff_df: pd.DataFrame, summary: pd.DataFrame, output_path: Path) -> None:
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    fig = plt.figure(figsize=(14, 7.8), facecolor="#F7F3EC")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1.0], wspace=0.18)
    ax1 = fig.add_subplot(gs[0, 0], facecolor="white")
    ax2 = fig.add_subplot(gs[0, 1], facecolor="white")

    common_row = cutoff_df.loc[cutoff_df["기준"].eq("공통")].iloc[0]
    for group_name in GROUP_ORDER:
        group_scores = df.loc[df["업종그룹"].eq(group_name), "위험점수_AHP"].dropna()
        ax1.hist(
            group_scores,
            bins=24,
            alpha=0.58,
            color=GROUP_COLORS[group_name],
            label=f"{group_name} ({len(group_scores):,})",
            density=True,
        )

    for cutoff, label in [
        (common_row["하한_안전상한"], f"안전 상한 {common_row['하한_안전상한']:.2f}"),
        (common_row["상한_보통상한"], f"보통 상한 {common_row['상한_보통상한']:.2f}"),
    ]:
        ax1.axvline(cutoff, color="#233D4D", linestyle="--", linewidth=2)
        ax1.text(cutoff + 0.7, ax1.get_ylim()[1] * 0.92, label, fontsize=10.5, color="#233D4D")

    ax1.set_title("위험점수_AHP 분포와 공통 3등급 컷오프", fontsize=16, fontweight="bold", loc="left")
    ax1.set_xlabel("위험점수_AHP")
    ax1.set_ylabel("밀도")
    ax1.legend(frameon=False, loc="upper right")
    ax1.grid(alpha=0.15, axis="y")

    common_summary = summary.loc[summary["기준"].eq("공통")].copy()
    plot_stacked_distribution(ax2, common_summary, "공통 기준 위험등급 분포", show_legend=True)

    fig.suptitle("위험점수_AHP 공통 기준 3등급", fontsize=20, fontweight="bold", y=0.98)
    fig.text(
        0.02,
        0.03,
        "전체 숙소 분포를 기준으로 같은 컷오프를 적용했을 때, 외국인관광도시민박업이 위험 쪽에 더 많이 분포하는지 확인하는 그림.",
        fontsize=11,
        color="#5A5A5A",
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_grouped_only(cutoff_df: pd.DataFrame, summary: pd.DataFrame, output_path: Path) -> None:
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    fig = plt.figure(figsize=(14, 7.8), facecolor="#F7F3EC")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1.0], wspace=0.18)
    ax1 = fig.add_subplot(gs[0, 0], facecolor="white")
    ax2 = fig.add_subplot(gs[0, 1], facecolor="#F7F3EC")

    grouped_summary = summary.loc[summary["기준"].eq("업종군별")].copy()
    plot_stacked_distribution(ax1, grouped_summary, "업종군 내부 기준 위험등급 분포", show_legend=True)

    ax2.axis("off")
    ax2.text(
        0.0,
        0.98,
        "업종군별 컷오프",
        fontsize=16,
        fontweight="bold",
        color="#233D4D",
        va="top",
    )
    ax2.text(
        0.0,
        0.90,
        "\n".join(build_cutoff_text(cutoff_df, "업종군별")),
        fontsize=13,
        color="#2F4858",
        va="top",
        linespacing=1.65,
        bbox=dict(boxstyle="round,pad=0.8", facecolor="white", edgecolor="#E6D5C3"),
    )

    fig.suptitle("위험점수_AHP 업종군별 3등급", fontsize=20, fontweight="bold", y=0.98)
    fig.text(
        0.02,
        0.03,
        "숙박업+관광숙박업과 외국인관광도시민박업을 각각 내부 분위수로 나눠, 각 그룹 안에서 상대적으로 위험한 시설을 고르는 기준.",
        fontsize=11,
        color="#5A5A5A",
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_dashboard(df: pd.DataFrame, cutoff_df: pd.DataFrame, summary: pd.DataFrame, output_path: Path) -> None:
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    fig = plt.figure(figsize=(16, 10), facecolor="#F7F3EC")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1.0], hspace=0.28, wspace=0.18)

    ax1 = fig.add_subplot(gs[0, 0], facecolor="white")
    ax2 = fig.add_subplot(gs[0, 1], facecolor="white")
    ax3 = fig.add_subplot(gs[1, 0], facecolor="white")
    ax4 = fig.add_subplot(gs[1, 1], facecolor="#F7F3EC")

    common_row = cutoff_df.loc[cutoff_df["기준"].eq("공통")].iloc[0]

    for group_name in GROUP_ORDER:
        group_scores = df.loc[df["업종그룹"].eq(group_name), "위험점수_AHP"].dropna()
        ax1.hist(
            group_scores,
            bins=24,
            alpha=0.55,
            color=GROUP_COLORS[group_name],
            label=f"{group_name} ({len(group_scores):,})",
            density=True,
        )

    for cutoff, label in [
        (common_row["하한_안전상한"], f"안전 상한 {common_row['하한_안전상한']:.2f}"),
        (common_row["상한_보통상한"], f"보통 상한 {common_row['상한_보통상한']:.2f}"),
    ]:
        ax1.axvline(cutoff, color="#233D4D", linestyle="--", linewidth=2)
        ax1.text(cutoff + 0.6, ax1.get_ylim()[1] * 0.92, label, fontsize=10, color="#233D4D")

    ax1.set_title("위험점수_AHP 분포와 공통 3등급 기준", fontsize=15, fontweight="bold", loc="left")
    ax1.set_xlabel("위험점수_AHP")
    ax1.set_ylabel("밀도")
    ax1.legend(frameon=False, loc="upper right")
    ax1.grid(alpha=0.15, axis="y")

    common_summary = summary.loc[summary["기준"].eq("공통")].copy()
    plot_stacked_distribution(ax2, common_summary, "공통 기준으로 봤을 때의 위험등급 분포", show_legend=True)

    grouped_summary = summary.loc[summary["기준"].eq("업종군별")].copy()
    plot_stacked_distribution(ax3, grouped_summary, "업종군 내부 기준으로 다시 자른 3등급 분포", show_legend=False)

    ax4.axis("off")
    ax4.text(
        0.0,
        0.98,
        "기준값 요약",
        fontsize=16,
        fontweight="bold",
        color="#233D4D",
        va="top",
    )

    text_lines = build_cutoff_text(cutoff_df, "공통") + [""] + build_cutoff_text(cutoff_df, "업종군별") + [""]

    matched = int(df["위험점수_AHP"].notna().sum())
    text_lines.extend(
        [
            f"AHP 매칭: {matched:,}/{len(df):,}행",
            f"기존숙박군: {int(df['업종그룹'].eq('기존숙박군').sum()):,}개",
            f"외국인관광도시민박업: {int(df['업종그룹'].eq('외국인관광도시민박업').sum()):,}개",
        ]
    )

    ax4.text(
        0.0,
        0.90,
        "\n".join(text_lines),
        fontsize=12,
        color="#2F4858",
        va="top",
        linespacing=1.55,
        bbox=dict(boxstyle="round,pad=0.8", facecolor="white", edgecolor="#E6D5C3"),
    )

    fig.suptitle("위험점수_AHP 3등급 비교: 공통 기준 vs 업종군 내부 기준", fontsize=20, fontweight="bold", y=0.98)
    fig.text(
        0.02,
        0.02,
        "숙박업+관광숙박업은 '기존숙박군'으로 묶고, 외국인관광도시민박업은 별도로 분위수 3등급을 계산함.",
        fontsize=11,
        color="#5A5A5A",
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    analysis, core = load_inputs()
    enriched = add_ahp_score(analysis, core)
    enriched = build_group_column(enriched)
    enriched, cutoff_df = build_grade_columns(enriched)
    summary = build_summary_table(enriched)

    output_table = TABLE_DIR / "분석변수_최종테이블0423_AHP3등급비교.csv"
    cutoff_path = TABLE_DIR / "위험점수_AHP_3등급_컷오프요약.csv"
    summary_path = TABLE_DIR / "위험점수_AHP_3등급_분포요약.csv"
    fig_path = FIG_DIR / "AHP_3등급_공통vs업종군비교.png"
    common_fig_path = FIG_DIR / "AHP_3등급_공통기준.png"
    grouped_fig_path = FIG_DIR / "AHP_3등급_업종군별기준.png"

    enriched.to_csv(output_table, index=False, encoding="utf-8-sig")
    cutoff_df.to_csv(cutoff_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    plot_common_only(enriched, cutoff_df, summary, common_fig_path)
    plot_grouped_only(cutoff_df, summary, grouped_fig_path)
    plot_dashboard(enriched, cutoff_df, summary, fig_path)

    print(f"Saved table: {output_table}")
    print(f"Saved cutoffs: {cutoff_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved common figure: {common_fig_path}")
    print(f"Saved grouped figure: {grouped_fig_path}")
    print(f"Saved figure: {fig_path}")
    print(
        {
            "rows": int(len(enriched)),
            "ahp_matched": int(enriched["위험점수_AHP"].notna().sum()),
            "existing_group_rows": int(enriched["업종그룹"].eq("기존숙박군").sum()),
            "foreign_group_rows": int(enriched["업종그룹"].eq("외국인관광도시민박업").sum()),
        }
    )


if __name__ == "__main__":
    main()
