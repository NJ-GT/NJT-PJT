# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "data" / "mapo_foreign_lodging_risk"
OUT.mkdir(parents=True, exist_ok=True)

SCORED = BASE / "data" / "team_pipeline_validation" / "team_pipeline_scored_dataset.csv"
FINAL = BASE / "data" / "final_spatial_pipeline" / "analysis_dataset.csv"
LICENSE = BASE / "원본데이터" / "서울시 외국인관광도시민박업 인허가 정보.csv"


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def main() -> None:
    scored = read_csv(SCORED)
    final = read_csv(FINAL)
    key_cols = ["구", "동", "숙소명", "위도", "경도"]
    final_keep = [
        c
        for c in key_cols
        + [
            "위험점수_AHP",
            "사각지대_위험도점수",
            "예상_화재발생확률",
            "기대피해액_백만원",
            "소방위험도_점수",
        ]
        if c in final.columns
    ]
    df = scored.merge(final[final_keep], on=key_cols, how="left", suffixes=("", "_final"))
    df["is_mapo"] = df["구"].eq("마포구")
    df["is_foreign_lodging"] = df["업종"].eq("외국인관광도시민박업")
    df["is_mapo_foreign"] = df["is_mapo"] & df["is_foreign_lodging"]

    metric_cols = [
        "위험도점수",
        "위험점수_AHP",
        "사각지대_위험도점수",
        "fire_count_150m",
        "fire_exists_150m",
        "target_damage_sum_천원",
        "예상_화재발생확률",
        "기대피해액_백만원",
        "주변건물수",
        "집중도",
        "단속위험도",
        "구조노후도",
        "도로폭위험도",
    ]
    for col in metric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    gu_rows = []
    for gu, g in df.groupby("구"):
        row = {"구": gu, "시설수": len(g), "외국인민박수": int(g["is_foreign_lodging"].sum())}
        for col in metric_cols:
            if col in g.columns:
                row[f"{col}_평균"] = g[col].mean()
                row[f"{col}_중앙값"] = g[col].median()
        gu_rows.append(row)
    gu_summary = pd.DataFrame(gu_rows)
    for col in [c for c in gu_summary.columns if c.endswith("_평균") or c.endswith("_중앙값")]:
        gu_summary[col.replace("_평균", "_평균순위").replace("_중앙값", "_중앙값순위")] = gu_summary[col].rank(
            ascending=False, method="min"
        ).astype(int)
    gu_summary.to_csv(OUT / "gu_risk_summary.csv", index=False, encoding="utf-8-sig")

    upjong_rows = []
    for (gu, upjong), g in df.groupby(["구", "업종"]):
        row = {"구": gu, "업종": upjong, "시설수": len(g)}
        for col in metric_cols:
            if col in g.columns:
                row[f"{col}_평균"] = g[col].mean()
                row[f"{col}_중앙값"] = g[col].median()
        upjong_rows.append(row)
    upjong_summary = pd.DataFrame(upjong_rows)
    upjong_summary.to_csv(OUT / "gu_upjong_risk_summary.csv", index=False, encoding="utf-8-sig")

    mapo_foreign = df[df["is_mapo_foreign"]].copy()
    top_cols = [
        "구",
        "동",
        "숙소명",
        "승인연도",
        "위험도점수",
        "위험점수_AHP",
        "사각지대_위험도점수",
        "fire_count_150m",
        "target_damage_sum_천원",
        "예상_화재발생확률",
        "기대피해액_백만원",
        "주변건물수",
        "집중도",
        "도로폭위험도",
        "단속위험도",
        "구조노후도",
    ]
    top_cols = [c for c in top_cols if c in mapo_foreign.columns]
    mapo_foreign.sort_values(["위험도점수", "fire_count_150m"], ascending=False)[top_cols].to_csv(
        OUT / "mapo_foreign_lodging_top_risk.csv", index=False, encoding="utf-8-sig"
    )

    year_summary = pd.DataFrame()
    if LICENSE.exists():
        lic = read_csv(LICENSE)
        lic["인허가연도"] = pd.to_datetime(lic["인허가일자"], errors="coerce").dt.year
        lic["구"] = lic["지번주소"].astype(str).str.extract(r"서울특별시\s+(\S+구)")
        lic = lic[lic["구"].notna()]
        year_summary = (
            lic[lic["인허가연도"].between(2020, 2025)]
            .groupby(["구", "인허가연도"])
            .size()
            .reset_index(name="신규인허가")
        )
        year_summary.to_csv(OUT / "foreign_lodging_license_trend_2020_2025.csv", index=False, encoding="utf-8-sig")

    mapo_stats = {
        "mapo_total_facilities": int((df["구"] == "마포구").sum()),
        "mapo_foreign_lodging_count": int(len(mapo_foreign)),
        "mapo_foreign_share_in_mapo": float(len(mapo_foreign) / max(1, (df["구"] == "마포구").sum())),
        "mapo_foreign_mean_risk_score": float(mapo_foreign["위험도점수"].mean()),
        "mapo_foreign_mean_fire_count_150m": float(mapo_foreign["fire_count_150m"].mean()),
        "mapo_foreign_fire_exists_rate": float(mapo_foreign["fire_exists_150m"].mean()),
        "mapo_foreign_mean_damage_sum_thousand": float(mapo_foreign["target_damage_sum_천원"].mean()),
    }
    (OUT / "mapo_summary.json").write_text(json.dumps(mapo_stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(mapo_stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
