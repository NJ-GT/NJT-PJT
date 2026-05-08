# -*- coding: utf-8 -*-
"""
마포구 외국인관광도시민박업 위험도를 분석해 구·업종별 요약/상위 위험 시설/연도별 인허가 추이를 산출한다.

목적:
    - 팀 파이프라인 점수(team_pipeline_scored_dataset) + 최종 공간 파이프라인 점수(final_spatial_pipeline) 결합
    - 마포구·외국인민박 시설을 별도 플래그로 구분, 위험점수 상위 정렬 후 저장
    - 외국인민박 인허가 데이터(2020~2025) 추이 함께 산출

입력:
    - data/team_pipeline_validation/team_pipeline_scored_dataset.csv
    - data/final_spatial_pipeline/analysis_dataset.csv
    - 원본데이터/서울시 외국인관광도시민박업 인허가 정보.csv

출력 (data/mapo_foreign_lodging_risk/ 하위):
    - gu_risk_summary.csv               구별 위험 지표 요약(평균/중앙값+순위)
    - gu_upjong_risk_summary.csv        구×업종별 위험 지표 요약
    - mapo_foreign_lodging_top_risk.csv 마포구 외국인민박 시설 위험순 상위 목록
    - foreign_lodging_license_trend_2020_2025.csv 구별 연도별 인허가 추이
    - mapo_summary.json                 마포구 외국인민박 핵심 지표 요약 JSON
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


# 프로젝트 루트(NJT-PJT) 및 산출 폴더
BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "data" / "mapo_foreign_lodging_risk"
OUT.mkdir(parents=True, exist_ok=True)  # 산출 폴더 자동 생성

# 입력 경로
SCORED = BASE / "data" / "team_pipeline_validation" / "team_pipeline_scored_dataset.csv"
FINAL = BASE / "data" / "final_spatial_pipeline" / "analysis_dataset.csv"
LICENSE = BASE / "원본데이터" / "서울시 외국인관광도시민박업 인허가 정보.csv"


def read_csv(path: Path) -> pd.DataFrame:
    """utf-8-sig 로 CSV 로딩 (BOM 처리 + 자료형 추정 비활성화)."""
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def main() -> None:
    """전체 분석 흐름."""
    # 두 점수 테이블 로딩
    scored = read_csv(SCORED)
    final = read_csv(FINAL)
    # 두 테이블을 결합할 키
    key_cols = ["구", "동", "숙소명", "위도", "경도"]
    # final 에서 가져올 점수 컬럼 (존재하는 것만 유지)
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
    # 좌측 조인 (scored 기준 + final 점수 추가)
    df = scored.merge(
        final[final_keep], on=key_cols, how="left", suffixes=("", "_final")
    )
    # 마포구 / 외국인민박 / 둘 모두 플래그 추가
    df["is_mapo"] = df["구"].eq("마포구")
    df["is_foreign_lodging"] = df["업종"].eq("외국인관광도시민박업")
    df["is_mapo_foreign"] = df["is_mapo"] & df["is_foreign_lodging"]

    # 분석 대상 수치 컬럼
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
    # 모두 숫자형으로 강제 (실패 시 NaN)
    for col in metric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── 1) 구별 위험 지표 요약 ────────────────────────────────────────
    gu_rows = []
    for gu, g in df.groupby("구"):
        row = {
            "구": gu,
            "시설수": len(g),
            "외국인민박수": int(g["is_foreign_lodging"].sum()),
        }
        # 각 지표마다 평균·중앙값 컬럼 추가
        for col in metric_cols:
            if col in g.columns:
                row[f"{col}_평균"] = g[col].mean()
                row[f"{col}_중앙값"] = g[col].median()
        gu_rows.append(row)
    gu_summary = pd.DataFrame(gu_rows)
    # 평균/중앙값 컬럼마다 내림차순 순위 컬럼 자동 부여
    for col in [
        c for c in gu_summary.columns if c.endswith("_평균") or c.endswith("_중앙값")
    ]:
        gu_summary[
            col.replace("_평균", "_평균순위").replace("_중앙값", "_중앙값순위")
        ] = gu_summary[col].rank(ascending=False, method="min").astype(int)
    # 저장
    gu_summary.to_csv(OUT / "gu_risk_summary.csv", index=False, encoding="utf-8-sig")

    # ── 2) 구×업종별 위험 요약 ─────────────────────────────────────────
    upjong_rows = []
    for (gu, upjong), g in df.groupby(["구", "업종"]):
        row = {"구": gu, "업종": upjong, "시설수": len(g)}
        for col in metric_cols:
            if col in g.columns:
                row[f"{col}_평균"] = g[col].mean()
                row[f"{col}_중앙값"] = g[col].median()
        upjong_rows.append(row)
    upjong_summary = pd.DataFrame(upjong_rows)
    upjong_summary.to_csv(
        OUT / "gu_upjong_risk_summary.csv", index=False, encoding="utf-8-sig"
    )

    # ── 3) 마포구 외국인민박 위험 상위 ────────────────────────────────
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
    # 존재하는 컬럼만 유지
    top_cols = [c for c in top_cols if c in mapo_foreign.columns]
    # 위험도점수 → fire_count_150m 순으로 내림차순 정렬 후 저장
    mapo_foreign.sort_values(["위험도점수", "fire_count_150m"], ascending=False)[
        top_cols
    ].to_csv(
        OUT / "mapo_foreign_lodging_top_risk.csv", index=False, encoding="utf-8-sig"
    )

    # ── 4) 외국인민박 인허가 추이 (2020~2025) ──────────────────────────
    year_summary = pd.DataFrame()
    if LICENSE.exists():
        lic = read_csv(LICENSE)
        # 인허가일자 → 연도 추출
        lic["인허가연도"] = pd.to_datetime(lic["인허가일자"], errors="coerce").dt.year
        # 지번주소에서 'OO구' 정규식 추출
        lic["구"] = lic["지번주소"].astype(str).str.extract(r"서울특별시\s+(\S+구)")
        lic = lic[lic["구"].notna()]
        # 연도/구별 신규 인허가 수
        year_summary = (
            lic[lic["인허가연도"].between(2020, 2025)]
            .groupby(["구", "인허가연도"])
            .size()
            .reset_index(name="신규인허가")
        )
        year_summary.to_csv(
            OUT / "foreign_lodging_license_trend_2020_2025.csv",
            index=False,
            encoding="utf-8-sig",
        )

    # ── 5) 마포구 외국인민박 핵심 지표 JSON ───────────────────────────
    mapo_stats = {
        "mapo_total_facilities": int((df["구"] == "마포구").sum()),
        "mapo_foreign_lodging_count": int(len(mapo_foreign)),
        # 마포구 내 외국인민박 비중
        "mapo_foreign_share_in_mapo": float(
            len(mapo_foreign) / max(1, (df["구"] == "마포구").sum())
        ),
        "mapo_foreign_mean_risk_score": float(mapo_foreign["위험도점수"].mean()),
        "mapo_foreign_mean_fire_count_150m": float(
            mapo_foreign["fire_count_150m"].mean()
        ),
        "mapo_foreign_fire_exists_rate": float(mapo_foreign["fire_exists_150m"].mean()),
        "mapo_foreign_mean_damage_sum_thousand": float(
            mapo_foreign["target_damage_sum_천원"].mean()
        ),
    }
    # JSON 저장 + 콘솔에 그대로 출력
    (OUT / "mapo_summary.json").write_text(
        json.dumps(mapo_stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(mapo_stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
