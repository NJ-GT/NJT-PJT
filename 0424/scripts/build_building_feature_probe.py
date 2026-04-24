from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = ROOT / "원본데이터"
ANALYSIS_DIR = ROOT / "0424" / "분석"
TABLE_DIR = ANALYSIS_DIR / "tables"


def find_one(pattern: str) -> Path:
    matches = glob.glob(str(ROOT / pattern))
    if not matches:
        raise FileNotFoundError(f"No files matched: {pattern}")
    return Path(matches[0])


def choose_non_null_first(series: pd.Series):
    non_null = series.dropna()
    if non_null.empty:
        return np.nan
    return non_null.iloc[0]


def prepare_tourism_like_source(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    keep_cols = [
        "사업장명",
        "좌표정보(X)",
        "좌표정보(Y)",
        "지역구분명",
        "주변환경명",
        "건물용도명",
        "총층수",
        "지상층수",
        "지하층수",
        "객실수",
        "건축연면적",
        "시설규모",
        "최종수정일자",
    ]
    keep_cols = [col for col in keep_cols if col in df.columns]
    df = df[keep_cols].copy()

    for col in [
        "지역구분명",
        "주변환경명",
        "건물용도명",
        "객실수",
        "건축연면적",
        "시설규모",
        "총층수",
        "지상층수",
        "지하층수",
    ]:
        if col in df.columns:
            df[col] = df[col].replace("", np.nan)

    numeric_cols = ["좌표정보(X)", "좌표정보(Y)", "총층수", "지상층수", "지하층수", "객실수", "건축연면적", "시설규모"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    info_cols = [col for col in keep_cols if col not in {"사업장명", "좌표정보(X)", "좌표정보(Y)", "최종수정일자"}]
    df["_info_score"] = df[info_cols].notna().sum(axis=1)
    sort_cols = ["사업장명", "좌표정보(X)", "좌표정보(Y)", "_info_score"]
    ascending = [True, True, True, False]
    if "최종수정일자" in df.columns:
        sort_cols.append("최종수정일자")
        ascending.append(False)

    df = df.sort_values(sort_cols, ascending=ascending)
    df = df.groupby(["사업장명", "좌표정보(X)", "좌표정보(Y)"], as_index=False).agg(
        {col: choose_non_null_first for col in df.columns if col not in {"사업장명", "좌표정보(X)", "좌표정보(Y)"}}
    )
    return df.drop(columns=["_info_score", "최종수정일자"], errors="ignore")


def prepare_lodging_source(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    keep_cols = [
        "사업장명",
        "좌표정보(X)",
        "좌표정보(Y)",
        "건물지상층수",
        "건물지하층수",
        "건물총층수",
        "한실수",
        "양실수",
        "최종수정일자",
    ]
    df = df[keep_cols].copy()
    numeric_cols = ["좌표정보(X)", "좌표정보(Y)", "건물지상층수", "건물지하층수", "건물총층수", "한실수", "양실수"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["객실수_추정"] = df[["한실수", "양실수"]].fillna(0).sum(axis=1)
    df["_info_score"] = df[["건물지상층수", "건물지하층수", "건물총층수", "객실수_추정"]].notna().sum(axis=1)
    df = df.sort_values(
        ["사업장명", "좌표정보(X)", "좌표정보(Y)", "_info_score", "최종수정일자"],
        ascending=[True, True, True, False, False],
    )
    df = df.groupby(["사업장명", "좌표정보(X)", "좌표정보(Y)"], as_index=False).agg(
        {
            "건물지상층수": choose_non_null_first,
            "건물지하층수": choose_non_null_first,
            "건물총층수": choose_non_null_first,
            "객실수_추정": choose_non_null_first,
        }
    )
    return df


def classify_housing_type(use_text: pd.Series) -> pd.Series:
    return pd.Series(
        np.select(
        [
            use_text.str.contains("아파트", na=False),
            use_text.str.contains("다세대", na=False),
            use_text.str.contains("다가구", na=False),
            use_text.str.contains("연립", na=False),
            use_text.str.contains("단독주택|단독", na=False),
        ],
        ["아파트", "다세대", "다가구", "연립", "단독"],
        default=None,
        ),
        index=use_text.index,
    )


def classify_region(region_text: pd.Series) -> pd.Series:
    return pd.Series(
        np.select(
        [
            region_text.str.contains("상업", na=False),
            region_text.str.contains("주거", na=False),
        ],
        ["상업지역", "주거지역"],
        default=None,
        ),
        index=region_text.index,
    )


def flag_from_environment(env_text: pd.Series, pattern: str) -> pd.Series:
    cleaned = env_text.fillna("").astype(str)
    result = cleaned.str.contains(pattern, na=False).astype("float")
    result = result.mask(cleaned.eq(""), np.nan)
    return result


def positive_numeric(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.where(numeric > 0)


def build_probe() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    analysis_path = find_one("0424/data/*0423.csv")
    core_path = find_one("data/*0421.csv")
    lodging_path = find_one("원본데이터/서울시 숙박업 인허가 정보.csv")
    tourism_path = find_one("원본데이터/서울시 관광숙박업 인허가 정보.csv")
    foreign_path = find_one("원본데이터/서울시 외국인관광도시민박업 인허가 정보.csv")

    analysis = pd.read_csv(analysis_path, encoding="utf-8-sig")
    core = pd.read_csv(core_path, encoding="utf-8-sig")

    core_key = core[
        [
            "업소명",
            "위도",
            "경도",
            "X좌표",
            "Y좌표",
            "관리건축물대장PK",
            "연면적(㎡)",
            "기타용도",
            "지상층수",
            "지하층수",
            "총층수",
        ]
    ].copy()
    core_key["위도"] = core_key["위도"].round(8)
    core_key["경도"] = core_key["경도"].round(8)
    core_key["X좌표"] = core_key["X좌표"].round(6)
    core_key["Y좌표"] = core_key["Y좌표"].round(6)
    core_key = core_key.drop_duplicates(subset=["업소명", "위도", "경도"], keep="first")

    enriched = analysis.copy()
    enriched["위도"] = enriched["위도"].round(8)
    enriched["경도"] = enriched["경도"].round(8)
    enriched = enriched.merge(core_key, left_on=["숙소명", "위도", "경도"], right_on=["업소명", "위도", "경도"], how="left")

    tourism = prepare_tourism_like_source(tourism_path)
    foreign = prepare_tourism_like_source(foreign_path)
    lodging = prepare_lodging_source(lodging_path)

    for frame in [tourism, foreign, lodging]:
        frame["좌표정보(X)"] = frame["좌표정보(X)"].round(6)
        frame["좌표정보(Y)"] = frame["좌표정보(Y)"].round(6)

    tourism_match = enriched.merge(
        tourism,
        left_on=["숙소명", "X좌표", "Y좌표"],
        right_on=["사업장명", "좌표정보(X)", "좌표정보(Y)"],
        how="left",
        suffixes=("", "_tourism"),
    )
    foreign_match = enriched.merge(
        foreign,
        left_on=["숙소명", "X좌표", "Y좌표"],
        right_on=["사업장명", "좌표정보(X)", "좌표정보(Y)"],
        how="left",
        suffixes=("", "_foreign"),
    )
    lodging_match = enriched.merge(
        lodging,
        left_on=["숙소명", "X좌표", "Y좌표"],
        right_on=["사업장명", "좌표정보(X)", "좌표정보(Y)"],
        how="left",
        suffixes=("", "_lodging"),
    )

    enriched["건물용도명_direct"] = pd.Series([None] * len(enriched), dtype="object")
    enriched["지역구분명_direct"] = pd.Series([None] * len(enriched), dtype="object")
    enriched["주변환경명_direct"] = pd.Series([None] * len(enriched), dtype="object")
    enriched["객실수_direct"] = np.nan
    enriched["시설규모_direct"] = np.nan
    enriched["건축연면적_direct"] = np.nan
    enriched["총층수_direct"] = np.nan
    enriched["지상층수_direct"] = np.nan
    enriched["지하층수_direct"] = np.nan

    tourism_mask = enriched["업종"].eq("관광숙박업")
    foreign_mask = enriched["업종"].eq("외국인관광도시민박업")
    lodging_mask = enriched["업종"].eq("숙박업")

    tourism_cols = ["건물용도명", "지역구분명", "주변환경명", "객실수", "시설규모", "건축연면적", "총층수", "지상층수", "지하층수"]
    for column in tourism_cols:
        target = f"{column}_direct"
        enriched.loc[tourism_mask, target] = tourism_match.loc[tourism_mask, column]
        enriched.loc[foreign_mask, target] = foreign_match.loc[foreign_mask, column]

    enriched.loc[lodging_mask, "객실수_direct"] = lodging_match.loc[lodging_mask, "객실수_추정"]
    enriched.loc[lodging_mask, "총층수_direct"] = lodging_match.loc[lodging_mask, "건물총층수"]
    enriched.loc[lodging_mask, "지상층수_direct"] = lodging_match.loc[lodging_mask, "건물지상층수"]
    enriched.loc[lodging_mask, "지하층수_direct"] = lodging_match.loc[lodging_mask, "건물지하층수"]

    enriched["건물용도명_통합"] = enriched["건물용도명_direct"].fillna(enriched["기타용도"])
    enriched["지역구분명_통합"] = enriched["지역구분명_direct"]
    enriched["객실수_통합"] = positive_numeric(enriched["객실수_direct"])
    enriched["시설규모_통합"] = positive_numeric(enriched["시설규모_direct"])
    enriched["연면적_통합"] = (
        positive_numeric(enriched["건축연면적_direct"])
        .fillna(positive_numeric(enriched["시설규모_direct"]))
        .fillna(positive_numeric(enriched["연면적(㎡)"]))
    )
    enriched["총층수_통합"] = positive_numeric(enriched["총층수_direct"]).fillna(
        positive_numeric(enriched["총층수"])
    )

    above = pd.to_numeric(enriched["지상층수_direct"], errors="coerce").fillna(pd.to_numeric(enriched["지상층수"], errors="coerce")).fillna(0)
    below = pd.to_numeric(enriched["지하층수_direct"], errors="coerce").fillna(pd.to_numeric(enriched["지하층수"], errors="coerce")).fillna(0)
    corrected_above = above.mask(above == 0, 1)
    enriched["총층수_0층만보정"] = (corrected_above + below).clip(lower=1)
    enriched["총층수_현재값과차이"] = pd.to_numeric(enriched["총층수_통합"], errors="coerce") - enriched["총층수_0층만보정"]

    use_text = enriched["건물용도명_통합"].fillna("").astype(str) + " " + enriched["기타용도"].fillna("").astype(str)
    enriched["주택유형_분류"] = classify_housing_type(use_text)
    enriched["지역분류_상업주거"] = classify_region(enriched["지역구분명_통합"].fillna("").astype(str))
    enriched["상업지역여부"] = enriched["지역분류_상업주거"].eq("상업지역").astype("float").mask(enriched["지역구분명_통합"].isna(), np.nan)
    enriched["주거지역여부"] = enriched["지역분류_상업주거"].eq("주거지역").astype("float").mask(enriched["지역구분명_통합"].isna(), np.nan)
    enriched["주택가주변_여부"] = flag_from_environment(enriched["주변환경명_direct"], "주택가")
    enriched["역세권_인접여부"] = flag_from_environment(enriched["주변환경명_direct"], "역세권")
    enriched["유흥가_인접여부"] = flag_from_environment(enriched["주변환경명_direct"], "유흥|유흥업소")

    enriched_out = TABLE_DIR / "분석변수_최종테이블0423_건물특성_probe.csv"
    summary_out = TABLE_DIR / "건물특성_가용성요약.csv"
    floor_rows_out = TABLE_DIR / "총층수_보정비교_rows.csv"
    floor_meta_out = TABLE_DIR / "총층수_보정비교_summary.json"

    enriched.to_csv(enriched_out, index=False, encoding="utf-8-sig")

    summary_rows = [
        {
            "요청변수": "건물용도명",
            "가용행수": int(enriched["건물용도명_통합"].notna().sum()),
            "가용비율": round(float(enriched["건물용도명_통합"].notna().mean() * 100), 2),
            "주요소스": "관광/외국인 원천 건물용도명 + 표제부 기타용도 fallback",
            "비고": "숙박업은 표제부 기타용도 기반 보완",
        },
        {
            "요청변수": "지역구분명",
            "가용행수": int(enriched["지역구분명_통합"].notna().sum()),
            "가용비율": round(float(enriched["지역구분명_통합"].notna().mean() * 100), 2),
            "주요소스": "관광숙박업/외국인관광도시민박업 원천",
            "비고": "숙박업 원천에는 직접 컬럼이 없음",
        },
        {
            "요청변수": "객실수",
            "가용행수": int(enriched["객실수_통합"].notna().sum()),
            "가용비율": round(float(enriched["객실수_통합"].notna().mean() * 100), 2),
            "주요소스": "관광/외국인 원천 객실수 + 숙박업 한실수+양실수 추정",
            "비고": "숙박업은 추정치",
        },
        {
            "요청변수": "총층수",
            "가용행수": int(enriched["총층수_통합"].notna().sum()),
            "가용비율": round(float(enriched["총층수_통합"].notna().mean() * 100), 2),
            "주요소스": "업종별 원천 총층수 + 표제부 총층수 fallback",
            "비고": "0층 보정 재계산 컬럼 함께 저장",
        },
        {
            "요청변수": "시설규모/연면적",
            "가용행수": int(enriched["연면적_통합"].notna().sum()),
            "가용비율": round(float(enriched["연면적_통합"].notna().mean() * 100), 2),
            "주요소스": "건축연면적/시설규모 + 표제부 연면적 fallback",
            "비고": "숙박업은 표제부 연면적 중심",
        },
        {
            "요청변수": "단독/다가구/다세대/아파트 여부",
            "가용행수": int(enriched["주택유형_분류"].notna().sum()),
            "가용비율": round(float(enriched["주택유형_분류"].notna().mean() * 100), 2),
            "주요소스": "건물용도명/기타용도 문자열 분류",
            "비고": "복합용도 건물은 일부 미분류 가능",
        },
        {
            "요청변수": "상업지역·주거지역 여부",
            "가용행수": int(enriched["지역분류_상업주거"].notna().sum()),
            "가용비율": round(float(enriched["지역분류_상업주거"].notna().mean() * 100), 2),
            "주요소스": "지역구분명 분류",
            "비고": "숙박업은 별도 용도지역 공간조인 없이는 공란",
        },
        {
            "요청변수": "주택가주변/역세권/유흥가 인접",
            "가용행수": int(enriched["주변환경명_direct"].notna().sum()),
            "가용비율": round(float(enriched["주변환경명_direct"].notna().mean() * 100), 2),
            "주요소스": "관광숙박업/외국인관광도시민박업 주변환경명",
            "비고": "플래그 3개로 분리 저장",
        },
    ]
    pd.DataFrame(summary_rows).to_csv(summary_out, index=False, encoding="utf-8-sig")

    diff_mask = enriched["총층수_현재값과차이"].abs() > 1e-9
    enriched.loc[
        diff_mask,
        [
            "구",
            "동",
            "숙소명",
            "업종",
            "지상층수_direct",
            "지하층수_direct",
            "총층수_direct",
            "지상층수",
            "지하층수",
            "총층수",
            "총층수_통합",
            "총층수_0층만보정",
            "총층수_현재값과차이",
        ],
    ].to_csv(floor_rows_out, index=False, encoding="utf-8-sig")

    floor_meta = {
        "analysis_rows": int(len(enriched)),
        "base_core_match_rows": int(enriched["관리건축물대장PK"].notna().sum()),
        "current_vs_corrected_diff_rows": int(diff_mask.sum()),
        "all_rows_match_0floor_only_rule": bool(diff_mask.sum() == 0),
        "rule": "총층수 = max((지상층수 if 지상층수!=0 else 1) + 지하층수, 1)",
        "files": {
            "analysis_table": str(analysis_path.relative_to(ROOT)),
            "core_table": str(core_path.relative_to(ROOT)),
            "lodging_source": str(lodging_path.relative_to(ROOT)),
            "tourism_source": str(tourism_path.relative_to(ROOT)),
            "foreign_source": str(foreign_path.relative_to(ROOT)),
        },
    }
    floor_meta_out.write_text(json.dumps(floor_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved: {enriched_out}")
    print(f"Saved: {summary_out}")
    print(f"Saved: {floor_rows_out}")
    print(f"Saved: {floor_meta_out}")
    print(json.dumps(floor_meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    build_probe()
