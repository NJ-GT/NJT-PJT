from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "새 폴더" / "날짜별_concat" / "생활인구수.csv"
OUTPUT_PATH = BASE_DIR / "새 폴더" / "날짜별_concat" / "생활인구수_한글컬럼.csv"
DICT_PATH = BASE_DIR / "새 폴더" / "날짜별_concat" / "생활인구수_컬럼정의.csv"

HOUR_PREFIX = {
    "ZERO": 0,
    "ONE": 1,
    "TWO": 2,
    "THREE": 3,
    "FOUR": 4,
    "FIVE": 5,
    "SIX": 6,
    "SEVEN": 7,
    "EIGHT": 8,
    "NINE": 9,
    "TEN": 10,
    "ELEVEN": 11,
    "TLV": 12,
    "THIRTEEN": 13,
    "FOURTEEN": 14,
    "FFTN": 15,
    "SXTN": 16,
    "SVNTN": 17,
    "EGHTN": 18,
    "NNTN": 19,
    "TWNT": 20,
    "TNT": 20,
    "TNONE": 21,
    "TNTONE": 21,
    "TNTT": 22,
    "TNTTH": 23,
}

BASE_RENAME = {
    "source_file": "원본파일명",
    "source_ym": "파일기준년월",
    "TRDAR_NO": "상권번호",
    "TRDAR_NM": "상권명",
    "SIGNGU_CD": "시군구코드",
    "SIGNGU_NM": "시군구명",
    "GRFA_SM": "총면적",
    "RSDNTL_BULD_GRFA": "주거건물면적",
    "RSDNTL_BULD_EXCL_GRFA": "비주거건물제외면적",
    "MOAV_STPRD_FRNR_LVLH_POPNO": "월평균_외국인_상주생활인구수",
    "MOAV_STPRD_FRNR_VST_POPNO": "월평균_외국인_방문생활인구수",
    "TRDAR_CRDT_CONT": "상권좌표내용",
    "TRDAR_CRDNT_CONT": "상권좌표내용",
    "DATA_STRD_YM": "데이터기준년월",
}

CANONICAL_PAIRS = {
    "TWNT_TIZN_STPRD_FRNR_LVLH_POPNO": "TNT_TIZN_STPRD_FRNR_LVLH_POPNO",
    "TWNT_TIZN_STPRD_FRNR_VST_POPNO": "TNT_TIZN_STPRD_FRNR_VST_POPNO",
    "TNONE_TIZN_STPRD_FRNR_LVLH_POPNO": "TNTONE_TIZN_STPRD_FRNR_LVLH_POPNO",
    "TNONE_TIZN_STPRD_FRNR_VST_POPNO": "TNTONE_TIZN_STPRD_FRNR_VST_POPNO",
    "TRDAR_CRDT_CONT": "TRDAR_CRDNT_CONT",
}


def korean_name(col: str) -> str:
    if col in BASE_RENAME:
        return BASE_RENAME[col]

    for prefix, hour in HOUR_PREFIX.items():
        living = f"{prefix}_TIZN_STPRD_FRNR_LVLH_POPNO"
        visiting = f"{prefix}_TIZN_STPRD_FRNR_VST_POPNO"
        if col == living:
            return f"{hour:02d}시_외국인_상주생활인구수"
        if col == visiting:
            return f"{hour:02d}시_외국인_방문생활인구수"

    return col


def merge_synonym_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for keep, alt in CANONICAL_PAIRS.items():
        if keep in df.columns and alt in df.columns:
            df[keep] = df[keep].combine_first(df[alt])
            df = df.drop(columns=[alt])
        elif alt in df.columns and keep not in df.columns:
            df = df.rename(columns={alt: keep})
    return df


def main() -> None:
    df = pd.read_csv(INPUT_PATH, encoding="utf-8-sig", dtype=str)
    original_columns = list(df.columns)
    df = merge_synonym_columns(df)

    rename_map = {col: korean_name(col) for col in df.columns}
    out = df.rename(columns=rename_map)
    out.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    dictionary = pd.DataFrame(
        {
            "원본컬럼": original_columns,
            "한글컬럼": [korean_name(CANONICAL_PAIRS.get(col, col)) for col in original_columns],
        }
    )
    dictionary.to_csv(DICT_PATH, index=False, encoding="utf-8-sig")

    print(OUTPUT_PATH)
    print(DICT_PATH)
    print(f"rows={len(out)} columns_before={len(original_columns)} columns_after={len(out.columns)}")


if __name__ == "__main__":
    main()
