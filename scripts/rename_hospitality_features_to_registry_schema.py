# -*- coding: utf-8 -*-
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
SOURCE_PATH = BASE_DIR / "통합숙박시설표제부0414.csv"
OUTPUT_PATH = BASE_DIR / "data" / "통합숙박시설표제부0414_등기부기준컬럼명.csv"


SPECIAL_RENAME = {
    "selected_registry_pk": "관리건축물대장PK_매칭",
    "selected_registry_name_similarity": "건물명유사도",
    "selected_by": "매칭선택기준",
    "match_type": "매칭유형",
    "match_candidate_count": "매칭후보수",
}


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(SOURCE_PATH, encoding="utf-8-sig", low_memory=False)

    rename_map = {}
    for col in df.columns:
        if col in SPECIAL_RENAME:
            rename_map[col] = SPECIAL_RENAME[col]
        elif col.startswith("registry_"):
            rename_map[col] = col.replace("registry_", "", 1)
        elif col in {
            "개방자치단체코드",
            "관리번호",
            "인허가일자",
            "인허가취소일자",
            "영업상태코드",
            "영업상태명",
            "상세영업상태코드",
            "상세영업상태명",
            "폐업일자",
            "휴업시작일자",
            "휴업종료일자",
            "재개업일자",
            "전화번호",
            "소재지면적",
            "소재지우편번호",
            "지번주소",
            "도로명주소",
            "도로명우편번호",
            "사업장명",
            "최종수정일자",
            "데이터갱신구분",
            "데이터갱신일자",
            "업태구분명",
            "좌표정보(X)",
            "좌표정보(Y)",
            "좌석수",
        }:
            rename_map[col] = f"숙박_{col}"

    renamed = df.rename(columns=rename_map)
    renamed.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print(f"rows={len(renamed)}")
    print(f"cols={len(renamed.columns)}")
    print(f"output={OUTPUT_PATH}")


if __name__ == "__main__":
    main()
