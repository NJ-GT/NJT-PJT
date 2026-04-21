# -*- coding: utf-8 -*-
"""
[파일 설명]
통합숙박시설표제부0414.csv의 컬럼명을 등기부등본 스키마 기준으로 변환하는 스크립트.

주요 역할:
  - 'registry_' 접두사가 붙은 컬럼에서 접두사를 제거한다.
    예: 'registry_도로명대지위치' → '도로명대지위치'
  - 매칭 메타데이터 컬럼은 별도 매핑 테이블(SPECIAL_RENAME)로 처리한다.
  - 숙박업 인허가 정보 컬럼에는 '숙박_' 접두사를 추가한다.
    예: '사업장명' → '숙박_사업장명'

입력: 통합숙박시설표제부0414.csv
출력: data/통합숙박시설표제부0414_등기부기준컬럼명.csv
"""

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent  # 프로젝트 루트
SOURCE_PATH = BASE_DIR / "통합숙박시설표제부0414.csv"                          # 입력 파일
OUTPUT_PATH = BASE_DIR / "data" / "통합숙박시설표제부0414_등기부기준컬럼명.csv"  # 출력 파일


# 매칭 메타데이터 컬럼명 변환 규칙 (registry_ 접두사 방식으로 처리되지 않는 특수 컬럼)
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

    # 컬럼별 이름 변환 규칙 결정
    rename_map = {}
    for col in df.columns:
        if col in SPECIAL_RENAME:
            rename_map[col] = SPECIAL_RENAME[col]   # 특수 매핑 우선 적용
        elif col.startswith("registry_"):
            rename_map[col] = col.replace("registry_", "", 1)  # 'registry_' 접두사 제거
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
