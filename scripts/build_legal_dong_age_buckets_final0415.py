# -*- coding: utf-8 -*-
"""
[파일 설명]
통합숙박시설최종안0415.csv의 사용승인일을 기준으로 법정동별 건물 노후화 구간 통계를 생성한다.

주요 역할:
  1. KIKmix.xlsx에서 법정동코드 → 구명·법정동명 매핑 테이블을 로드한다.
  2. 각 숙박시설의 사용승인일 → 연한 계산 → 구간(10/30/50년) 집계
  3. 자치구·법정동별 건물 수와 연한 구간 통계를 CSV로 저장한다.

입력: data/통합숙박시설최종안0415.csv
      data/jscode20200515/jscode20200515/KIKmix.20200515.xlsx (법정동 코드 매핑)
출력: data/자치구_법정동별_사용승인연한구간_최종안0415.csv
"""
from __future__ import annotations

import csv
import datetime as dt
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

SOURCE_PATH = DATA_DIR / "통합숙박시설최종안0415.csv"
KIKMIX_PATH = DATA_DIR / "jscode20200515" / "jscode20200515" / "KIKmix.20200515.xlsx"
OUT_PATH = DATA_DIR / "자치구_법정동별_사용승인연한구간_최종안0415.csv"

AS_OF_DATE = dt.date(2026, 4, 15)


def load_legal_dong_mapping(path: Path) -> dict[str, dict[str, str]]:
    """KIKmix.xlsx에서 서울 법정동코드 → {구, 법정동명} 딕셔너리를 반환한다."""
    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

    with zipfile.ZipFile(path) as zf:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in root.findall("a:si", ns):
                texts = [t.text or "" for t in si.findall(".//a:t", ns)]
                shared_strings.append("".join(texts))

        def cell_value(cell: ET.Element) -> str:
            value = cell.find("a:v", ns)
            if value is None:
                return ""
            if cell.get("t") == "s":
                return shared_strings[int(value.text)]
            return value.text or ""

        sheet = ET.fromstring(zf.read("xl/worksheets/sheet1.xml"))
        rows = sheet.findall(".//a:sheetData/a:row", ns)

        header: list[str] = []
        mapping: dict[str, dict[str, str]] = {}
        for idx, row in enumerate(rows):
            values = [cell_value(cell) for cell in row.findall("a:c", ns)]
            if idx == 0:
                header = values
                continue
            if not header or len(values) < 6:
                continue

            record = {header[i]: values[i] if i < len(values) else "" for i in range(len(header))}
            if record.get("시도명") != "서울특별시":
                continue
            if record.get("말소일자"):
                continue

            legal_code = str(record.get("법정동코드", "")).strip()
            if legal_code:
                mapping[legal_code] = {
                    "구": str(record.get("시군구명", "")).strip(),
                    "법정동명": str(record.get("동리명", "")).strip(),
                }

        return mapping


def parse_approval_date(value: str) -> dt.date | None:
    """'20050312' 형식의 8자리 날짜 문자열을 date 객체로 변환한다. 형식 오류면 None."""
    text = str(value or "").strip()
    if text.endswith(".0"):
        text = text[:-2]
    if len(text) != 8 or not text.isdigit():
        return None
    try:
        return dt.date(int(text[:4]), int(text[4:6]), int(text[6:8]))
    except ValueError:
        return None


def age_in_years(approval_date: dt.date, as_of_date: dt.date) -> float:
    """사용승인일로부터 기준일까지의 건물 연한을 연 단위 실수로 반환한다."""
    return (as_of_date - approval_date).days / 365.2425


def main() -> None:
    mapping = load_legal_dong_mapping(KIKMIX_PATH)

    grouped: dict[tuple[str, str, str], dict[str, int]] = defaultdict(
        lambda: {
            "전체건물수": 0,
            "사용승인일유효건수": 0,
            "10년미만건물수": 0,
            "10년이상건물수": 0,
            "30년이상건물수": 0,
            "50년이상건물수": 0,
        }
    )

    missing_mapping = 0
    invalid_approval = 0

    with SOURCE_PATH.open("r", encoding="utf-8-sig", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            full_legal_code = f"{str(row.get('시군구코드', '')).strip()}{str(row.get('법정동코드', '')).strip()}"
            mapped = mapping.get(full_legal_code)

            if mapped is None:
                missing_mapping += 1
                continue

            key = (mapped["구"], full_legal_code, mapped["법정동명"])
            grouped[key]["전체건물수"] += 1

            approval_date = parse_approval_date(row.get("사용승인일", ""))
            if approval_date is None:
                invalid_approval += 1
                continue

            grouped[key]["사용승인일유효건수"] += 1
            years = age_in_years(approval_date, AS_OF_DATE)

            if years < 10:
                grouped[key]["10년미만건물수"] += 1
            else:
                grouped[key]["10년이상건물수"] += 1

            if years >= 30:
                grouped[key]["30년이상건물수"] += 1
            if years >= 50:
                grouped[key]["50년이상건물수"] += 1

    rows_out: list[dict[str, object]] = []
    for (gu_name, legal_code, legal_dong_name), stats in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][2])):
        rows_out.append(
            {
                "기준일": AS_OF_DATE.isoformat(),
                "구": gu_name,
                "법정동코드": legal_code,
                "법정동명": legal_dong_name,
                **stats,
            }
        )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "기준일",
                "구",
                "법정동코드",
                "법정동명",
                "전체건물수",
                "사용승인일유효건수",
                "10년미만건물수",
                "10년이상건물수",
                "30년이상건물수",
                "50년이상건물수",
            ],
        )
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"Wrote: {OUT_PATH}")
    print(f"Rows: {len(rows_out)}")
    print(f"Skipped rows without legal-dong mapping: {missing_mapping}")
    print(f"Rows with invalid approval date: {invalid_approval}")


if __name__ == "__main__":
    main()
