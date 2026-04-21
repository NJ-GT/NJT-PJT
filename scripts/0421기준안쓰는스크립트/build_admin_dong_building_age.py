"""
[파일 설명]
서울 10개 자치구의 행정동별 평균 사용승인연도(건축연도)와 건물 나이를 계산하는 스크립트.

주요 역할:
  1. 통합숙박시설 피처 선정본 CSV에서 사용승인일을 파싱한다.
  2. KIKmix 엑셀에서 행정동 코드 → 행정동명 매핑을 로드한다.
  3. 행정동별로 평균 건물 나이(2026년 4월 15일 기준)를 계산하여 CSV로 저장한다.

입력: data/통합숙박시설표제부0415_피처선정본.csv   (숙박시설 피처 데이터)
      data/jscode20200515/KIKmix.20200515.xlsx     (행정동 코드 매핑)
출력: data/서울10개구_행정동별_평균사용승인연도_건물나이_0415.csv
"""

from __future__ import annotations

import csv
import datetime as dt
import statistics
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]  # 프로젝트 루트 (scripts의 부모)
DATA_DIR = ROOT / "data"

FEATURES_PATH = DATA_DIR / "통합숙박시설표제부0415_피처선정본.csv"                     # 입력: 숙박시설 피처
MAPPING_XLSX_PATH = DATA_DIR / "jscode20200515" / "jscode20200515" / "KIKmix.20200515.xlsx"  # 행정동 코드표
OUT_PATH = DATA_DIR / "서울10개구_행정동별_평균사용승인연도_건물나이_0415.csv"            # 출력

# 분석 대상 10개 자치구
TOP_10_GU = {
    "중구",
    "종로구",
    "영등포구",
    "강남구",
    "강서구",
    "마포구",
    "서초구",
    "송파구",
    "용산구",
    "성동구",
}

AS_OF_DATE = dt.date(2026, 4, 15)


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fp:
        return list(csv.DictReader(fp))


def load_kikmix_mapping(path: Path) -> dict[str, dict[str, str]]:
    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

    with zipfile.ZipFile(path) as zf:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in root.findall("a:si", ns):
                texts = [t.text or "" for t in si.findall(".//a:t", ns)]
                shared_strings.append("".join(texts))

        sheet = ET.fromstring(zf.read("xl/worksheets/sheet1.xml"))
        rows = sheet.findall(".//a:sheetData/a:row", ns)

        def cell_value(cell: ET.Element) -> str:
            value = cell.find("a:v", ns)
            if value is None:
                return ""
            if cell.get("t") == "s":
                return shared_strings[int(value.text)]
            return value.text or ""

        header: list[str] = []
        mapping: dict[str, dict[str, str]] = {}
        for row_idx, row in enumerate(rows):
            values = [cell_value(cell) for cell in row.findall("a:c", ns)]
            if row_idx == 0:
                header = values
                continue
            if not header or len(values) < 6:
                continue
            record = {header[i]: values[i] if i < len(values) else "" for i in range(len(header))}
            if record.get("시도명") != "서울특별시":
                continue
            if record.get("시군구명") not in TOP_10_GU:
                continue
            if record.get("말소일자"):
                continue
            legal_code = str(record.get("법정동코드", "")).strip()
            admin_dong = str(record.get("읍면동명", "")).strip()
            if legal_code and admin_dong:
                mapping[legal_code] = {
                    "시군구명": str(record.get("시군구명", "")).strip(),
                    "행정동명": admin_dong,
                    "행정동코드": str(record.get("행정동코드", "")).strip(),
                    "법정동명": str(record.get("동리명", "")).strip(),
                }
        return mapping


def parse_approval_date(value: str) -> dt.date | None:
    text = str(value or "").strip()
    if len(text) != 8 or not text.isdigit():
        return None
    try:
        return dt.date(int(text[:4]), int(text[4:6]), int(text[6:8]))
    except ValueError:
        return None


def round_years(date_value: dt.date, as_of: dt.date) -> float:
    return round((as_of - date_value).days / 365.2425, 2)


def main() -> None:
    rows = load_csv(FEATURES_PATH)
    mapping = load_kikmix_mapping(MAPPING_XLSX_PATH)

    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    unmapped = 0
    invalid_dates = 0

    for row in rows:
        gu_code = str(row.get("시군구코드", "")).strip()
        legal_code_tail = str(row.get("법정동코드", "")).strip()
        full_legal_code = f"{gu_code}{legal_code_tail}"

        approval_date = parse_approval_date(row.get("사용승인일", ""))
        if approval_date is None:
            invalid_dates += 1
            continue

        mapping_row = mapping.get(full_legal_code)
        if mapping_row is None:
            unmapped += 1
            continue

        gu_name = mapping_row["시군구명"]
        admin_dong = mapping_row["행정동명"]
        grouped[(gu_name, admin_dong)].append(
            {
                "approval_date": approval_date,
                "approval_year": approval_date.year,
                "building_age_years": round_years(approval_date, AS_OF_DATE),
                "법정동명": mapping_row["법정동명"],
                "행정동코드": mapping_row["행정동코드"],
            }
        )

    out_rows: list[dict[str, object]] = []
    for (gu_name, admin_dong), items in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1])):
        approval_years = [item["approval_year"] for item in items]
        ages = [item["building_age_years"] for item in items]
        approval_dates = [item["approval_date"] for item in items]
        legal_dongs = sorted({str(item["법정동명"]) for item in items if item["법정동명"]})
        admin_codes = sorted({str(item["행정동코드"]) for item in items if item["행정동코드"]})

        out_rows.append(
            {
                "기준일": AS_OF_DATE.isoformat(),
                "구": gu_name,
                "행정동명": admin_dong,
                "행정동코드": admin_codes[0] if admin_codes else "",
                "건물수": len(items),
                "평균사용승인연도": round(statistics.mean(approval_years), 2),
                "평균건물나이_년": round(statistics.mean(ages), 2),
                "최소사용승인일": min(approval_dates).isoformat(),
                "최대사용승인일": max(approval_dates).isoformat(),
                "포함법정동명": " / ".join(legal_dongs),
            }
        )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "기준일",
                "구",
                "행정동명",
                "행정동코드",
                "건물수",
                "평균사용승인연도",
                "평균건물나이_년",
                "최소사용승인일",
                "최대사용승인일",
                "포함법정동명",
            ],
        )
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"Wrote: {OUT_PATH}")
    print(f"Rows: {len(out_rows)}")
    print(f"Skipped invalid approval dates: {invalid_dates}")
    print(f"Skipped unmapped legal dong codes: {unmapped}")


if __name__ == "__main__":
    main()
