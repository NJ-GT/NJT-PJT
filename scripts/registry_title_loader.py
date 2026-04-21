"""
[파일 설명]
등기부등본 표제부 CSV 파일들을 로드하고 통합하는 유틸리티 모듈.

다른 스크립트에서 `from registry_title_loader import load_registry` 형태로 임포트하여 사용한다.
여러 구별 표제부 CSV(등기부등본_표제부_강남.csv 등)를 자동 탐색하여 하나로 합친다.

주요 함수:
  - load_registry()       : 통합 CSV가 있으면 그것을 읽고, 없으면 개별 파일들을 합쳐 반환
  - build_merged_registry(): 개별 파일들을 합쳐 통합 CSV로 저장
  - discover_registry_title_files(): '표제부'가 포함된 CSV 파일들을 자동 탐색

인코딩: utf-8-sig → cp949 → euc-kr 순서로 시도
"""

from pathlib import Path
import csv

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent          # 프로젝트 루트
MERGED_REGISTRY_PATH = BASE_DIR / "data" / "등기부등본_표제부_통합.csv"  # 통합 파일 경로
READ_ENCODINGS = ("utf-8-sig", "cp949", "euc-kr")          # 인코딩 시도 순서


def _read_csv(path, **kwargs):
    """여러 인코딩을 순서대로 시도하여 CSV를 읽는다. 모두 실패하면 마지막 에러를 올린다."""
    last_error = None
    for enc in READ_ENCODINGS:
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except Exception as exc:
            last_error = exc
    raise last_error


def _read_header(path):
    """CSV 파일의 첫 번째 행(헤더)만 읽어 반환한다. 여러 인코딩 시도."""
    last_error = None
    for enc in READ_ENCODINGS:
        try:
            with open(path, "r", encoding=enc, newline="") as fp:
                return next(csv.reader(fp))
        except Exception as exc:
            last_error = exc
    raise last_error


def discover_registry_title_files(base_dir=BASE_DIR):
    """
    BASE_DIR에서 '표제부'가 이름에 포함된 CSV 파일을 탐색한다.
    같은 헤더 구조를 가진 파일들 중 가장 많은 그룹을 반환한다 (구별 분할 파일 자동 탐지).
    """
    # '표제부' 이름이 포함된 CSV 파일 전체 목록
    candidates = sorted(
        path for path in Path(base_dir).glob("*.csv") if "표제부" in path.name
    )
    if not candidates:
        raise FileNotFoundError("표제부 CSV 파일을 찾지 못했습니다.")

    # 헤더가 동일한 파일끼리 그룹화 (같은 스키마인 파일들을 합쳐야 하므로)
    header_groups = {}
    for path in candidates:
        header = tuple(_read_header(path))
        header_groups.setdefault(header, []).append(path)

    # 파일 수가 가장 많은 그룹 선택
    _, matched = max(
        header_groups.items(), key=lambda item: (len(item[1]), len(item[0]))
    )
    if not matched:
        raise FileNotFoundError("같은 표제부 스키마를 가진 CSV 파일을 찾지 못했습니다.")
    return sorted(matched)


def load_registry(prefer_merged=True, strip_columns=True, **read_csv_kwargs):
    """
    등기부등본 표제부 데이터를 DataFrame으로 반환한다.
    통합 파일(등기부등본_표제부_통합.csv)이 있으면 그것을 우선 읽고,
    없으면 개별 파일들을 자동 탐색하여 합친다.
    """
    if prefer_merged and MERGED_REGISTRY_PATH.exists():
        reg = _read_csv(MERGED_REGISTRY_PATH, **read_csv_kwargs)
        if strip_columns:
            reg.columns = reg.columns.str.strip()  # 열 이름 공백 제거
        return reg

    # 개별 파일 탐색 후 병합
    files = discover_registry_title_files()
    frames = []
    for path in files:
        df = _read_csv(path, **read_csv_kwargs)
        if strip_columns:
            df.columns = df.columns.str.strip()
        frames.append(df)
    return pd.concat(frames, ignore_index=True)  # 인덱스 재정렬하여 합치기


def build_merged_registry(output_path=MERGED_REGISTRY_PATH, dedupe_on="관리건축물대장PK"):
    """
    개별 표제부 파일들을 합쳐 통합 CSV를 만들고 저장한다.
    PK 기준으로 중복 행을 제거한다.
    """
    files = discover_registry_title_files()
    reg = load_registry(prefer_merged=False, low_memory=False)  # 통합 파일 무시하고 개별 파일 읽기

    # 관리건축물대장PK 중복 제거
    duplicate_rows_removed = 0
    if dedupe_on in reg.columns:
        before = len(reg)
        reg = reg.drop_duplicates(subset=[dedupe_on]).copy()
        duplicate_rows_removed = before - len(reg)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    reg.to_csv(output_path, index=False, encoding="utf-8-sig")

    return {
        "path": str(output_path),
        "file_count": len(files),
        "row_count": len(reg),
        "column_count": len(reg.columns),
        "duplicate_rows_removed": duplicate_rows_removed,
    }


if __name__ == "__main__":
    info = build_merged_registry()
    print(
        f"통합 완료: {info['file_count']}개 파일 -> "
        f"{info['row_count']}행 x {info['column_count']}컬럼 "
        f"({info['duplicate_rows_removed']}행 중복 제거)"
    )
