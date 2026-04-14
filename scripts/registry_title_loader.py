from pathlib import Path
import csv

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
MERGED_REGISTRY_PATH = BASE_DIR / "data" / "등기부등본_표제부_통합.csv"
READ_ENCODINGS = ("utf-8-sig", "cp949", "euc-kr")


def _read_csv(path, **kwargs):
    last_error = None
    for enc in READ_ENCODINGS:
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except Exception as exc:
            last_error = exc
    raise last_error


def _read_header(path):
    last_error = None
    for enc in READ_ENCODINGS:
        try:
            with open(path, "r", encoding=enc, newline="") as fp:
                return next(csv.reader(fp))
        except Exception as exc:
            last_error = exc
    raise last_error


def discover_registry_title_files(base_dir=BASE_DIR):
    candidates = sorted(
        path for path in Path(base_dir).glob("*.csv") if "표제부" in path.name
    )
    if not candidates:
        raise FileNotFoundError("표제부 CSV 파일을 찾지 못했습니다.")

    header_groups = {}
    for path in candidates:
        header = tuple(_read_header(path))
        header_groups.setdefault(header, []).append(path)

    _, matched = max(
        header_groups.items(), key=lambda item: (len(item[1]), len(item[0]))
    )
    if not matched:
        raise FileNotFoundError("같은 표제부 스키마를 가진 CSV 파일을 찾지 못했습니다.")
    return sorted(matched)


def load_registry(prefer_merged=True, strip_columns=True, **read_csv_kwargs):
    if prefer_merged and MERGED_REGISTRY_PATH.exists():
        reg = _read_csv(MERGED_REGISTRY_PATH, **read_csv_kwargs)
        if strip_columns:
            reg.columns = reg.columns.str.strip()
        return reg

    files = discover_registry_title_files()
    frames = []
    for path in files:
        df = _read_csv(path, **read_csv_kwargs)
        if strip_columns:
            df.columns = df.columns.str.strip()
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def build_merged_registry(output_path=MERGED_REGISTRY_PATH, dedupe_on="관리건축물대장PK"):
    files = discover_registry_title_files()
    reg = load_registry(prefer_merged=False, low_memory=False)

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
