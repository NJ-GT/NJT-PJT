from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "새 폴더"
OUTPUT_DIR = INPUT_DIR / "날짜별_concat"
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "ES1001AH00301MM_202403_202603_concat.csv"
SUMMARY_CSV = OUTPUT_DIR / "concat_summary.csv"


def read_pipe_csv(path: Path) -> pd.DataFrame:
    last_error: Exception | None = None
    for enc in ("cp949", "utf-8-sig", "utf-8"):
        try:
            return pd.read_csv(path, sep="|", encoding=enc, dtype=str)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Could not read {path}") from last_error


def ym_from_name(path: Path) -> str:
    match = re.search(r"(\d{4})_csv\.csv$", path.name)
    if not match:
        return ""
    yy_mm = match.group(1)
    return f"20{yy_mm}"


def main() -> None:
    files = sorted(INPUT_DIR.glob("ES1001AH00301MM*_csv.csv"))
    if not files:
        raise FileNotFoundError(f"No input CSV files found in {INPUT_DIR}")

    frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []

    for path in files:
        df = read_pipe_csv(path)
        source_ym = ym_from_name(path)
        df.insert(0, "source_file", path.name)
        df.insert(1, "source_ym", source_ym)
        frames.append(df)
        summary_rows.append(
            {
                "source_file": path.name,
                "source_ym": source_ym,
                "rows": len(df),
                "columns": len(df.columns),
            }
        )

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    summary = pd.DataFrame(summary_rows)
    summary.loc[len(summary)] = {
        "source_file": "TOTAL",
        "source_ym": f"{summary['source_ym'].min()}-{summary['source_ym'].max()}",
        "rows": int(summary["rows"].sum()),
        "columns": len(combined.columns),
    }
    summary.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")

    print(OUTPUT_CSV)
    print(SUMMARY_CSV)
    print(f"files={len(files)} rows={len(combined)} columns={len(combined.columns)}")


if __name__ == "__main__":
    main()
