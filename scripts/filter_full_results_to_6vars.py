# -*- coding: utf-8 -*-
"""GWR/MGWR 전체 결과 CSV에서 핵심 6변수만 남기는 후처리 스크립트.

배경:
    GWR/MGWR 결과 CSV에는 변수마다 coef_/tval_/bw_/z_/contrib_ 등의 컬럼이 펼쳐진다.
    분석 결론에서 채택된 6개 변수만 남기고, 나머지 변수의 모든 컬럼을 일괄 제거한다.

대상 파일 (in-place 갱신):
    data/full_gwr_mgwr/gwr_results_full.csv
    data/full_gwr_mgwr/mgwr_results_full.csv

남기는 변수:
    구조노후도, 단속위험도, 도로폭위험도, 최근접_소화용수_거리등급,
    소방위험도_점수, 집중도

제거 변수:
    승인연도, 연면적, 주변건물수, 총층수
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


# scripts 기준 한 단계 위 (NJT-PJT/)
BASE = Path(__file__).resolve().parents[1]
RESULT_DIR = BASE / "data" / "full_gwr_mgwr"
# 후처리 대상 두 파일
FILES = [
    RESULT_DIR / "gwr_results_full.csv",
    RESULT_DIR / "mgwr_results_full.csv",
]

# 유지 변수 — 6개
KEEP_VARIABLES = {
    "구조노후도",
    "단속위험도",
    "도로폭위험도",
    "최근접_소화용수_거리등급",
    "소방위험도_점수",
    "집중도",
}

# 명시적으로 제거 변수 — 안전장치(이중 검사)
DROP_VARIABLES = {
    "승인연도",
    "연면적",
    "주변건물수",
    "총층수",
}

# 변수 prefix가 없는 식별/공통 컬럼 — 무조건 보존
BASE_COLUMNS = {
    "구",
    "동",
    "숙소명",
    "위도",
    "경도",
    "x_5181",
    "y_5181",
    "최종위험점수_new",
    "local_R2",
    "bandwidth",
    "residual",
    "coef_intercept",
    "tval_intercept",
    "bw_intercept",
}

# 변수별 컬럼은 이 prefix 중 하나로 시작 (coef_도로폭위험도 등)
VARIABLE_PREFIXES = ("coef_", "tval_", "bw_", "z_", "contrib_")


def variable_name(column: str) -> str | None:
    """컬럼명에서 prefix를 떼고 변수명만 추출 (해당 안 되면 None)."""
    for prefix in VARIABLE_PREFIXES:
        if column.startswith(prefix):
            return column.removeprefix(prefix)
    return None


def keep_column(column: str) -> bool:
    """주어진 컬럼을 유지할지 결정."""
    # 공통 식별 컬럼은 무조건 유지
    if column in BASE_COLUMNS:
        return True
    var = variable_name(column)
    # prefix가 없는데 BASE_COLUMNS도 아니면 안전하게 제거
    if var is None:
        return False
    # 명시 제거 변수면 즉시 제거 (이중 안전망)
    if var in DROP_VARIABLES:
        return False
    # KEEP에 들어 있을 때만 유지
    return var in KEEP_VARIABLES


def main() -> None:
    """두 파일에 대해 컬럼 필터링 후 in-place 덮어쓰기."""
    for path in FILES:
        df = pd.read_csv(path, encoding="utf-8-sig")
        keep_cols = [col for col in df.columns if keep_column(col)]
        # 모델 계수가 모두 사라졌다면 잘못된 입력 — 사용자가 원본 파일을 다시 두도록 안내
        if not any(col.startswith("coef_") for col in keep_cols):
            raise RuntimeError(
                f"{path} does not contain model coefficient columns. "
                "Use the original full result CSV, then run this script again."
            )
        filtered = df[keep_cols]
        # 같은 경로에 덮어쓰기 (UTF-8 BOM)
        filtered.to_csv(path, index=False, encoding="utf-8-sig")
        # 변경 요약 출력 — 컬럼 수 / 행 수 / 최종 컬럼 목록
        print(
            f"{path}: {len(df.columns)} -> {len(filtered.columns)} columns, rows={len(filtered):,}"
        )
        print(filtered.columns.tolist())


if __name__ == "__main__":
    main()
