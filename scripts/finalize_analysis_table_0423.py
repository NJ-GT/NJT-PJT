# -*- coding: utf-8 -*-
"""
분석변수 최종 테이블(0423) 생성 스크립트.

목적:
    build_building_feature_probe.py 가 만든 probe CSV(건물 특성이 부착된 중간 표)를
    분석에 바로 쓸 수 있는 컬럼 집합으로 정제하여 최종 테이블로 저장한다.

핵심 정제:
    - 건물용도명 결측은 '미상'으로 채움
    - 총층수: 0층 보정 후 정수화 (결측은 1)
    - 연면적: 수치형으로 변환 (결측은 0.0)
    - 다운스트림 모델에 필수인 컬럼은 사후에 NaN이 없는지 검증
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# probe 단계에서 쓰는 빌드 함수 — probe CSV가 없으면 즉석에서 생성
from build_building_feature_probe import build_probe


# scripts 기준 한 단계 위 (NJT-PJT/)
ROOT = Path(__file__).resolve().parents[1]
# probe 산출물 (입력)
PROBE_PATH = (
    ROOT / "0424" / "분석" / "tables" / "분석변수_최종테이블0423_건물특성_probe.csv"
)
# 최종 테이블 (출력)
OUTPUT_PATH = ROOT / "0424" / "data" / "분석변수_최종테이블0423.csv"


def prepare_output_table(df: pd.DataFrame) -> pd.DataFrame:
    """probe DataFrame -> 최종 테이블 스키마로 정제."""
    out = pd.DataFrame()

    # 그대로 복사할 식별/위치/위험 변수
    base_cols = [
        "구",
        "동",
        "숙소명",
        "승인연도",
        "주변건물수",
        "집중도",
        "단속위험도",
        "구조노후도",
        "도로폭위험도",
        "위도",
        "경도",
        "업종",
    ]
    for col in base_cols:
        out[col] = df[col]

    # 통합된 '건물용도명' — 결측은 '미상'으로 채움
    out["건물용도명"] = df["건물용도명_통합"].fillna("미상").astype(str)
    # 0층 보정된 총층수 — 결측 1, 반올림 후 int 변환
    out["총층수"] = (
        pd.to_numeric(df["총층수_0층만보정"], errors="coerce")
        .fillna(1)
        .round()
        .astype(int)
    )
    # 통합된 연면적 — 결측 0.0
    out["연면적"] = pd.to_numeric(df["연면적_통합"], errors="coerce").fillna(0.0)

    # 후속 모델/지표 계산이 NaN에 민감 — 필수 컬럼은 NaN이 없어야 함
    required_non_null = ["건물용도명", "총층수", "연면적"]
    for col in required_non_null:
        if out[col].isna().any():
            raise ValueError(f"Required column still has nulls: {col}")

    return out


def main() -> None:
    """엔트리 — probe가 없으면 만들고, 정제 후 CSV 저장."""
    if not PROBE_PATH.exists():
        build_probe()

    probe_df = pd.read_csv(PROBE_PATH, encoding="utf-8-sig")
    final_df = prepare_output_table(probe_df)
    final_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    # 산출 정보 출력
    print(f"Saved: {OUTPUT_PATH}")
    print(f"Rows: {len(final_df)}")
    print(f"Columns: {len(final_df.columns)}")
    # 핵심 컬럼의 NaN 개수 — 위 검증 후라 모두 0이어야 정상
    print(
        {
            "건물용도명_nulls": int(final_df["건물용도명"].isna().sum()),
            "총층수_nulls": int(final_df["총층수"].isna().sum()),
            "연면적_nulls": int(final_df["연면적"].isna().sum()),
        }
    )


if __name__ == "__main__":
    main()
