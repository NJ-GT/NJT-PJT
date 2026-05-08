# -*- coding: utf-8 -*-
"""
마포구 연남동·서교동 두 동의 핵심 변수 분포를 describe로 비교 출력하는 스크립트.

목적:
    "마포구 화재 안전 정책 검토" 맥락에서 연남동/서교동을 클러스터별 비율과 함께
    동 단위 통계로 빠르게 비교한다.
출력:
    콘솔에 동별 건수/cluster 분포, 각 동별 describe, 평균 요약을 차례로 출력 (별도 저장 X)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


# scripts/ 기준 한 단계 위 (NJT-PJT/)
ROOT = Path(__file__).resolve().parents[1]
# 입력 — 0430/ 폴더의 최종테이블
DATA_PATH = ROOT / "0430" / "최종테이블0429.csv"


def main() -> None:
    """엔트리 포인트 — 두 동 데이터 추리고 describe 출력."""
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    # 컬럼 양끝 공백 제거 — 헤더에 공백이 섞이는 경우 방지
    df.columns = df.columns.str.strip()

    # 비교 대상 동
    target_dongs = ["연남동", "서교동"]
    # 마포구 + 위 두 동만 필터
    subset = df[df["구"].eq("마포구") & df["동"].isin(target_dongs)].copy()

    # describe 대상 핵심 변수 (수치형 강제 변환)
    cols = [
        "최종위험점수_new",
        "구조노후도",
        "단속위험도",
        "도로폭위험도",
        "최근접_소화용수_거리등급",
        "소방위험도_점수",
        "승인연도",
        "연면적",
        "집중도",
        "주변건물수",
        "총층수",
    ]
    for col in cols:
        subset[col] = pd.to_numeric(subset[col], errors="coerce")

    # 전체 행 수 + 동 × 클러스터 분포 표
    print(f"rows: {len(subset):,}")
    print("\n[동별 건수 / 위험군 구성]")
    print(pd.crosstab(subset["동"], subset["cluster_label"]).to_string())

    # 동별로 describe 출력 (행: 통계, 열: 변수)
    print("\n[동별 주요 변수 describe]")
    for dong in target_dongs:
        part = subset[subset["동"].eq(dong)]
        print(f"\n=== {dong} n={len(part):,} ===")
        desc = (
            part[cols]
            .describe()
            .T[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]
        )
        print(desc.round(4).to_string())

    # 두 동 평균값을 한 표로 — reindex로 표시 순서 고정
    print("\n[동별 평균 요약]")
    print(subset.groupby("동")[cols].mean().round(4).reindex(target_dongs).to_string())


# 직접 실행 시에만 main 호출
if __name__ == "__main__":
    main()
