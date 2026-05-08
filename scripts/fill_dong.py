# -*- coding: utf-8 -*-
"""
통합숙박시설 데이터에서 '동' 컬럼이 비어 있는 행을, 주소 문자열에서 추출해 보충하는 스크립트.

전제:
    한국 주소 패턴은 보통 "서울특별시 ○○구 △△동 …" 순서.
    '구'로 끝나는 토큰을 찾고, 그 다음 토큰이 '동' 단위일 가능성이 가장 높다는 휴리스틱을 사용.

주의:
    완벽한 행정동 파싱이 아니라 빠른 보강용. 서울 외 주소나 비표준 입력에는 한계가 있음.
"""

import pandas as pd
import sys

# Windows 콘솔 한글 깨짐 방지
sys.stdout.reconfigure(encoding="utf-8")

# 원본 데이터 로드
df = pd.read_csv("data/통합숙박시설_최종안0421.csv", encoding="utf-8-sig")
# 처리 전 결측 행 수
print(f"처리 전 null: {df['동'].isna().sum()}행")


def extract_dong(addr):
    """주소 문자열에서 '○○구' 다음에 오는 토큰을 동 후보로 추출."""
    if pd.isna(addr):
        return None
    # 공백 단위 토큰화
    parts = str(addr).strip().split()
    # 예: '서울특별시 종로구 종로2가 9번지' → '구' 다음 토큰이 '동' 후보
    for i, p in enumerate(parts):
        # '시' 사용 안 함 — '서울특별시'와 충돌 위험
        if p.endswith("구"):
            if i + 1 < len(parts):
                return parts[i + 1]
    return None


# 동 결측 마스크 — 결측 행만 부분 업데이트
mask = df["동"].isna()
df.loc[mask, "동"] = df.loc[mask, "주소"].apply(extract_dong)

# 처리 후 결측 행 수 (얼마나 줄었는지 체크)
print(f"처리 후 null: {df['동'].isna().sum()}행")
print("\n[샘플 확인]")
# 보강된 행 일부를 출력 — 파싱 결과 확인
print(df.loc[mask, ["구", "동", "주소"]].head(15).to_string())

# 같은 파일에 덮어쓰기 (UTF-8 BOM)
df.to_csv("data/통합숙박시설_최종안0421.csv", index=False, encoding="utf-8-sig")
print("\n저장 완료")
