# -*- coding: utf-8 -*-
"""
[파일 설명]
화재출동 데이터에서 사상자(사망 또는 부상 1명 이상) 발생 건만 추출하는 스크립트.

필터 조건: 사망자수 >= 1  OR  부상자수 >= 1

입력: data/화재출동/화재출동_2021_2024.csv   (merge_fire.py가 생성한 통합 출동 데이터)
출력: data/화재출동/화재출동_사상자발생.csv   (사상자 발생 건만 필터링된 결과)

실행 순서: merge_fire.py → filter_casualties.py
"""

import pandas as pd, os

BASE = r'C:\Users\USER\Documents\GitHub\기말공모전\NJT-PJT'  # 프로젝트 루트 경로
IN   = os.path.join(BASE, 'data', '화재출동', '화재출동_2021_2024.csv')  # 통합 출동 데이터
OUT  = os.path.join(BASE, 'data', '화재출동', '화재출동_사상자발생.csv')  # 출력 파일

# ─── 데이터 로드 ─────────────────────────────────────────────────
df = pd.read_csv(IN, encoding='utf-8-sig', low_memory=False)
print(f'전체: {len(df)}행')

# 사망/부상자 수를 숫자로 변환 (문자열이나 빈값이 있을 수 있으므로 coerce 처리)
df['사망자수'] = pd.to_numeric(df['사망자수'], errors='coerce').fillna(0)
df['부상자수'] = pd.to_numeric(df['부상자수'], errors='coerce').fillna(0)

# ─── 사상자 발생 건 필터링 ─────────────────────────────────────────
result = df[(df['사망자수'] >= 1) | (df['부상자수'] >= 1)].copy()
print(f'사상자 발생: {len(result)}행')
print(f'  - 사망자수 1명 이상: {(result["사망자수"] >= 1).sum()}건')
print(f'  - 부상자수 1명 이상: {(result["부상자수"] >= 1).sum()}건')

result.to_csv(OUT, index=False, encoding='utf-8-sig')
print(f'저장: {OUT}')
