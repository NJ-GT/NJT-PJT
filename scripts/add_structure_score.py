# -*- coding: utf-8 -*-
"""
숙박시설 '기타구조' 정보를 활용해 구조_노후_통합점수를 산출하고 핵심 테이블들에 반영하는 스크립트.

산출 로직:
    구조_위험점수(1~7) + 노후도_점수(MinMax) + 상호작용항(MinMax) → MinMax 재정규화
    (목조=7, 샌드위치판넬=6, 경량철골=5, 조적/연와=4, 일반철골=3, RC=2, SRC=1)

처리 흐름:
    1. 통합숙박시설최종안0415.csv 에서 '기타구조' 컬럼을 메인 테이블에 병합
       (사업장명+대지위치 키 → 미매칭 시 사업장명만으로 보조 매칭)
    2. 기타구조 문자열을 정규식 매칭해 구조_위험점수(1~7) 산출
       (복합재 표기 시 가장 취약한 소재 우선)
    3. 상호작용항 = 구조_위험점수 × max(건물나이, 1) 후 MinMax 정규화
    4. 세 점수를 합산 → 0.1~1.0 범위로 재정규화 (구조_노후_통합점수)
    5. 메인 CSV (서울10구_숙소_소방거리_유클리드.csv) 갱신
    6. 핵심분석변수 / 도로폭추가 류 테이블에도 위경도 또는 업소명 키로 병합 갱신
"""

# 데이터프레임 처리
import pandas as pd
# 정규식 (구조 문자열 패턴 매칭)
import re
# 파일 일괄 처리용 와일드카드 매칭
import glob
import sys

# 콘솔 출력 인코딩을 UTF-8 로 강제 (Windows cp949 환경 대응)
sys.stdout.reconfigure(encoding="utf-8")

# 프로젝트 루트 경로 (Windows 경로 슬래시 통일)
BASE = "c:/Users/USER/Documents/GitHub/기말공모전/NJT-PJT"
# 메인 분석 테이블 (서울 10구 숙소 + 소방거리)
df = pd.read_csv(
    f"{BASE}/data/서울10구_숙소_소방거리_유클리드.csv",
    encoding="utf-8-sig",
    on_bad_lines="skip",  # 깨진 라인은 건너뜀
)
# 기타구조 정보를 가진 원천 (통합숙박시설최종안0415)
src = pd.read_csv(
    f"{BASE}/data/통합숙박시설최종안0415.csv", encoding="utf-8-sig", on_bad_lines="skip"
)

# ── 1. 기타구조 병합 (복합재 → 가장 취약한 소재 기준) ─────────────────
# 매칭 키: 사업장명+대지위치 / 메인은 업소명+주소
src["_key"] = src["사업장명"].str.strip() + "|" + src["대지위치"].str.strip()
df["_key"] = df["업소명"].str.strip() + "|" + df["주소"].str.strip()
# src 측 중복 제거(첫 행 유지)
src_dedup = src[["_key", "기타구조"]].drop_duplicates(subset="_key", keep="first")
# 이미 존재할 수 있는 점수 컬럼 제거 (중복 방지)
df = df.drop(
    columns=["기타구조", "구조_위험점수", "구조_노후_상호작용", "구조_노후_통합점수"],
    errors="ignore",
)
# 키로 좌측 조인 → 기타구조 컬럼 부착
merged = df.merge(src_dedup, on="_key", how="left")

# 미매칭(예: 업소명에 쉼표 포함되어 키가 깨진 경우) → 사업장명 단독 매칭으로 보강
src_name = src[["사업장명", "기타구조"]].drop_duplicates("사업장명", keep="first")
for idx in merged[merged["기타구조"].isna()].index:
    hit = src_name[src_name["사업장명"].str.strip() == merged.at[idx, "업소명"].strip()]
    if not hit.empty:
        merged.at[idx, "기타구조"] = hit.iloc[0]["기타구조"]
# 매칭 진단 출력
print(f"기타구조 매칭: {merged['기타구조'].notna().sum()}/{len(merged)}")


# ── 2. 구조_위험점수 (1~7, 높을수록 취약, 복합재는 취약 소재 우선) ──────
def get_score(s):
    """기타구조 문자열을 기반으로 1~7 점 위험 점수 반환.

    - 결측/미상은 중간값 3 반환
    - 복합 표기는 위→아래 순서로 매칭(가장 취약한 소재가 먼저 잡히도록)
    """
    if pd.isna(s):
        return 3
    # 공백 제거 + 대문자화 (정규식 안정화)
    s = str(s).replace(" ", "").upper()
    if re.search(r"목조|목구조", s):
        return 7  # 목구조: 가장 취약
    if re.search(r"샌드위치|판넬|패널", s):
        return 6  # 샌드위치판넬
    if re.search(r"경량철골", s):
        return 5  # 경량철골
    if re.search(r"연와|벽돌|조적|세멘|시멘트벽돌|부럭|부록", s):
        return 4  # 조적/연와
    if re.search(r"일반철골|철골구조", s):
        return 3  # 일반철골
    if re.search(r"철근콘크리트|R\.C|RC조|라멘|벽식|콘크리트", s):
        return 2  # RC
    if re.search(r"SRC|철골철근콘크리트", s):
        return 1  # SRC: 가장 안전
    return 3  # 패턴 미일치 시 중간값


# 위험 점수 컬럼 생성
merged["구조_위험점수"] = merged["기타구조"].apply(get_score)

# ── 3. 상호작용항: 구조_위험점수 × max(건물나이, 1), MinMax 정규화 ──────
# 신축(나이=0)도 0 이 되지 않도록 하한 1
interact_raw = merged["구조_위험점수"] * merged["건물나이"].clip(lower=1)
mn, mx = interact_raw.min(), interact_raw.max()
# 0~1 정규화
interact_norm = ((interact_raw - mn) / (mx - mn)).round(4)
# 0 값을 0이 아닌 최소값으로 치환(가중치가 0 이 되지 않도록)
min_nonzero = interact_norm[interact_norm > 0].min()
merged["구조_노후_상호작용"] = interact_norm.replace(0.0, min_nonzero)

# ── 4. 구조_노후_통합점수: 3개 합산 후 MinMax 재정규화 ───────────────────
# 합산 점수
raw_combined = (
    merged["구조_위험점수"] + merged["노후도_점수"] + merged["구조_노후_상호작용"]
)
mn2, mx2 = raw_combined.min(), raw_combined.max()
# 0.1 ~ 1.0 범위로 변환 (하한을 0으로 두지 않고 0.1 보장)
merged["구조_노후_통합점수"] = (0.1 + (raw_combined - mn2) / (mx2 - mn2) * 0.9).round(4)

# 임시 키 컬럼 제거
merged = merged.drop(columns=["_key"])

# 결과 진단
print("\n구조_노후_통합점수 describe:")
print(merged["구조_노후_통합점수"].describe())
print("\n샘플:")
print(
    merged[["업소명", "기타구조", "건물나이", "구조_위험점수", "구조_노후_통합점수"]]
    .head(8)
    .to_string()
)

# ── 5. 메인 CSV 저장 ──────────────────────────────────────────────────
merged.to_csv(
    f"{BASE}/data/서울10구_숙소_소방거리_유클리드.csv",
    index=False,
    encoding="utf-8-sig",
)
print("\n[저장] 서울10구_숙소_소방거리_유클리드.csv")

# ── 6. 핵심 테이블들 갱신 ─────────────────────────────────────────────
# 위경도 6자리 반올림을 키로 잡기 위한 보조 테이블
src_upd = merged[["위도", "경도", "구조_노후_통합점수"]].drop_duplicates(
    ["위도", "경도"]
)
src_upd["_key"] = (
    src_upd["위도"].round(6).astype(str) + "|" + src_upd["경도"].round(6).astype(str)
)

# data/ 안 모든 csv 중 '핵심분析변수' / '도로폭추가' 류 테이블만 대상으로 갱신
for f in glob.glob(f"{BASE}/data/*.csv"):
    fname = f.replace("\\", "/").split("/")[-1]
    if "핵심분析변수" not in fname and "도로폭추가" not in fname:
        continue
    t = pd.read_csv(f, encoding="utf-8-sig", on_bad_lines="skip")
    # 기존 점수 컬럼 모두 제거
    t = t.drop(
        columns=[
            "구조_노후_통합점수",
            "구조_위험점수",
            "노후도_점수",
            "구조_노후_상호작용",
        ],
        errors="ignore",
    )
    # 위경도 컬럼이 있으면 좌표 키로 병합, 없으면 업소명+주소 키로 병합
    if "위도" in t.columns:
        t["_key"] = (
            t["위도"].round(6).astype(str) + "|" + t["경도"].round(6).astype(str)
        )
        t = t.merge(
            src_upd[["_key", "구조_노후_통합점수"]], on="_key", how="left"
        ).drop(columns=["_key"])
    else:
        # 업소명+주소로 보조 매칭 테이블 구성
        nm = merged[["업소명", "주소", "구조_노후_통합점수"]].copy()
        nm["_key"] = nm["업소명"].str.strip() + "|" + nm["주소"].str.strip()
        t["_key"] = t["업소명"].str.strip() + "|" + t["주소"].str.strip()
        t = t.merge(
            nm[["_key", "구조_노후_통합점수"]].drop_duplicates("_key"),
            on="_key",
            how="left",
        ).drop(columns=["_key"])
    # 갱신된 테이블 덮어쓰기
    t.to_csv(f, index=False, encoding="utf-8-sig")
    print(f"[저장] {fname}")
