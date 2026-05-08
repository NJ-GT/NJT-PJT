"""
[파일 설명]
집계구별 화재위험도 점수를 계산하고, 소방서/안전센터 위치를 추출하는 핵심 분석 스크립트.

주요 역할:
    1. 통합숙박시설 CSV 에서 건축 노후도, 건폐율, 용적률, 층수를 읽어온다.
    2. 숙박시설 포인트를 집계구 경계 안으로 공간결합(spatial join).
    3. 집계구별로 건축 지표를 평균 내고 화재위험점수(0~100)를 계산한다.
       화재위험점수 = 노후도(30%) + 건폐율(25%) + 용적률(25%) + 층수(20%)
    4. 화재출동 이력 CSV 에서 소방서/안전센터 위치를 추출한다.
    5. oa_density.json 에 결과를 덮어써 업데이트, firestation_data.json 새로 생성.

입력:
    data/통합숙박시설최종안0415.csv         (숙박시설 4,246개)
    data/화재출동/화재출동_2021_2024.csv      (화재출동 이력)
    data/bnd_oa_11_2025_2Q/bnd_oa_11_2025_2Q.shp (집계구 경계)
    data/oa_density.json                    (기존 집계구 데이터, 덮어씀)
출력:
    data/oa_density.json                    (화재위험점수 추가된 버전)
    data/firestation_data.json               (소방서·안전센터 위치 리스트)
"""

import sys
import json
import pandas as pd
import numpy as np
import geopandas as gpd
from pyproj import Transformer

# 콘솔 UTF-8 출력
sys.stdout.reconfigure(encoding="utf-8")

# ─── 1. 숙박시설 CSV 로드 ───────────────────────────────────────
print("1. 숙박시설 데이터 로드...")
df = pd.read_csv("data/통합숙박시설최종안0415.csv", encoding="utf-8-sig")
cols = df.columns.tolist()  # 위치 인덱스로 컬럼 접근하기 위해 보존

# EPSG:5181(한국 중부원점 TM) → EPSG:4326(WGS84) 좌표 변환
tf = Transformer.from_crs("EPSG:5181", "EPSG:4326", always_xy=True)
# CSV 0번(X) / 1번(Y) 열을 위경도로 변환
xs, ys = tf.transform(df[cols[0]].values, df[cols[1]].values)
df["lng"] = xs  # 경도
df["lat"] = ys  # 위도

# 분석에 필요한 수치형 컬럼 추출 (errors='coerce': 변환 실패 시 NaN)
df["연면적"] = pd.to_numeric(df[cols[11]], errors="coerce")  # 11번: 연면적(㎡)
df["층수"] = pd.to_numeric(df[cols[16]], errors="coerce")  # 16번: 지상층수
df["사용승인일"] = pd.to_numeric(
    df[cols[18]], errors="coerce"
)  # 18번: 사용승인일자(YYYYMMDD)
df["용적률"] = pd.to_numeric(df[cols[31]], errors="coerce")  # 31번: 용적률(%)
df["건폐율"] = pd.to_numeric(df[cols[32]], errors="coerce")  # 32번: 건폐율(%)

# 사용승인일(YYYYMMDD) → 건축연도(YYYY) 추출
# 예: 19870512 → 1987 ; 10000 미만이거나 NaN 이면 결측
df["건축연도"] = df["사용승인일"].apply(
    lambda x: int(str(int(x))[:4]) if pd.notna(x) and x >= 10000 else np.nan
)

# 노후연수 = 2025 - 건축연도 (0~120 범위 클립)
df["노후연수"] = (2025 - df["건축연도"]).clip(0, 120)

# ─── 2. 집계구 경계 로드 ──────────────────────────────────────────
print("2. 집계구 경계 로드...")
# shapefile 을 WGS84 로 변환
oa = gpd.read_file("data/bnd_oa_11_2025_2Q/bnd_oa_11_2025_2Q.shp").to_crs("EPSG:4326")
# 면적 계산용 EPSG:5179(UTM-K) 재투영
oa_m = oa.to_crs("EPSG:5179")
oa["area_ha"] = oa_m.geometry.area / 10000  # m² → ha (1ha = 10,000m²)

# 집계구 코드 앞 5자리 = 자치구 코드 매핑
gu_map = {
    "11010": "종로구",
    "11020": "중구",
    "11030": "용산구",
    "11040": "성동구",
    "11050": "광진구",
    "11060": "동대문구",
    "11070": "중랑구",
    "11080": "노원구",
    "11090": "강북구",
    "11100": "도봉구",
    "11110": "은평구",
    "11120": "서대문구",
    "11130": "마포구",
    "11140": "양천구",
    "11150": "강서구",
    "11160": "구로구",
    "11170": "금천구",
    "11180": "영등포구",
    "11190": "동작구",
    "11200": "관악구",
    "11210": "서초구",
    "11220": "강남구",
    "11230": "송파구",
    "11240": "강동구",
    "11250": "도봉구",
}
oa["gu_name"] = (
    oa["TOT_OA_CD"].str[:5].map(gu_map).fillna("알수없음")
)  # 코드→구명 매핑

# 후속 단계 빠른 조회용 dict
oa_gu = dict(zip(oa["TOT_OA_CD"], oa["gu_name"]))
oa_adm = dict(zip(oa["TOT_OA_CD"], oa["ADM_CD"]))

# ─── 3. 숙박시설 → 집계구 공간결합 ───────────────────────────────
print("3. 숙박시설 공간결합...")
# 좌표 기반 GeoDataFrame 생성
gdf = gpd.GeoDataFrame(
    df, geometry=gpd.points_from_xy(df["lng"], df["lat"]), crs="EPSG:4326"
)
# 각 숙박시설 포인트가 어느 집계구 폴리곤 안에 있는지 매핑 (within = 내부 포함)
joined = gpd.sjoin(gdf, oa[["TOT_OA_CD", "geometry"]], how="left", predicate="within")

# 집계구별 건축 지표 평균/극값 집계
grp = (
    joined.groupby("TOT_OA_CD")
    .agg(
        avg_floors=("층수", "mean"),  # 평균 층수
        avg_yongjuk=("용적률", "mean"),  # 평균 용적률(%)
        avg_geonpye=("건폐율", "mean"),  # 평균 건폐율(%)
        avg_age=("노후연수", "mean"),  # 평균 건축연령
        max_age=("노후연수", "max"),  # 가장 오래된 건물의 연령
        min_age=("노후연수", "min"),  # 가장 새 건물의 연령
    )
    .round(1)
)
print(f"   집계 완료: {len(grp)}개 집계구")

# ─── 4. 소방안전센터 위치 추출 (출동 좌표 중앙값으로 추산) ────────
print("4. 소방안전센터 위치 추출...")
fire = pd.read_csv(
    "data/화재출동/화재출동_2021_2024.csv", encoding="utf-8-sig", low_memory=False
)
fc = fire.columns.tolist()

# 33번(경도) / 34번(위도) 숫자형 변환 + 결측 제거
fire["lng"] = pd.to_numeric(fire[fc[33]], errors="coerce")
fire["lat"] = pd.to_numeric(fire[fc[34]], errors="coerce")
fire = fire.dropna(subset=["lng", "lat"])

# 서울 영역 클립 (위도 37.3~37.8, 경도 126.5~127.3)
fire = fire[
    (fire["lng"] > 126.5)
    & (fire["lng"] < 127.3)
    & (fire["lat"] > 37.3)
    & (fire["lat"] < 37.8)
]

# 관할소방서(본서) — 25번 열 이름으로 그룹화, 출동 위치의 중앙값 = 시설 위치 추정
hs_grp = (
    fire.groupby(fc[25])
    .agg(
        lat=("lat", "median"),
        lng=("lng", "median"),  # median: 극단값 영향 최소화
        count=("lat", "count"),  # 출동 건수
    )
    .reset_index()
    .rename(columns={fc[25]: "name"})
)
hs_grp["type"] = "소방서"

# 출동안전센터 — 26번 열 이름으로 그룹화
sc_grp = (
    fire.groupby(fc[26])
    .agg(lat=("lat", "median"), lng=("lng", "median"), count=("lat", "count"))
    .reset_index()
    .rename(columns={fc[26]: "name"})
)
sc_grp["type"] = "안전센터"

# 두 시설을 합치고 출동건수 10건 이상인 신뢰성 있는 위치만 채택
stations = pd.concat([hs_grp, sc_grp], ignore_index=True)
stations = stations[stations["count"] >= 10]
stations_list = stations[["name", "type", "lat", "lng", "count"]].to_dict("records")
print(
    f"   소방서: {len(hs_grp)}개, 안전센터: {len(sc_grp)}개 → 총 {len(stations_list)}개"
)

# JSON 으로 저장 (separators 로 공백 제거 → 파일 크기 절약)
with open("data/firestation_data.json", "w", encoding="utf-8") as f:
    json.dump(stations_list, f, ensure_ascii=False, separators=(",", ":"))
print("   → data/firestation_data.json 저장")

# ─── 5. 화재위험점수 계산 ─────────────────────────────────────────
print("5. 화재위험점수 계산...")
# 만점 기준 (해당 값 이상은 만점 처리)
MAX_AGE = 80  # 80년 이상 = 노후도 만점
MAX_FLOOR = 20  # 20층 이상 = 층수 만점
MAX_YONG = 600  # 용적률 600% 이상 = 만점
MAX_GPYE = 80  # 건폐율 80% 이상 = 만점


def fire_score(p):
    """집계구 속성 dict → 0~100 화재위험점수.

    - 노후도 30점 : 오래될수록 화재 위험 ↑
    - 건폐율 25점 : 건물 빽빽할수록 화재 확산 ↑
    - 용적률 25점 : 입체적으로 클수록 화재 규모 ↑
    - 층수   20점 : 높을수록 피난·소방 접근 어려움
    """
    age = min(p.get("avg_age") or 0, MAX_AGE) / MAX_AGE * 30
    gpye = min(p.get("avg_geonpye") or 0, MAX_GPYE) / MAX_GPYE * 25
    yong = min(p.get("avg_yongjuk") or 0, MAX_YONG) / MAX_YONG * 25
    floor = min(p.get("avg_floors") or 0, MAX_FLOOR) / MAX_FLOOR * 20
    return round(age + gpye + yong + floor, 1)


# ─── 6. oa_density.json 업데이트 ─────────────────────────────────
print("6. oa_density.json 업데이트...")
with open("data/oa_density.json", encoding="utf-8") as f:
    geo = json.load(f)

# 모든 feature 에 대해 구명/동코드/집계구번호 + 건축 지표 + 화재위험점수 부여
for feat in geo["features"]:
    oid = feat["properties"]["id"]  # 집계구 고유 ID
    p = feat["properties"]

    # 구명/동코드/집계구번호 파싱
    p["gu_name"] = oa_gu.get(oid, "알수없음")
    p["dong_code"] = oa_adm.get(oid, "00000000")[5:]  # ADM_CD 뒤 8자리 = 동 코드
    p["oa_no"] = oid[8:]  # ID 앞 8자리 = 구/동 코드, 나머지 = 집계구 번호

    if oid in grp.index:
        # 숙박시설이 있는 집계구 → 집계 결과 부착
        r = grp.loc[oid]

        def sv(v):
            """NaN 안전 변환 (NaN → None, 그 외 float 반올림)."""
            return round(float(v), 1) if pd.notna(v) else None

        p["avg_floors"] = sv(r["avg_floors"])
        p["avg_yongjuk"] = sv(r["avg_yongjuk"])
        p["avg_geonpye"] = sv(r["avg_geonpye"])
        p["avg_age"] = sv(r["avg_age"])
        p["max_age"] = int(r["max_age"]) if pd.notna(r["max_age"]) else None
        p["min_age"] = int(r["min_age"]) if pd.notna(r["min_age"]) else None
    else:
        # 숙박시설 없는 집계구 → 모든 건축 지표 None
        for k in [
            "avg_floors",
            "avg_yongjuk",
            "avg_geonpye",
            "avg_age",
            "max_age",
            "min_age",
        ]:
            p[k] = None

    # 화재위험점수 산출 후 저장
    p["fire_score"] = fire_score(p)

# 결과 JSON 덮어쓰기
with open("data/oa_density.json", "w", encoding="utf-8") as f:
    json.dump(geo, f, ensure_ascii=False, separators=(",", ":"))

# 숙박시설이 있는 집계구의 점수 통계 출력
scores = [
    ft["properties"]["fire_score"]
    for ft in geo["features"]
    if ft["properties"]["count"] > 0
]
arr = np.array(scores)
print(f"   저장 완료 | 점수범위: {arr.min():.1f}~{arr.max():.1f} 평균:{arr.mean():.1f}")
print("Done.")
