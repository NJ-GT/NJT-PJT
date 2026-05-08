# -*- coding: utf-8 -*-
"""
3D 지도용 경량 oa_3d.json 생성 스크립트.

목적:
    집계구 shapefile + 분석 결과(oa_density.json)를 한 파일로 합치되,
    웹에서 빠르게 로드되도록 좌표를 단순화·반올림하고 속성 키를 짧게 줄인다.

처리:
    1) shapefile -> WGS84 변환 + simplify(약 30m 오차) 로 폴리곤 단순화
    2) oa_density.json 의 속성을 id 기준으로 dict화 (빠른 조회)
    3) slim() 으로 필요 키만 + 짧은 키 이름으로 변환
    4) 좌표 소수점 4자리 반올림 (약 11m 정밀도)
    5) GeoJSON FeatureCollection으로 oa_3d.json 저장 (콤마/콜론 공백 제거)

산출:
    NJT-PJT/data/oa_3d.json
"""

import sys
import json
import geopandas as gpd
import os

# Windows 콘솔 한글 깨짐 방지
sys.stdout.reconfigure(encoding="utf-8")

# ─── 1. 집계구 경계 shapefile 로드 및 단순화 ─────────────────────────────
# WGS84(웹 지도 표준)로 좌표계 통일
oa = gpd.read_file("data/bnd_oa_11_2025_2Q/bnd_oa_11_2025_2Q.shp").to_crs("EPSG:4326")
# simplify(0.0003): 폴리곤 꼭짓점 수를 줄여 파일 크기 절감 (약 30m 오차 허용)
# preserve_topology=True : 폴리곤이 서로 겹치거나 갈라지지 않도록 보존
oa["geometry"] = oa["geometry"].simplify(0.0003, preserve_topology=True)

# ─── 2. 집계구별 분석 데이터 로드 ─────────────────────────────────────────
with open("data/oa_density.json", encoding="utf-8") as f:
    raw = json.load(f)

# 14자리 집계구 ID -> 속성 dict 매핑 (빠른 조회)
prop_map = {f["properties"]["id"]: f["properties"] for f in raw["features"]}


# ─── 3. 속성 경량화 함수 ──────────────────────────────────────────────────
def slim(p):
    """3D 지도에서 필요한 속성만 짧은 키로 변환 — 파일 크기 ↓."""
    return {
        # 식별 정보
        "id": p.get("id", ""),
        "gu": p.get("gu_name", ""),
        "no": p.get("oa_no", ""),
        # 시각화에 직접 쓰는 통계
        "cnt": p.get("count", 0),
        "fl": round(p.get("avg_floors") or 0, 1),
        "age": round(p.get("avg_age") or 0, 1),
        "rat": round(p.get("ratio", 0), 1),
        "fire": round(p.get("fire_score", 0), 1),
        "area": round(p.get("area_ha", 0), 2),
    }


# ─── 4. 좌표 반올림 함수 ─────────────────────────────────────────────────
def round_geom(geom):
    """GeoJSON Polygon/MultiPolygon 좌표를 소수점 4자리(약 11m)로 반올림."""
    t = geom["type"]
    if t == "Polygon":
        # 외곽 + 구멍 모두 처리
        coords = [
            [[round(x, 4), round(y, 4)] for x, y in ring]
            for ring in geom["coordinates"]
        ]
        return {"type": "Polygon", "coordinates": coords}
    elif t == "MultiPolygon":
        # 여러 폴리곤 (섬 등)
        mp = []
        for poly in geom["coordinates"]:
            rings = [[[round(x, 4), round(y, 4)] for x, y in ring] for ring in poly]
            mp.append(rings)
        return {"type": "MultiPolygon", "coordinates": mp}
    # Point 등 기타 타입은 그대로
    return geom


# ─── 5. GeoJSON Feature 리스트 생성 ─────────────────────────────────────
features = []
for _, row in oa.iterrows():
    # 집계구 코드 (14자리)
    oid = row["TOT_OA_CD"]
    # 분석 데이터 — 매칭 실패 시 카운트 0으로 더미
    props = prop_map.get(oid, {"id": oid, "count": 0})
    # shapely geometry -> GeoJSON dict + 좌표 반올림
    geom = round_geom(row["geometry"].__geo_interface__)
    features.append({"type": "Feature", "geometry": geom, "properties": slim(props)})

# ─── 6. 결과 저장 ────────────────────────────────────────────────────────
out = {"type": "FeatureCollection", "features": features}
# separators=(',',':') — JSON 직렬화에서 공백 제거로 파일 크기 절감
with open("data/oa_3d.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, separators=(",", ":"))

# 결과 파일 크기 + 피처 수 출력 (CLI 확인용)
sz = os.path.getsize("data/oa_3d.json")
print(f"저장 완료: {sz // 1024} KB  집계구 {len(features)}개")
