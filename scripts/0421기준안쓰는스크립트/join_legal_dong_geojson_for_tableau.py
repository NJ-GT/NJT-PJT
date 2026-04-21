"""
[파일 설명]
법정동 경계 GeoJSON과 건물 건축연한 구간 통계를 조인하여 Tableau 시각화용 GeoJSON을 생성하는 스크립트.

join_legal_dong_geojson_approval_0415.py와 유사하지만 입력 CSV와 출력 파일명이 다르다.
Tableau에서 지도 시각화를 위해 건물 노후화 구간 통계를 법정동 경계에 붙인다.

입력: data/seoul_neighborhoods_geo_simple.json          (법정동 경계)
      data/자치구_법정동별_사용승인연한구간_0415.csv        (건물 연한 구간 통계)
출력: data/seoul_legal_dong_age_buckets_joined_0415.geojson          (전체)
      data/seoul_legal_dong_age_buckets_joined_0415_only_data.geojson (데이터 있는 것만)
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]  # 프로젝트 루트
DATA_DIR = ROOT / "data"

GEOJSON_PATH = DATA_DIR / "seoul_neighborhoods_geo_simple.json"                           # 법정동 경계
STATS_PATH = DATA_DIR / "자치구_법정동별_사용승인연한구간_0415.csv"                         # 건물 연한 통계
OUT_ALL_PATH = DATA_DIR / "seoul_legal_dong_age_buckets_joined_0415.geojson"              # 전체 출력
OUT_FILTERED_PATH = DATA_DIR / "seoul_legal_dong_age_buckets_joined_0415_only_data.geojson"  # 데이터 있는 것만


def load_stats(path: Path) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            code = str(row.get("법정동코드", "")).strip()
            if not code:
                continue
            geojson_code = code[:8] if len(code) >= 8 else code
            result[geojson_code] = {
                "기준일": str(row.get("기준일", "")).strip(),
                "구": str(row.get("구", "")).strip(),
                "법정동코드": code,
                "법정동명": str(row.get("법정동명", "")).strip(),
                "전체건물수": int(float(row.get("전체건물수", "0") or 0)),
                "사용승인일유효건수": int(float(row.get("사용승인일유효건수", "0") or 0)),
                "10년이상건물수": int(float(row.get("10년이상건물수", "0") or 0)),
                "30년이상건물수": int(float(row.get("30년이상건물수", "0") or 0)),
                "50년이상건물수": int(float(row.get("50년이상건물수", "0") or 0)),
            }
    return result


def main() -> None:
    stats = load_stats(STATS_PATH)
    geo = json.loads(GEOJSON_PATH.read_text(encoding="utf-8"))

    all_features = []
    filtered_features = []

    for feature in geo.get("features", []):
        props = dict(feature.get("properties", {}))
        code = str(props.get("EMD_CD", "")).strip()
        joined = stats.get(code)

        new_props = {
            **props,
            "join_code": code,
            "has_data": 1 if joined else 0,
            "기준일": joined["기준일"] if joined else None,
            "구": joined["구"] if joined else None,
            "법정동코드": joined["법정동코드"] if joined else code,
            "법정동명": joined["법정동명"] if joined else props.get("EMD_KOR_NM"),
            "전체건물수": joined["전체건물수"] if joined else 0,
            "사용승인일유효건수": joined["사용승인일유효건수"] if joined else 0,
            "10년이상건물수": joined["10년이상건물수"] if joined else 0,
            "30년이상건물수": joined["30년이상건물수"] if joined else 0,
            "50년이상건물수": joined["50년이상건물수"] if joined else 0,
        }

        new_feature = {
            "type": feature["type"],
            "geometry": feature["geometry"],
            "properties": new_props,
        }
        all_features.append(new_feature)
        if joined:
            filtered_features.append(new_feature)

    out_all = {"type": "FeatureCollection", "features": all_features}
    out_filtered = {"type": "FeatureCollection", "features": filtered_features}

    OUT_ALL_PATH.write_text(json.dumps(out_all, ensure_ascii=False), encoding="utf-8")
    OUT_FILTERED_PATH.write_text(json.dumps(out_filtered, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote: {OUT_ALL_PATH}")
    print(f"Wrote: {OUT_FILTERED_PATH}")
    print(f"All features: {len(all_features)}")
    print(f"Joined features: {len(filtered_features)}")


if __name__ == "__main__":
    main()
