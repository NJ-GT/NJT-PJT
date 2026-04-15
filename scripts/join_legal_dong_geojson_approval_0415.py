from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

GEOJSON_PATH = DATA_DIR / "seoul_neighborhoods_geo_simple.json"
STATS_PATH = DATA_DIR / "법정동별_사용승인구간_0415.csv"
OUT_ALL_PATH = DATA_DIR / "법정동별_사용승인구간_공간정보0415.geojson"
OUT_FILTERED_PATH = DATA_DIR / "[오피셜]법정동승인일자_공간정보0415.geojson"


INT_FIELDS = [
    "전체건물수",
    "사용승인일유효건수",
    "10년미만건물수",
    "10년이상건물수",
    "30년이상건물수",
    "50년이상건물수",
]


def load_stats(path: Path) -> dict[str, dict[str, object]]:
    stats: dict[str, dict[str, object]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            full_code = str(row.get("법정동코드", "")).strip()
            if not full_code:
                continue
            join_code = full_code[:8] if len(full_code) >= 8 else full_code
            record: dict[str, object] = {
                "기준일": str(row.get("기준일", "")).strip(),
                "구": str(row.get("구", "")).strip(),
                "법정동코드": full_code,
                "법정동명": str(row.get("법정동명", "")).strip(),
            }
            for field in INT_FIELDS:
                raw = str(row.get(field, "")).strip().replace(",", "")
                record[field] = int(float(raw)) if raw else 0
            stats[join_code] = record
    return stats


def main() -> None:
    geo = json.loads(GEOJSON_PATH.read_text(encoding="utf-8"))
    stats = load_stats(STATS_PATH)

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
            "법정동코드": joined["법정동코드"] if joined else None,
            "법정동명": joined["법정동명"] if joined else props.get("EMD_KOR_NM"),
        }
        for field in INT_FIELDS:
            new_props[field] = joined[field] if joined else 0

        new_feature = {
            "type": feature["type"],
            "geometry": feature["geometry"],
            "properties": new_props,
        }
        all_features.append(new_feature)
        if joined:
            filtered_features.append(new_feature)

    OUT_ALL_PATH.write_text(
        json.dumps({"type": "FeatureCollection", "features": all_features}, ensure_ascii=False),
        encoding="utf-8",
    )
    OUT_FILTERED_PATH.write_text(
        json.dumps({"type": "FeatureCollection", "features": filtered_features}, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Wrote: {OUT_ALL_PATH}")
    print(f"Wrote: {OUT_FILTERED_PATH}")
    print(f"All features: {len(all_features)}")
    print(f"Joined features: {len(filtered_features)}")


if __name__ == "__main__":
    main()
