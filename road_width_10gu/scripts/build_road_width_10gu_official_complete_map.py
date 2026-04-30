from __future__ import annotations

import json
import math
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, MultiLineString


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
INPUT_CSV = next(DATA_DIR.glob("seoul_road_width_viRoutDt_10*.csv"))
OFFICIAL_SHP = DATA_DIR / "official_road_shape_202603_seoul" / "11000" / "TL_SPRD_MANAGE.shp"
OUTPUT_HTML = DATA_DIR / "seoul_road_width_10gu_official_complete_map.html"
OUTPUT_GEOJSON = DATA_DIR / "seoul_road_width_10gu_official_complete_lines.geojson"
UNMATCHED_API_CSV = DATA_DIR / "seoul_road_width_10gu_official_complete_api_unmatched.csv"
OFFICIAL_ONLY_CSV = DATA_DIR / "seoul_road_width_10gu_official_only_roads.csv"

SRC_CRS = "EPSG:5179"

SIG_TO_GU = {
    "11110": "종로구",
    "11140": "중구",
    "11170": "용산구",
    "11200": "성동구",
    "11440": "마포구",
    "11500": "강서구",
    "11560": "영등포구",
    "11650": "서초구",
    "11680": "강남구",
    "11710": "송파구",
}

GU_ORDER = ["강남구", "강서구", "마포구", "서초구", "성동구", "송파구", "영등포구", "용산구", "종로구", "중구"]
WIDTH_ORDER = [
    "6m미만",
    "폭6-8m",
    "폭8-10m",
    "폭10-12m",
    "폭12-15m",
    "폭15-20m",
    "폭20-25m",
    "폭25-30m",
    "폭30-35m",
    "폭35-40m",
    "폭40-50m",
    "폭50-70m",
    "불가",
]
WIDTH_COLORS = {
    "6m미만": "#2f855a",
    "폭6-8m": "#63a15f",
    "폭8-10m": "#d6a419",
    "폭10-12m": "#e27b36",
    "폭12-15m": "#e05d3d",
    "폭15-20m": "#d64545",
    "폭20-25m": "#b83266",
    "폭25-30m": "#7f3c8d",
    "폭30-35m": "#2f6fb5",
    "폭35-40m": "#1e4e8c",
    "폭40-50m": "#2d3748",
    "폭50-70m": "#111827",
    "불가": "#7a7f87",
}
WIDTH_WEIGHTS = {
    "6m미만": 2,
    "폭6-8m": 3,
    "폭8-10m": 4,
    "폭10-12m": 5,
    "폭12-15m": 5,
    "폭15-20m": 6,
    "폭20-25m": 7,
    "폭25-30m": 8,
    "폭30-35m": 9,
    "폭35-40m": 10,
    "폭40-50m": 11,
    "폭50-70m": 12,
    "불가": 3,
}


def normalize_name(value: object) -> str:
    return "".join(str(value or "").strip().split())


def official_width_bucket(value: float | None) -> str:
    if value is None or math.isnan(value):
        return "불가"
    if value < 6:
        return "6m미만"
    if value < 8:
        return "폭6-8m"
    if value < 10:
        return "폭8-10m"
    if value < 12:
        return "폭10-12m"
    if value < 15:
        return "폭12-15m"
    if value < 20:
        return "폭15-20m"
    if value < 25:
        return "폭20-25m"
    if value < 30:
        return "폭25-30m"
    if value < 35:
        return "폭30-35m"
    if value < 40:
        return "폭35-40m"
    if value < 50:
        return "폭40-50m"
    if value < 70:
        return "폭50-70m"
    return "폭50-70m"


def distance_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    mean_lat = math.radians((lat1 + lat2) / 2)
    dx = (lng2 - lng1) * 111_320 * math.cos(mean_lat)
    dy = (lat2 - lat1) * 110_574
    return math.hypot(dx, dy)


def perpendicular_distance_m(point: list[float], start: list[float], end: list[float]) -> float:
    lat0, lng0 = point[1], point[0]
    lat1, lng1 = start[1], start[0]
    lat2, lng2 = end[1], end[0]
    mean_lat = math.radians((lat0 + lat1 + lat2) / 3)

    def project(lat: float, lng: float) -> tuple[float, float]:
        return lng * 111_320 * math.cos(mean_lat), lat * 110_574

    px, py = project(lat0, lng0)
    x1, y1 = project(lat1, lng1)
    x2, y2 = project(lat2, lng2)
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return math.hypot(px - x1, py - y1)
    return abs(dy * px - dx * py + x2 * y1 - y2 * x1) / math.hypot(dx, dy)


def simplify_line(coords: list[list[float]], tolerance_m: float = 1.2) -> list[list[float]]:
    if len(coords) <= 2:
        return coords
    start = coords[0]
    end = coords[-1]
    max_dist = -1.0
    max_idx = 0
    for idx in range(1, len(coords) - 1):
        dist = perpendicular_distance_m(coords[idx], start, end)
        if dist > max_dist:
            max_dist = dist
            max_idx = idx
    if max_dist > tolerance_m:
        left = simplify_line(coords[: max_idx + 1], tolerance_m)
        right = simplify_line(coords[max_idx:], tolerance_m)
        return left[:-1] + right
    return [start, end]


def geometry_to_lines(geom: LineString | MultiLineString) -> list[list[list[float]]]:
    if geom.is_empty:
        return []
    if geom.geom_type == "LineString":
        return [[[float(x), float(y)] for x, y in geom.coords]]
    if geom.geom_type == "MultiLineString":
        return [[[float(x), float(y)] for x, y in part.coords] for part in geom.geoms]
    return []


def choose_endpoints(lines: list[list[list[float]]]) -> tuple[list[float], list[float]]:
    endpoints = []
    for line in lines:
        if len(line) >= 2:
            endpoints.extend([line[0], line[-1]])
    if len(endpoints) < 2:
        flat = [coord for line in lines for coord in line]
        return flat[0], flat[-1]
    best_pair = (endpoints[0], endpoints[-1])
    best_dist = -1.0
    for i, a in enumerate(endpoints):
        for b in endpoints[i + 1 :]:
            dist = distance_m(a[1], a[0], b[1], b[0])
            if dist > best_dist:
                best_dist = dist
                best_pair = (a, b)
    return best_pair


def line_length_m(lines: list[list[list[float]]]) -> float:
    total = 0.0
    for line in lines:
        for a, b in zip(line, line[1:]):
            total += distance_m(a[1], a[0], b[1], b[0])
    return total


def load_input() -> pd.DataFrame:
    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig", dtype=str).fillna("")
    seq_col, gu_col, road_col, kind_col, function_col, scale_col, width_col = df.columns[:7]
    df = df.rename(
        columns={
            seq_col: "순번",
            gu_col: "구",
            road_col: "도로명",
            kind_col: "도로구분",
            function_col: "도로기능",
            scale_col: "도로규모",
            width_col: "API도로폭",
        }
    )
    df["_key"] = df["구"].map(normalize_name) + "|" + df["도로명"].map(normalize_name)
    return df.drop_duplicates("_key", keep="first")


def load_official() -> gpd.GeoDataFrame:
    gdf = gpd.read_file(OFFICIAL_SHP, engine="pyogrio", encoding="cp949")
    gdf["구"] = gdf["SIG_CD"].astype(str).map(SIG_TO_GU).fillna("")
    gdf = gdf[gdf["구"] != ""].copy()
    gdf["_key"] = gdf["구"].map(normalize_name) + "|" + gdf["RN"].map(normalize_name)
    return gdf.set_crs(SRC_CRS, allow_override=True).to_crs("EPSG:4326")


def build_features(input_df: pd.DataFrame, official: gpd.GeoDataFrame) -> tuple[list[dict], pd.DataFrame, pd.DataFrame]:
    meta_by_key = {row["_key"]: row for _, row in input_df.iterrows()}
    official_keys = set(official["_key"])
    api_only = input_df[~input_df["_key"].isin(official_keys)].copy()
    official_only_keys = sorted(official_keys - set(meta_by_key))

    features: list[dict] = []
    official_only_rows: list[dict] = []
    for key, group in official.groupby("_key", sort=False):
        official_widths = pd.to_numeric(group["ROAD_BT"], errors="coerce").dropna()
        official_lengths = pd.to_numeric(group["ROAD_LT"], errors="coerce").dropna()
        official_avg_width = float(official_widths.mean()) if not official_widths.empty else None
        fallback_width = official_width_bucket(official_avg_width)

        if key in meta_by_key:
            meta = meta_by_key[key]
            api_width = str(meta["API도로폭"]).strip() or "불가"
            if api_width and api_width != "불가":
                display_width = api_width
                width_source = "API"
            else:
                display_width = fallback_width
                width_source = "공식폭보완"
            seq = int(float(meta["순번"]))
            road_kind = str(meta["도로구분"])
            road_function = str(meta["도로기능"])
            road_scale = str(meta["도로규모"])
        else:
            first = group.iloc[0]
            api_width = ""
            display_width = fallback_width
            width_source = "공식만"
            seq = int(first["RDS_MAN_NO"]) if pd.notna(first["RDS_MAN_NO"]) else 0
            road_kind = "공식도로구간"
            road_function = ""
            road_scale = ""
            official_only_rows.append(
                {
                    "구": first["구"],
                    "도로명": first["RN"],
                    "표시도로폭": display_width,
                    "공식도로폭평균m": official_avg_width,
                    "공식구간수": len(group),
                }
            )

        raw_lines: list[list[list[float]]] = []
        for geom in group.geometry:
            raw_lines.extend(geometry_to_lines(geom))
        lines = [simplify_line(line) for line in raw_lines if len(line) >= 2]
        if not lines:
            continue
        start, end = choose_endpoints(lines)
        official_min_width = float(official_widths.min()) if not official_widths.empty else None
        official_max_width = float(official_widths.max()) if not official_widths.empty else None
        props = {
            "순번": seq,
            "구": str(group.iloc[0]["구"]),
            "도로명": str(group.iloc[0]["RN"]),
            "도로구분": road_kind,
            "도로기능": road_function,
            "도로규모": road_scale,
            "API도로폭": api_width,
            "표시도로폭": display_width,
            "폭출처": width_source,
            "시작위도": start[1],
            "시작경도": start[0],
            "끝위도": end[1],
            "끝경도": end[0],
            "공식선형길이m": round(line_length_m(lines), 1),
            "공식도로폭평균m": round(official_avg_width, 2) if official_avg_width is not None else None,
            "공식도로폭최소m": round(official_min_width, 2) if official_min_width is not None else None,
            "공식도로폭최대m": round(official_max_width, 2) if official_max_width is not None else None,
            "공식도로길이합m": round(float(official_lengths.sum()), 1) if not official_lengths.empty else None,
            "RN_CD": "|".join(sorted({str(v) for v in group["RN_CD"].dropna().unique()})),
            "공식구간수": int(len(group)),
            "색상": WIDTH_COLORS.get(display_width, WIDTH_COLORS["불가"]),
            "두께": WIDTH_WEIGHTS.get(display_width, WIDTH_WEIGHTS["불가"]),
        }
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "MultiLineString", "coordinates": lines},
                "properties": props,
            }
        )

    official_only = pd.DataFrame(official_only_rows)
    if not official_only_keys and official_only.empty:
        official_only = pd.DataFrame(columns=["구", "도로명", "표시도로폭", "공식도로폭평균m", "공식구간수"])
    return features, api_only, official_only


def count_by(features: list[dict], prop: str, order: list[str]) -> list[dict[str, object]]:
    counts = {name: 0 for name in order}
    for feature in features:
        value = str(feature["properties"].get(prop, ""))
        counts[value] = counts.get(value, 0) + 1
    return [{"name": name, "count": counts[name]} for name in order if counts.get(name, 0)]


def source_counts(features: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {"API": 0, "공식폭보완": 0, "공식만": 0}
    for feature in features:
        src = str(feature["properties"].get("폭출처", ""))
        counts[src] = counts.get(src, 0) + 1
    return counts


def leaflet_rows(features: list[dict]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for feature in features:
        props = feature["properties"]
        rows.append(
            {
                "id": props["순번"],
                "gu": props["구"],
                "road": props["도로명"],
                "kind": props["도로구분"],
                "function": props["도로기능"],
                "scale": props["도로규모"],
                "apiWidth": props["API도로폭"],
                "width": props["표시도로폭"],
                "widthSource": props["폭출처"],
                "officialAvgWidth": props["공식도로폭평균m"],
                "officialMinWidth": props["공식도로폭최소m"],
                "officialMaxWidth": props["공식도로폭최대m"],
                "lengthM": props["공식선형길이m"],
                "sectionCount": props["공식구간수"],
                "start": [props["시작위도"], props["시작경도"]],
                "end": [props["끝위도"], props["끝경도"]],
                "color": props["색상"],
                "weight": props["두께"],
                "lines": [
                    [[coord[1], coord[0]] for coord in line]
                    for line in feature["geometry"]["coordinates"]
                    if len(line) >= 2
                ],
            }
        )
    return rows


def build_html(features: list[dict], api_only_count: int, official_only_count: int) -> str:
    rows = leaflet_rows(features)
    center_lat = sum((row["start"][0] + row["end"][0]) / 2 for row in rows) / len(rows)
    center_lng = sum((row["start"][1] + row["end"][1]) / 2 for row in rows) / len(rows)
    counts = source_counts(features)
    data_json = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
    gu_counts_json = json.dumps(count_by(features, "구", GU_ORDER), ensure_ascii=False, separators=(",", ":"))
    width_counts_json = json.dumps(count_by(features, "표시도로폭", WIDTH_ORDER), ensure_ascii=False, separators=(",", ":"))
    gu_json = json.dumps(GU_ORDER, ensure_ascii=False)
    colors_json = json.dumps(WIDTH_COLORS, ensure_ascii=False)

    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>서울 10개구 공식 도로구간 완성 지도</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <style>
    :root {{
      --ink: #172033;
      --muted: #687389;
      --line: #d8dde8;
      --panel: rgba(255,255,255,.96);
      --accent: #2563eb;
    }}
    * {{ box-sizing: border-box; }}
    html, body {{ height: 100%; margin: 0; }}
    body {{ font-family: "Malgun Gothic", "Apple SD Gothic Neo", Arial, sans-serif; color: var(--ink); background: #eef2f7; }}
    #map {{ position: fixed; inset: 0; }}
    .panel {{
      position: fixed;
      top: 16px;
      left: 16px;
      z-index: 1000;
      width: 400px;
      max-height: calc(100vh - 32px);
      overflow: auto;
      background: var(--panel);
      border: 1px solid rgba(122, 132, 154, .32);
      border-radius: 8px;
      box-shadow: 0 18px 40px rgba(28,39,64,.18);
      backdrop-filter: blur(8px);
    }}
    .panel-header {{ padding: 16px; border-bottom: 1px solid var(--line); }}
    h1 {{ margin: 0 0 8px; font-size: 19px; line-height: 1.28; font-weight: 800; letter-spacing: 0; }}
    .note {{ margin: 7px 0 0; color: var(--muted); font-size: 12px; line-height: 1.45; }}
    .meta {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-top: 12px; }}
    .metric {{ min-width: 0; padding: 9px 8px; border: 1px solid var(--line); border-radius: 6px; background: #f8fafc; }}
    .metric strong {{ display: block; font-size: 17px; line-height: 1.1; }}
    .metric span {{ display: block; margin-top: 4px; font-size: 11px; color: var(--muted); white-space: nowrap; }}
    .panel-body {{ padding: 14px 16px 16px; }}
    .field {{ margin-bottom: 14px; }}
    .field label {{ display: block; margin-bottom: 7px; font-size: 12px; font-weight: 700; color: #35415a; }}
    .search-row {{ display: grid; grid-template-columns: 1fr 72px; gap: 8px; }}
    input[type="search"] {{ width: 100%; height: 36px; padding: 0 10px; border: 1px solid #bcc6d8; border-radius: 6px; font-size: 13px; outline: none; background: white; }}
    input[type="search"]:focus {{ border-color: var(--accent); box-shadow: 0 0 0 3px rgba(37,99,235,.14); }}
    button {{ height: 36px; border: 1px solid #bcc6d8; border-radius: 6px; background: white; color: #253148; font-weight: 700; font-size: 12px; cursor: pointer; }}
    button:hover {{ background: #f1f5fb; }}
    .chips {{ display: flex; flex-wrap: wrap; gap: 7px; }}
    .chip {{ display: inline-flex; align-items: center; gap: 6px; min-height: 30px; max-width: 100%; padding: 5px 8px; border: 1px solid #c8d0df; border-radius: 6px; background: white; font-size: 12px; line-height: 1.2; user-select: none; cursor: pointer; }}
    .chip input {{ margin: 0; }}
    .swatch {{ width: 10px; height: 10px; border-radius: 50%; flex: 0 0 auto; border: 1px solid rgba(0,0,0,.16); }}
    .rows {{ display: grid; gap: 6px; font-size: 12px; }}
    .bar-row {{ display: grid; grid-template-columns: 64px 1fr 42px; align-items: center; gap: 8px; min-height: 22px; }}
    .bar-label, .bar-count {{ white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
    .bar-track {{ height: 8px; border-radius: 999px; background: #e8edf5; overflow: hidden; }}
    .bar-fill {{ height: 100%; min-width: 2px; border-radius: 999px; background: #4169a8; }}
    .legend-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 6px 8px; font-size: 12px; }}
    .legend-item {{ display: flex; align-items: center; min-width: 0; gap: 6px; }}
    .legend-item span:last-child {{ overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
    .leaflet-popup-content {{ font-family: "Malgun Gothic", "Apple SD Gothic Neo", Arial, sans-serif; font-size: 12px; margin: 12px; min-width: 264px; }}
    .popup-title {{ font-size: 14px; font-weight: 800; margin-bottom: 6px; }}
    .popup-table {{ width: 100%; border-collapse: collapse; }}
    .popup-table th {{ width: 94px; text-align: left; color: #667085; font-weight: 700; padding: 3px 8px 3px 0; vertical-align: top; }}
    .popup-table td {{ padding: 3px 0; vertical-align: top; }}
    @media (max-width: 760px) {{
      .panel {{ top: auto; left: 8px; right: 8px; bottom: 8px; width: auto; max-height: 52vh; }}
      .panel-header {{ padding: 13px; }}
      .panel-body {{ padding: 12px 13px 13px; }}
      h1 {{ font-size: 17px; }}
    }}
  </style>
</head>
<body>
  <div id="map" aria-label="서울 10개구 공식 도로구간 완성 지도"></div>
  <aside class="panel">
    <div class="panel-header">
      <h1>서울 10개구 공식 도로구간</h1>
      <p class="note">도로폭 API를 우선 적용하고, 비는 공식 도로는 행안부 ROAD_BT 폭으로 보완했습니다. 마우스를 올리면 강조되고 클릭하면 시작과 끝이 표시됩니다.</p>
      <div class="meta">
        <div class="metric"><strong id="visibleCount">0</strong><span>표시 도로</span></div>
        <div class="metric"><strong>{counts.get("공식만", 0):,}</strong><span>공식 보완</span></div>
        <div class="metric"><strong>{api_only_count:,}</strong><span>API만 있음</span></div>
      </div>
      <p class="note">API 폭: {counts.get("API", 0):,}개 · API 불가→공식폭: {counts.get("공식폭보완", 0):,}개 · 공식만: {official_only_count:,}개</p>
    </div>
    <div class="panel-body">
      <div class="field">
        <label for="search">검색</label>
        <div class="search-row">
          <input id="search" type="search" placeholder="도로명">
          <button id="reset">초기화</button>
        </div>
      </div>
      <div class="field">
        <label>구</label>
        <div id="guFilters" class="chips"></div>
      </div>
      <div class="field">
        <label>도로폭</label>
        <div id="widthFilters" class="chips"></div>
      </div>
      <div class="field">
        <label>구별 표시 건수</label>
        <div id="guSummary" class="rows"></div>
      </div>
      <div class="field">
        <label>도로폭 색상</label>
        <div id="legend" class="legend-grid"></div>
      </div>
    </div>
  </aside>
  <script>
    const DATA = {data_json};
    const GU_ORDER = {gu_json};
    const WIDTH_COLORS = {colors_json};
    const GU_COUNTS = {gu_counts_json};
    const WIDTH_COUNTS = {width_counts_json};
    const MAP_CENTER = [{center_lat:.8f}, {center_lng:.8f}];
    const canvasRenderer = L.canvas({{ padding: 0.5 }});
    const map = L.map("map", {{ preferCanvas: true, zoomControl: false }}).setView(MAP_CENTER, 11);
    L.control.zoom({{ position: "bottomright" }}).addTo(map);
    L.tileLayer("https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png", {{
      maxZoom: 19,
      attribution: '&copy; OpenStreetMap contributors &copy; CARTO'
    }}).addTo(map);

    const layer = L.layerGroup().addTo(map);
    const hoverLayer = L.layerGroup().addTo(map);
    const highlightLayer = L.layerGroup().addTo(map);
    let firstRender = true;

    function escapeHtml(value) {{
      return String(value ?? "").replace(/[&<>"']/g, (char) => ({{
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
      }}[char]));
    }}
    function createCheckbox(containerId, name, value, count, color) {{
      const label = document.createElement("label");
      label.className = "chip";
      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.name = name;
      checkbox.value = value;
      checkbox.checked = true;
      checkbox.addEventListener("change", render);
      label.appendChild(checkbox);
      if (color) {{
        const swatch = document.createElement("span");
        swatch.className = "swatch";
        swatch.style.background = color;
        label.appendChild(swatch);
      }}
      const text = document.createElement("span");
      text.textContent = `${{value}} ${{count}}`;
      label.appendChild(text);
      document.getElementById(containerId).appendChild(label);
    }}
    GU_COUNTS.forEach((item) => createCheckbox("guFilters", "gu", item.name, item.count));
    WIDTH_COUNTS.forEach((item) => createCheckbox("widthFilters", "width", item.name, item.count, WIDTH_COLORS[item.name]));
    function getSelected(name) {{
      return new Set([...document.querySelectorAll(`input[name="${{name}}"]:checked`)].map((el) => el.value));
    }}
    function formatMaybe(value, suffix = "") {{
      if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
      return Number(value).toLocaleString("ko-KR", {{ maximumFractionDigits: 1 }}) + suffix;
    }}
    function popupHtml(row) {{
      const apiWidth = row.apiWidth || "-";
      return `
        <div class="popup-title">${{escapeHtml(row.road)}}</div>
        <table class="popup-table">
          <tr><th>구</th><td>${{escapeHtml(row.gu)}}</td></tr>
          <tr><th>표시 폭</th><td>${{escapeHtml(row.width)}} <small>(${{escapeHtml(row.widthSource)}})</small></td></tr>
          <tr><th>API 폭</th><td>${{escapeHtml(apiWidth)}}</td></tr>
          <tr><th>공식 폭</th><td>${{formatMaybe(row.officialAvgWidth, " m")}} 평균 / ${{formatMaybe(row.officialMinWidth, " m")}}~${{formatMaybe(row.officialMaxWidth, " m")}}</td></tr>
          <tr><th>선형길이</th><td>${{formatMaybe(row.lengthM, " m")}}</td></tr>
          <tr><th>공식구간</th><td>${{row.sectionCount.toLocaleString("ko-KR")}}개</td></tr>
        </table>`;
    }}
    function filteredData() {{
      const selectedGu = getSelected("gu");
      const selectedWidth = getSelected("width");
      const query = document.getElementById("search").value.trim().toLowerCase();
      return DATA.filter((row) => {{
        if (!selectedGu.has(row.gu)) return false;
        if (!selectedWidth.has(row.width)) return false;
        if (!query) return true;
        return `${{row.road}} ${{row.gu}} ${{row.width}} ${{row.widthSource}}`.toLowerCase().includes(query);
      }});
    }}
    function updateSummary(rows) {{
      const guCounts = new Map();
      const widthValues = new Set();
      rows.forEach((row) => {{
        guCounts.set(row.gu, (guCounts.get(row.gu) || 0) + 1);
        widthValues.add(row.width);
      }});
      document.getElementById("visibleCount").textContent = rows.length.toLocaleString("ko-KR");
      const maxCount = Math.max(1, ...GU_ORDER.map((gu) => guCounts.get(gu) || 0));
      document.getElementById("guSummary").innerHTML = GU_ORDER.map((gu) => {{
        const count = guCounts.get(gu) || 0;
        const pct = (count / maxCount) * 100;
        return `
          <div class="bar-row">
            <div class="bar-label">${{gu}}</div>
            <div class="bar-track"><div class="bar-fill" style="width:${{pct}}%"></div></div>
            <div class="bar-count">${{count.toLocaleString("ko-KR")}}</div>
          </div>`;
      }}).join("");
    }}
    function renderLegend() {{
      document.getElementById("legend").innerHTML = WIDTH_COUNTS.map((item) => `
        <div class="legend-item">
          <span class="swatch" style="background:${{WIDTH_COLORS[item.name]}}"></span>
          <span>${{item.name}} ${{item.count.toLocaleString("ko-KR")}}</span>
        </div>`).join("");
    }}
    function showEndpoints(row) {{
      highlightLayer.clearLayers();
      row.lines.forEach((line) => {{
        L.polyline(line, {{
          color: "#111827",
          weight: Math.max(row.weight + 4, 7),
          opacity: 0.9,
          lineCap: "round",
          lineJoin: "round"
        }}).addTo(highlightLayer);
        L.polyline(line, {{
          color: row.color,
          weight: Math.max(row.weight + 1, 4),
          opacity: 1,
          lineCap: "round",
          lineJoin: "round"
        }}).addTo(highlightLayer);
      }});
      L.circleMarker(row.start, {{ radius: 7, color: "#ffffff", weight: 2, fillColor: "#111827", fillOpacity: 0.96 }})
        .bindTooltip(`시작: ${{row.road}}`, {{ permanent: true, direction: "top", offset: [0, -8] }})
        .addTo(highlightLayer);
      L.circleMarker(row.end, {{ radius: 7, color: "#ffffff", weight: 2, fillColor: "#f43f5e", fillOpacity: 0.96 }})
        .bindTooltip(`끝: ${{row.road}}`, {{ permanent: true, direction: "top", offset: [0, -8] }})
        .addTo(highlightLayer);
    }}
    function showHover(row) {{
      hoverLayer.clearLayers();
      row.lines.forEach((line) => {{
        L.polyline(line, {{ color: "#ffffff", weight: Math.max(row.weight + 5, 8), opacity: 0.95, lineCap: "round", lineJoin: "round" }}).addTo(hoverLayer);
        L.polyline(line, {{ color: row.color, weight: Math.max(row.weight + 2, 5), opacity: 1, lineCap: "round", lineJoin: "round" }}).addTo(hoverLayer);
      }});
    }}
    function render() {{
      const rows = filteredData();
      layer.clearLayers();
      hoverLayer.clearLayers();
      highlightLayer.clearLayers();
      const bounds = [];
      rows.forEach((row) => {{
        row.lines.forEach((line) => {{
          line.forEach((latlng) => bounds.push(latlng));
          const polyline = L.polyline(line, {{
            renderer: canvasRenderer,
            color: row.color,
            weight: Math.max(row.weight, 3),
            opacity: row.widthSource === "공식만" ? 0.72 : 0.93,
            lineCap: "round",
            lineJoin: "round",
            dashArray: row.widthSource === "공식만" ? "6 4" : null
          }});
          polyline
            .bindTooltip(`[${{row.gu}}] ${{row.road}} | ${{row.width}} | ${{row.widthSource}}`, {{ sticky: true }})
            .bindPopup(popupHtml(row), {{ maxWidth: 360 }})
            .addTo(layer);
          polyline.on("mouseover", () => showHover(row));
          polyline.on("mouseout", () => hoverLayer.clearLayers());
          polyline.on("click", () => showEndpoints(row));
        }});
      }});
      updateSummary(rows);
      if (firstRender && bounds.length) {{
        map.fitBounds(bounds, {{ padding: [28, 28] }});
        firstRender = false;
      }}
    }}
    document.getElementById("search").addEventListener("input", render);
    document.getElementById("reset").addEventListener("click", () => {{
      document.getElementById("search").value = "";
      document.querySelectorAll('input[type="checkbox"]').forEach((el) => el.checked = true);
      render();
    }});
    renderLegend();
    render();
    const initialRoad = DATA.find((row) => row.road === "올림픽대로") || DATA[0];
    if (initialRoad) showEndpoints(initialRoad);
  </script>
</body>
</html>
"""


def main() -> int:
    input_df = load_input()
    official = load_official()
    features, api_only, official_only = build_features(input_df, official)
    OUTPUT_GEOJSON.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}, ensure_ascii=False),
        encoding="utf-8",
    )
    api_only.drop(columns=[col for col in ["_key"] if col in api_only.columns]).to_csv(
        UNMATCHED_API_CSV, index=False, encoding="utf-8-sig"
    )
    official_only.to_csv(OFFICIAL_ONLY_CSV, index=False, encoding="utf-8-sig")
    OUTPUT_HTML.write_text(build_html(features, len(api_only), len(official_only)), encoding="utf-8")

    counts = source_counts(features)
    print(f"API roads: {len(input_df)}")
    print(f"Official roads: {len(features)}")
    print(f"API matched or supplemented: {counts.get('API', 0) + counts.get('공식폭보완', 0)}")
    print(f"Official-only supplemented roads: {counts.get('공식만', 0)}")
    print(f"API-only unmatched roads: {len(api_only)}")
    print(f"HTML: {OUTPUT_HTML}")
    print(f"GeoJSON: {OUTPUT_GEOJSON}")
    print(f"API-only CSV: {UNMATCHED_API_CSV}")
    print(f"Official-only CSV: {OFFICIAL_ONLY_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
