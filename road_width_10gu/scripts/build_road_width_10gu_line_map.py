from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
import requests


BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_PATH = BASE_DIR / "data" / "seoul_road_width_viRoutDt_10개구_대표좌표.csv"
OSM_CACHE_PATH = BASE_DIR / "data" / "osm_seoul_10gu_named_highways.json"
MATCHED_GEOJSON_PATH = BASE_DIR / "data" / "seoul_road_width_10gu_osm_lines.geojson"
UNMATCHED_PATH = BASE_DIR / "data" / "seoul_road_width_10gu_osm_line_unmatched.csv"
OUTPUT_PATH = BASE_DIR / "data" / "seoul_road_width_10gu_line_map.html"

OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter",
]

GU_ORDER = ["강남구", "강서구", "마포구", "서초구", "성동구", "송파구", "영등포구", "용산구", "종로구", "중구"]
WIDTH_ORDER = [
    "6m미만",
    "폭6-8m",
    "폭8-10m",
    "폭10-12m",
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


def distance_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    mean_lat = math.radians((lat1 + lat2) / 2)
    dx = (lng2 - lng1) * 111_320 * math.cos(mean_lat)
    dy = (lat2 - lat1) * 110_574
    return math.hypot(dx, dy)


def simplify_line(coords: list[list[float]], tolerance_m: float = 3.0) -> list[list[float]]:
    if len(coords) <= 2:
        return coords

    start = coords[0]
    end = coords[-1]
    line_len = distance_m(start[1], start[0], end[1], end[0])
    if line_len == 0:
        return [start, end]

    max_dist = -1.0
    max_idx = 0
    for idx in range(1, len(coords) - 1):
        point = coords[idx]
        d = perpendicular_distance_m(point, start, end)
        if d > max_dist:
            max_dist = d
            max_idx = idx

    if max_dist > tolerance_m:
        left = simplify_line(coords[: max_idx + 1], tolerance_m)
        right = simplify_line(coords[max_idx:], tolerance_m)
        return left[:-1] + right
    return [start, end]


def perpendicular_distance_m(point: list[float], start: list[float], end: list[float]) -> float:
    lat0 = point[1]
    lng0 = point[0]
    lat1 = start[1]
    lng1 = start[0]
    lat2 = end[1]
    lng2 = end[0]
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


class DSU:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def load_input() -> pd.DataFrame:
    df = pd.read_csv(INPUT_PATH, encoding="utf-8-sig", dtype=str).fillna("")
    df["위도"] = pd.to_numeric(df["위도"], errors="coerce")
    df["경도"] = pd.to_numeric(df["경도"], errors="coerce")
    df = df.dropna(subset=["위도", "경도"]).copy()
    df["_road_key"] = df["도로명"].map(normalize_name)
    return df


def fetch_osm_ways(df: pd.DataFrame) -> dict:
    if OSM_CACHE_PATH.exists():
        return json.loads(OSM_CACHE_PATH.read_text(encoding="utf-8"))

    south = max(37.3, float(df["위도"].min()) - 0.03)
    north = min(37.8, float(df["위도"].max()) + 0.03)
    west = max(126.65, float(df["경도"].min()) - 0.03)
    east = min(127.25, float(df["경도"].max()) + 0.03)
    bbox = f"{south:.7f},{west:.7f},{north:.7f},{east:.7f}"
    query = f"""
[out:json][timeout:240];
(
  way["highway"]["name"]({bbox});
  way["highway"]["name:ko"]({bbox});
);
out geom tags;
"""
    last_error: Exception | None = None
    for url in OVERPASS_URLS:
        for attempt in range(2):
            try:
                response = requests.get(
                    url,
                    params={"data": query},
                    headers={"User-Agent": "seoul-road-width-map/1.0"},
                    timeout=300,
                )
                response.raise_for_status()
                data = response.json()
                OSM_CACHE_PATH.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
                return data
            except Exception as exc:
                last_error = exc
                time.sleep(3 + attempt * 5)
    raise RuntimeError(f"Failed to fetch OSM ways: {last_error}")


def index_ways(osm: dict) -> dict[str, list[dict]]:
    index: dict[str, list[dict]] = defaultdict(list)
    for element in osm.get("elements", []):
        if element.get("type") != "way":
            continue
        geometry = element.get("geometry") or []
        if len(geometry) < 2:
            continue
        tags = element.get("tags") or {}
        names = {tags.get("name", ""), tags.get("name:ko", ""), tags.get("addr:street", "")}
        way = {
            "id": element.get("id"),
            "nodes": element.get("nodes") or [],
            "tags": tags,
            "coords": [[float(pt["lon"]), float(pt["lat"])] for pt in geometry],
        }
        for name in names:
            key = normalize_name(name)
            if key:
                index[key].append(way)
    return index


def way_distance_to_point(way: dict, lat: float, lng: float) -> float:
    return min(distance_m(lat, lng, coord[1], coord[0]) for coord in way["coords"])


def component_for_row(ways: list[dict], lat: float, lng: float) -> list[dict]:
    if len(ways) <= 1:
        return ways
    dsu = DSU(len(ways))
    node_to_way: dict[int, int] = {}
    for way_idx, way in enumerate(ways):
        for node_id in way.get("nodes") or []:
            if node_id in node_to_way:
                dsu.union(way_idx, node_to_way[node_id])
            else:
                node_to_way[node_id] = way_idx

    nearest_idx = min(range(len(ways)), key=lambda idx: way_distance_to_point(ways[idx], lat, lng))
    nearest_root = dsu.find(nearest_idx)
    component = [way for idx, way in enumerate(ways) if dsu.find(idx) == nearest_root]
    return component or [ways[nearest_idx]]


def endpoint_pair(ways: list[dict]) -> tuple[list[float], list[float]]:
    node_degree: dict[int, int] = defaultdict(int)
    node_coord: dict[int, list[float]] = {}
    all_coords: list[list[float]] = []

    for way in ways:
        nodes = way.get("nodes") or []
        coords = way["coords"]
        all_coords.extend(coords)
        for node_id, coord in zip(nodes, coords):
            node_coord[node_id] = coord
        for a, b in zip(nodes, nodes[1:]):
            node_degree[a] += 1
            node_degree[b] += 1

    candidates = [node_coord[node_id] for node_id, degree in node_degree.items() if degree == 1 and node_id in node_coord]
    if len(candidates) < 2:
        candidates = all_coords
    if len(candidates) < 2:
        return all_coords[0], all_coords[-1]

    best_pair = (candidates[0], candidates[-1])
    best_dist = -1.0
    for i in range(len(candidates)):
        a = candidates[i]
        for j in range(i + 1, len(candidates)):
            b = candidates[j]
            d = distance_m(a[1], a[0], b[1], b[0])
            if d > best_dist:
                best_dist = d
                best_pair = (a, b)
    return best_pair


def line_length_m(ways: list[dict]) -> float:
    total = 0.0
    for way in ways:
        coords = way["coords"]
        for a, b in zip(coords, coords[1:]):
            total += distance_m(a[1], a[0], b[1], b[0])
    return total


def matched_features(df: pd.DataFrame, index: dict[str, list[dict]]) -> tuple[list[dict], pd.DataFrame]:
    features: list[dict] = []
    unmatched: list[dict] = []
    seen: set[tuple[str, str]] = set()

    for _, row in df.iterrows():
        key = str(row["_road_key"])
        gu = str(row["구"])
        dedupe_key = (gu, key)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        ways = index.get(key, [])
        if not ways:
            unmatched.append(row.to_dict())
            continue

        selected = component_for_row(ways, float(row["위도"]), float(row["경도"]))
        start, end = endpoint_pair(selected)
        geometry = [simplify_line(way["coords"], tolerance_m=3.0) for way in selected]
        length = line_length_m(selected)
        width = str(row["도로폭"]).strip() or "불가"
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "MultiLineString", "coordinates": geometry},
                "properties": {
                    "순번": int(float(row["순번"])),
                    "구": gu,
                    "도로명": str(row["도로명"]),
                    "도로구분": str(row["도로구분"]),
                    "도로기능": str(row["도로기능"]),
                    "도로규모": str(row["도로규모"]),
                    "도로폭": width,
                    "대표위도": float(row["위도"]),
                    "대표경도": float(row["경도"]),
                    "시작위도": start[1],
                    "시작경도": start[0],
                    "끝위도": end[1],
                    "끝경도": end[0],
                    "선형길이m": round(length, 1),
                    "OSM_way_count": len(selected),
                    "색상": WIDTH_COLORS.get(width, WIDTH_COLORS["불가"]),
                    "두께": WIDTH_WEIGHTS.get(width, WIDTH_WEIGHTS["불가"]),
                },
            }
        )

    return features, pd.DataFrame(unmatched)


def count_by(features: list[dict], prop: str, order: list[str]) -> list[dict[str, object]]:
    counts = {name: 0 for name in order}
    for feature in features:
        value = str(feature["properties"].get(prop, ""))
        counts[value] = counts.get(value, 0) + 1
    return [{"name": name, "count": counts[name]} for name in order if counts.get(name, 0)]


def to_leaflet_features(features: list[dict]) -> list[dict[str, object]]:
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
                "width": props["도로폭"],
                "lengthM": props["선형길이m"],
                "wayCount": props["OSM_way_count"],
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


def build_html(features: list[dict], unmatched_count: int) -> str:
    rows = to_leaflet_features(features)
    center_lat = sum((row["start"][0] + row["end"][0]) / 2 for row in rows) / len(rows)
    center_lng = sum((row["start"][1] + row["end"][1]) / 2 for row in rows) / len(rows)
    data_json = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
    gu_json = json.dumps(GU_ORDER, ensure_ascii=False)
    width_json = json.dumps(WIDTH_ORDER, ensure_ascii=False)
    colors_json = json.dumps(WIDTH_COLORS, ensure_ascii=False)
    gu_counts_json = json.dumps(count_by(features, "구", GU_ORDER), ensure_ascii=False, separators=(",", ":"))
    width_counts_json = json.dumps(count_by(features, "도로폭", WIDTH_ORDER), ensure_ascii=False, separators=(",", ":"))

    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>서울 10개구 도로폭 선형 지도</title>
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
      width: 378px;
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
    .leaflet-popup-content {{ font-family: "Malgun Gothic", "Apple SD Gothic Neo", Arial, sans-serif; font-size: 12px; margin: 12px; min-width: 240px; }}
    .popup-title {{ font-size: 14px; font-weight: 800; margin-bottom: 6px; }}
    .popup-table {{ width: 100%; border-collapse: collapse; }}
    .popup-table th {{ width: 72px; text-align: left; color: #667085; font-weight: 700; padding: 3px 8px 3px 0; vertical-align: top; }}
    .popup-table td {{ padding: 3px 0; vertical-align: top; }}
    @media (max-width: 760px) {{
      .panel {{ top: auto; left: 8px; right: 8px; bottom: 8px; width: auto; max-height: 50vh; }}
      .panel-header {{ padding: 13px; }}
      .panel-body {{ padding: 12px 13px 13px; }}
      h1 {{ font-size: 17px; }}
    }}
  </style>
</head>
<body>
  <div id="map" aria-label="서울 10개구 도로폭 선형 지도"></div>
  <aside class="panel">
    <div class="panel-header">
      <h1>서울 10개구 도로폭 선형</h1>
      <p class="note">OSM 도로망과 도로명을 매칭해 선으로 표시했습니다. 도로선을 클릭하면 해당 도로의 시작과 끝이 표시됩니다.</p>
      <div class="meta">
        <div class="metric"><strong id="visibleCount">0</strong><span>표시 도로</span></div>
        <div class="metric"><strong>{unmatched_count:,}</strong><span>선형 미매칭</span></div>
        <div class="metric"><strong id="visibleWidth">0</strong><span>도로폭</span></div>
      </div>
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
    const WIDTH_ORDER = {width_json};
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
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;"
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
    function popupHtml(row) {{
      return `
        <div class="popup-title">${{escapeHtml(row.road)}}</div>
        <table class="popup-table">
          <tr><th>구</th><td>${{escapeHtml(row.gu)}}</td></tr>
          <tr><th>도로폭</th><td>${{escapeHtml(row.width)}}</td></tr>
          <tr><th>도로규모</th><td>${{escapeHtml(row.scale)}}</td></tr>
          <tr><th>도로기능</th><td>${{escapeHtml(row.function)}}</td></tr>
          <tr><th>길이</th><td>${{Math.round(row.lengthM).toLocaleString("ko-KR")}} m</td></tr>
          <tr><th>OSM 선</th><td>${{row.wayCount.toLocaleString("ko-KR")}}개</td></tr>
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
        return `${{row.road}} ${{row.gu}} ${{row.width}}`.toLowerCase().includes(query);
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
      document.getElementById("visibleWidth").textContent = widthValues.size.toLocaleString("ko-KR");
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
          opacity: 0.9
        }}).addTo(highlightLayer);
        L.polyline(line, {{
          color: row.color,
          weight: Math.max(row.weight + 1, 4),
          opacity: 1
        }}).addTo(highlightLayer);
      }});
      L.circleMarker(row.start, {{
        radius: 7,
        color: "#ffffff",
        weight: 2,
        fillColor: "#111827",
        fillOpacity: 0.96
      }})
        .bindTooltip(`시작: ${{row.road}}`, {{ permanent: true, direction: "top", offset: [0, -8] }})
        .addTo(highlightLayer);
      L.circleMarker(row.end, {{
        radius: 7,
        color: "#ffffff",
        weight: 2,
        fillColor: "#f43f5e",
        fillOpacity: 0.96
      }})
        .bindTooltip(`끝: ${{row.road}}`, {{ permanent: true, direction: "top", offset: [0, -8] }})
        .addTo(highlightLayer);
    }}
    function showHover(row) {{
      hoverLayer.clearLayers();
      row.lines.forEach((line) => {{
        L.polyline(line, {{
          color: "#ffffff",
          weight: Math.max(row.weight + 5, 8),
          opacity: 0.95
        }}).addTo(hoverLayer);
        L.polyline(line, {{
          color: row.color,
          weight: Math.max(row.weight + 2, 5),
          opacity: 1
        }}).addTo(hoverLayer);
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
            opacity: 0.92
          }});
          polyline
            .bindTooltip(`[${{row.gu}}] ${{row.road}} | ${{row.width}}`, {{ sticky: true }})
            .bindPopup(popupHtml(row), {{ maxWidth: 320 }})
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
    if (initialRoad) {{
      showEndpoints(initialRoad);
    }}
  </script>
</body>
</html>
"""


def main() -> int:
    df = load_input()
    osm = fetch_osm_ways(df)
    index = index_ways(osm)
    features, unmatched = matched_features(df, index)
    geojson = {"type": "FeatureCollection", "features": features}
    MATCHED_GEOJSON_PATH.write_text(json.dumps(geojson, ensure_ascii=False), encoding="utf-8")
    if unmatched.empty:
        unmatched = pd.DataFrame(columns=df.columns)
    unmatched.drop(columns=[col for col in ["_road_key"] if col in unmatched.columns]).to_csv(
        UNMATCHED_PATH, index=False, encoding="utf-8-sig"
    )
    if not features:
        raise RuntimeError("No road line geometries matched.")
    OUTPUT_PATH.write_text(build_html(features, len(unmatched)), encoding="utf-8")

    print(f"Input rows: {len(df)}")
    print(f"OSM ways: {len(osm.get('elements', []))}")
    print(f"Matched line roads: {len(features)}")
    print(f"Unmatched roads: {len(unmatched)}")
    print(f"GeoJSON: {MATCHED_GEOJSON_PATH}")
    print(f"Unmatched CSV: {UNMATCHED_PATH}")
    print(f"HTML: {OUTPUT_PATH}")
    print("By district:")
    for item in count_by(features, "구", GU_ORDER):
        print(f"  {item['name']}: {item['count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
