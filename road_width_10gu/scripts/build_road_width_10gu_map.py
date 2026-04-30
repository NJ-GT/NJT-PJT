from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_PATH = BASE_DIR / "data" / "seoul_road_width_viRoutDt_10개구_대표좌표.csv"
OUTPUT_PATH = BASE_DIR / "data" / "seoul_road_width_10gu_map.html"

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

WIDTH_RADIUS = {
    "6m미만": 4,
    "폭6-8m": 5,
    "폭8-10m": 6,
    "폭10-12m": 7,
    "폭15-20m": 8,
    "폭20-25m": 9,
    "폭25-30m": 10,
    "폭30-35m": 11,
    "폭35-40m": 12,
    "폭40-50m": 13,
    "폭50-70m": 14,
    "불가": 5,
}


def as_int(value: object, default: int = 0) -> int:
    try:
        return int(float(str(value).strip()))
    except Exception:
        return default


def load_points() -> list[dict[str, object]]:
    df = pd.read_csv(INPUT_PATH, encoding="utf-8-sig", dtype=str).fillna("")
    df["위도"] = pd.to_numeric(df["위도"], errors="coerce")
    df["경도"] = pd.to_numeric(df["경도"], errors="coerce")
    df = df.dropna(subset=["위도", "경도"]).copy()
    df = df.sort_values(["구", "도로폭", "도로명"], kind="stable")

    points: list[dict[str, object]] = []
    for _, row in df.iterrows():
        width = str(row["도로폭"]).strip() or "불가"
        points.append(
            {
                "id": as_int(row["순번"]),
                "gu": str(row["구"]).strip(),
                "road": str(row["도로명"]).strip(),
                "kind": str(row["도로구분"]).strip(),
                "function": str(row["도로기능"]).strip(),
                "scale": str(row["도로규모"]).strip(),
                "width": width,
                "lat": round(float(row["위도"]), 12),
                "lng": round(float(row["경도"]), 12),
                "address": str(row["좌표조회주소"]).strip(),
                "candidateCount": as_int(row["좌표후보수"]),
                "color": WIDTH_COLORS.get(width, WIDTH_COLORS["불가"]),
                "radius": WIDTH_RADIUS.get(width, WIDTH_RADIUS["불가"]),
            }
        )
    return points


def count_by(points: list[dict[str, object]], key: str, order: list[str]) -> list[dict[str, object]]:
    counts = {name: 0 for name in order}
    for point in points:
        value = str(point.get(key, ""))
        counts[value] = counts.get(value, 0) + 1
    return [{"name": name, "count": counts[name]} for name in order if counts.get(name, 0)]


def build_html(points: list[dict[str, object]]) -> str:
    gu_counts = count_by(points, "gu", GU_ORDER)
    width_counts = count_by(points, "width", WIDTH_ORDER)
    center_lat = sum(float(point["lat"]) for point in points) / len(points)
    center_lng = sum(float(point["lng"]) for point in points) / len(points)

    data_json = json.dumps(points, ensure_ascii=False, separators=(",", ":"))
    gu_json = json.dumps(GU_ORDER, ensure_ascii=False)
    width_json = json.dumps(WIDTH_ORDER, ensure_ascii=False)
    colors_json = json.dumps(WIDTH_COLORS, ensure_ascii=False)
    gu_counts_json = json.dumps(gu_counts, ensure_ascii=False, separators=(",", ":"))
    width_counts_json = json.dumps(width_counts, ensure_ascii=False, separators=(",", ":"))

    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>서울 10개구 도로폭 지도</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <style>
    :root {{
      --ink: #172033;
      --muted: #697386;
      --line: #d8dde8;
      --panel: rgba(255, 255, 255, 0.96);
      --accent: #2563eb;
    }}
    * {{ box-sizing: border-box; }}
    html, body {{ height: 100%; margin: 0; }}
    body {{
      font-family: "Malgun Gothic", "Apple SD Gothic Neo", Arial, sans-serif;
      color: var(--ink);
      background: #eef2f7;
    }}
    #map {{ position: fixed; inset: 0; }}
    .panel {{
      position: fixed;
      top: 16px;
      left: 16px;
      z-index: 1000;
      width: 360px;
      max-height: calc(100vh - 32px);
      overflow: auto;
      background: var(--panel);
      border: 1px solid rgba(122, 132, 154, 0.32);
      border-radius: 8px;
      box-shadow: 0 18px 40px rgba(28, 39, 64, 0.18);
      backdrop-filter: blur(8px);
    }}
    .panel-header {{
      padding: 16px 16px 12px;
      border-bottom: 1px solid var(--line);
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 19px;
      line-height: 1.28;
      font-weight: 800;
      letter-spacing: 0;
    }}
    .meta {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 8px;
      margin-top: 12px;
    }}
    .metric {{
      min-width: 0;
      padding: 9px 8px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #f8fafc;
    }}
    .metric strong {{
      display: block;
      font-size: 17px;
      line-height: 1.1;
    }}
    .metric span {{
      display: block;
      margin-top: 4px;
      font-size: 11px;
      color: var(--muted);
      white-space: nowrap;
    }}
    .panel-body {{ padding: 14px 16px 16px; }}
    .field {{ margin-bottom: 14px; }}
    .field label {{
      display: block;
      margin-bottom: 7px;
      font-size: 12px;
      font-weight: 700;
      color: #35415a;
    }}
    .search-row {{
      display: grid;
      grid-template-columns: 1fr 72px;
      gap: 8px;
    }}
    input[type="search"] {{
      width: 100%;
      height: 36px;
      padding: 0 10px;
      border: 1px solid #bcc6d8;
      border-radius: 6px;
      font-size: 13px;
      outline: none;
      background: white;
    }}
    input[type="search"]:focus {{
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.14);
    }}
    button {{
      height: 36px;
      border: 1px solid #bcc6d8;
      border-radius: 6px;
      background: #ffffff;
      color: #253148;
      font-weight: 700;
      font-size: 12px;
      cursor: pointer;
    }}
    button:hover {{ background: #f1f5fb; }}
    .chips {{
      display: flex;
      flex-wrap: wrap;
      gap: 7px;
    }}
    .chip {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      min-height: 30px;
      max-width: 100%;
      padding: 5px 8px;
      border: 1px solid #c8d0df;
      border-radius: 6px;
      background: white;
      font-size: 12px;
      line-height: 1.2;
      user-select: none;
      cursor: pointer;
    }}
    .chip input {{ margin: 0; }}
    .swatch {{
      width: 10px;
      height: 10px;
      border-radius: 50%;
      flex: 0 0 auto;
      border: 1px solid rgba(0,0,0,.16);
    }}
    .summary {{
      margin-top: 10px;
      border-top: 1px solid var(--line);
      padding-top: 12px;
    }}
    .rows {{
      display: grid;
      gap: 6px;
      font-size: 12px;
    }}
    .bar-row {{
      display: grid;
      grid-template-columns: 64px 1fr 42px;
      align-items: center;
      gap: 8px;
      min-height: 22px;
    }}
    .bar-label, .bar-count {{
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }}
    .bar-track {{
      height: 8px;
      border-radius: 999px;
      background: #e8edf5;
      overflow: hidden;
    }}
    .bar-fill {{
      height: 100%;
      min-width: 2px;
      border-radius: 999px;
      background: #4169a8;
    }}
    .legend-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 6px 8px;
      font-size: 12px;
    }}
    .legend-item {{
      display: flex;
      align-items: center;
      min-width: 0;
      gap: 6px;
    }}
    .legend-item span:last-child {{
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }}
    .leaflet-popup-content {{
      font-family: "Malgun Gothic", "Apple SD Gothic Neo", Arial, sans-serif;
      font-size: 12px;
      margin: 12px;
      min-width: 220px;
    }}
    .popup-title {{
      font-size: 14px;
      font-weight: 800;
      margin-bottom: 6px;
    }}
    .popup-table {{
      width: 100%;
      border-collapse: collapse;
    }}
    .popup-table th {{
      width: 68px;
      text-align: left;
      color: #667085;
      font-weight: 700;
      padding: 3px 8px 3px 0;
      vertical-align: top;
    }}
    .popup-table td {{ padding: 3px 0; vertical-align: top; }}
    @media (max-width: 760px) {{
      .panel {{
        top: auto;
        left: 8px;
        right: 8px;
        bottom: 8px;
        width: auto;
        max-height: 48vh;
      }}
      .panel-header {{ padding: 13px 13px 10px; }}
      .panel-body {{ padding: 12px 13px 13px; }}
      h1 {{ font-size: 17px; }}
      .meta {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
      .legend-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    }}
  </style>
</head>
<body>
  <div id="map" aria-label="서울 10개구 도로폭 지도"></div>
  <aside class="panel">
    <div class="panel-header">
      <h1>서울 10개구 도로폭 대표좌표</h1>
      <div class="meta">
        <div class="metric"><strong id="visibleCount">0</strong><span>표시 도로</span></div>
        <div class="metric"><strong id="visibleGu">0</strong><span>자치구</span></div>
        <div class="metric"><strong id="visibleWidth">0</strong><span>도로폭</span></div>
      </div>
    </div>
    <div class="panel-body">
      <div class="field">
        <label for="search">검색</label>
        <div class="search-row">
          <input id="search" type="search" placeholder="도로명, 주소">
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
      <div class="summary">
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
    const map = L.map("map", {{
      preferCanvas: true,
      zoomControl: false,
      attributionControl: true
    }}).setView(MAP_CENTER, 11);
    L.control.zoom({{ position: "bottomright" }}).addTo(map);
    L.tileLayer("https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png", {{
      maxZoom: 19,
      attribution: '&copy; OpenStreetMap contributors &copy; CARTO'
    }}).addTo(map);

    const layer = L.layerGroup().addTo(map);
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

    function markerPopup(point) {{
      return `
        <div class="popup-title">${{escapeHtml(point.road)}}</div>
        <table class="popup-table">
          <tr><th>구</th><td>${{escapeHtml(point.gu)}}</td></tr>
          <tr><th>도로폭</th><td>${{escapeHtml(point.width)}}</td></tr>
          <tr><th>도로규모</th><td>${{escapeHtml(point.scale)}}</td></tr>
          <tr><th>도로기능</th><td>${{escapeHtml(point.function)}}</td></tr>
          <tr><th>도로구분</th><td>${{escapeHtml(point.kind)}}</td></tr>
          <tr><th>조회주소</th><td>${{escapeHtml(point.address)}}</td></tr>
        </table>`;
    }}

    function filteredData() {{
      const selectedGu = getSelected("gu");
      const selectedWidth = getSelected("width");
      const query = document.getElementById("search").value.trim().toLowerCase();
      return DATA.filter((point) => {{
        if (!selectedGu.has(point.gu)) return false;
        if (!selectedWidth.has(point.width)) return false;
        if (!query) return true;
        return `${{point.road}} ${{point.address}} ${{point.gu}} ${{point.width}}`.toLowerCase().includes(query);
      }});
    }}

    function updateSummary(points) {{
      const guCounts = new Map();
      const widthValues = new Set();
      points.forEach((point) => {{
        guCounts.set(point.gu, (guCounts.get(point.gu) || 0) + 1);
        widthValues.add(point.width);
      }});
      document.getElementById("visibleCount").textContent = points.length.toLocaleString("ko-KR");
      document.getElementById("visibleGu").textContent = guCounts.size.toLocaleString("ko-KR");
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

    function render() {{
      const points = filteredData();
      layer.clearLayers();
      const bounds = [];
      points.forEach((point) => {{
        const latlng = [point.lat, point.lng];
        bounds.push(latlng);
        L.circleMarker(latlng, {{
          renderer: canvasRenderer,
          radius: point.radius,
          color: "#172033",
          weight: 0.8,
          opacity: 0.65,
          fillColor: point.color,
          fillOpacity: 0.78
        }})
          .bindTooltip(`[${{point.gu}}] ${{point.road}} | ${{point.width}}`, {{ sticky: true }})
          .bindPopup(markerPopup(point), {{ maxWidth: 300 }})
          .addTo(layer);
      }});
      updateSummary(points);
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
  </script>
</body>
</html>
"""


def main() -> int:
    points = load_points()
    if not points:
        raise RuntimeError(f"No valid points found in {INPUT_PATH}")
    OUTPUT_PATH.write_text(build_html(points), encoding="utf-8")
    print(f"Input: {INPUT_PATH}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Points: {len(points)}")
    print("By district:")
    for item in count_by(points, "gu", GU_ORDER):
        print(f"  {item['name']}: {item['count']}")
    print("By road width:")
    for item in count_by(points, "width", WIDTH_ORDER):
        print(f"  {item['name']}: {item['count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
