import sys, json, math, os
sys.stdout.reconfigure(encoding='utf-8')

with open('c:/Users/USER/Documents/GitHub/기말공모전/NJT-PJT/data/map_data.json', encoding='utf-8') as f:
    d = json.load(f)

places_json = json.dumps(d['places'], ensure_ascii=False)
grid_json   = json.dumps(d['grid'],   ensure_ascii=False)

LAT_STEP = 15 / 111320
LNG_STEP = 15 / (111320 * math.cos(37.5 * math.pi / 180))

parts = []

parts.append("""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>서울 숙박시설 밀집도 (15m 그리드)</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
html,body{width:100%;height:100%;background:#0e0e1a;font-family:'Segoe UI',sans-serif}
#map{width:100%;height:100vh}
#panel{position:fixed;top:14px;left:14px;background:rgba(10,10,25,0.92);border:1px solid rgba(255,255,255,0.15);border-radius:10px;padding:14px 16px;color:#ddd;font-size:13px;z-index:2000;min-width:190px}
#panel h3{color:#ffb432;font-size:14px;margin-bottom:10px}
#panel .stat{display:flex;justify-content:space-between;margin-bottom:4px;gap:16px}
#panel .val{color:#fff;font-weight:600}
#legend{position:fixed;bottom:22px;right:14px;background:rgba(10,10,25,0.92);border:1px solid rgba(255,255,255,0.15);border-radius:10px;padding:12px 16px;color:#ccc;font-size:12px;z-index:2000}
#legend h4{color:#aaa;font-size:11px;margin-bottom:8px;text-transform:uppercase;letter-spacing:1px}
.lg-row{display:flex;align-items:center;margin-bottom:5px;gap:8px}
.lg-cell{width:20px;height:20px;border-radius:2px;flex-shrink:0}
#layers{position:fixed;top:14px;right:14px;background:rgba(10,10,25,0.92);border:1px solid rgba(255,255,255,0.15);border-radius:10px;padding:12px 16px;color:#ccc;font-size:13px;z-index:2000}
#layers h4{color:#aaa;font-size:11px;margin-bottom:8px;text-transform:uppercase;letter-spacing:1px}
.toggle{display:flex;align-items:center;gap:8px;margin-bottom:6px;cursor:pointer}
.toggle input{cursor:pointer;accent-color:#ffb432}
#zoom-note{position:fixed;bottom:22px;left:50%;transform:translateX(-50%);background:rgba(10,10,25,0.88);border:1px solid rgba(255,180,50,0.3);border-radius:20px;padding:6px 18px;color:#ffb432;font-size:12px;z-index:2000;pointer-events:none;transition:opacity 0.3s}
.leaflet-popup-content-wrapper{background:rgba(10,10,25,0.96);color:#eee;border-radius:8px;border:1px solid rgba(255,255,255,0.4);box-shadow:0 4px 20px rgba(0,0,0,0.7)}
.leaflet-popup-tip{background:rgba(10,10,25,0.96)}
.leaflet-popup-content{margin:10px 14px}
.p-name{font-size:14px;font-weight:700;color:#fff;margin-bottom:4px}
.p-addr{font-size:11px;color:#999;margin-bottom:5px}
.p-meta{font-size:11px;color:#bbb}
</style>
</head>
<body>
<div id="map"></div>

<div id="panel">
  <h3>서울 숙박시설 밀집도</h3>
  <div class="stat"><span>총 시설</span><span class="val">4,246 개소</span></div>
  <div class="stat"><span>점유 15m 셀</span><span class="val">4,074</span></div>
  <div class="stat"><span>밀집 셀 (2+)</span><span class="val">171</span></div>
  <div class="stat"><span>현재 줌</span><span class="val" id="zoom-val">-</span></div>
</div>

<div id="layers">
  <h4>레이어</h4>
  <label class="toggle"><input type="checkbox" id="chk-grid" checked> 15m 그리드</label>
  <label class="toggle"><input type="checkbox" id="chk-bldg" checked> 건물 위치 (줌15+)</label>
  <label class="toggle"><input type="checkbox" id="chk-dot"  checked> 마커</label>
</div>

<div id="legend">
  <h4>밀집도 (15m 셀)</h4>
  <div class="lg-row"><div class="lg-cell" style="background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.25)"></div><span>빈 셀</span></div>
  <div class="lg-row"><div class="lg-cell" style="background:rgba(80,140,255,0.4);border:1px solid rgba(80,140,255,0.9)"></div><span>1개 시설</span></div>
  <div class="lg-row"><div class="lg-cell" style="background:rgba(80,220,120,0.4);border:1px solid rgba(80,220,120,0.95)"></div><span>2개 시설</span></div>
  <div class="lg-row"><div class="lg-cell" style="background:rgba(255,80,30,0.4);border:1px solid rgba(255,80,30,1)"></div><span>3개+ 시설</span></div>
  <hr style="border-color:rgba(255,255,255,0.1);margin:8px 0">
  <div class="lg-row"><div class="lg-cell" style="background:rgba(255,255,255,0.05);border:2px solid rgba(255,255,255,0.9)"></div><span>건물 위치 (줌15+)</span></div>
</div>

<div id="zoom-note">줌 15 이상에서 건물 위치가 표시됩니다</div>
""")

parts.append("<script>\n")
parts.append("const PLACES   = " + places_json + ";\n")
parts.append("const GRID_OCC = " + grid_json   + ";\n")
parts.append("const LAT_STEP = " + repr(LAT_STEP) + ";\n")
parts.append("const LNG_STEP = " + repr(LNG_STEP) + ";\n")

parts.append("""
// ── 지도 ──────────────────────────────────────────────────────
const map = L.map('map', {center:[37.5530, 126.9740], zoom:13, zoomControl:true});
L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
  attribution:'© CARTO © OSM', subdomains:'abcd', maxZoom:20
}).addTo(map);

// ── 점유 셀 조회 Map ──────────────────────────────────────────
const cellMap = new Map();
GRID_OCC.forEach(function(g) {
  var kr = Math.round(g.lat / LAT_STEP);
  var kc = Math.round(g.lng / LNG_STEP);
  cellMap.set(kr + ',' + kc, g.count);
});

// ── Canvas 균일 그리드 (서울 전체, 뷰포트 동적 렌더링) ────────
var gridVisible = true;

var CanvasGrid = L.Layer.extend({
  onAdd: function(map) {
    this._map = map;
    var cv = document.createElement('canvas');
    cv.style.cssText = 'position:absolute;top:0;left:0;pointer-events:none;z-index:300;';
    map.getContainer().appendChild(cv);
    this._canvas = cv;
    map.on('moveend zoomend resize', this._draw, this);
    this._draw();
  },
  onRemove: function(map) {
    map.getContainer().removeChild(this._canvas);
    map.off('moveend zoomend resize', this._draw, this);
  },
  _draw: function() {
    var map   = this._map;
    var size  = map.getSize();
    var cv    = this._canvas;
    cv.width  = size.x;
    cv.height = size.y;
    var ctx   = cv.getContext('2d');
    ctx.clearRect(0, 0, size.x, size.y);

    if (!gridVisible) return;

    var bounds = map.getBounds();
    var minLat = Math.floor(bounds.getSouth() / LAT_STEP) * LAT_STEP;
    var maxLat = Math.ceil (bounds.getNorth() / LAT_STEP) * LAT_STEP;
    var minLng = Math.floor(bounds.getWest()  / LNG_STEP) * LNG_STEP;
    var maxLng = Math.ceil (bounds.getEast()  / LNG_STEP) * LNG_STEP;

    // 셀 픽셀 크기 미리 계산 (대표값)
    var refSW = map.latLngToContainerPoint([minLat,            minLng           ]);
    var refNE = map.latLngToContainerPoint([minLat + LAT_STEP, minLng + LNG_STEP]);
    var cellPx = Math.max(Math.abs(refNE.x - refSW.x), Math.abs(refSW.y - refNE.y));

    // 너무 작으면 빈 셀 선 생략 (성능)
    var drawEmpty = cellPx >= 2;

    for (var lat = minLat; lat < maxLat; lat += LAT_STEP) {
      for (var lng = minLng; lng < maxLng; lng += LNG_STEP) {
        var sw = map.latLngToContainerPoint([lat,            lng           ]);
        var ne = map.latLngToContainerPoint([lat + LAT_STEP, lng + LNG_STEP]);
        var x  = ne.x,  y = ne.y;
        var w  = sw.x - ne.x,  h = sw.y - ne.y;
        if (w < 0.5 || h < 0.5) continue;

        var kr    = Math.round(lat / LAT_STEP);
        var kc    = Math.round(lng / LNG_STEP);
        var count = cellMap.get(kr + ',' + kc) || 0;

        if (count >= 3) {
          // 빨강 - 3개+
          ctx.fillStyle   = 'rgba(255,80,30,0.4)';
          ctx.strokeStyle = 'rgba(255,80,30,0.9)';
          ctx.lineWidth   = 1;
          ctx.fillRect(x, y, w, h);
          ctx.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1);
        } else if (count === 2) {
          // 초록 - 2개
          ctx.fillStyle   = 'rgba(80,220,120,0.4)';
          ctx.strokeStyle = 'rgba(80,220,120,0.9)';
          ctx.lineWidth   = 1;
          ctx.fillRect(x, y, w, h);
          ctx.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1);
        } else if (count === 1) {
          // 파랑 - 1개
          ctx.fillStyle   = 'rgba(80,140,255,0.4)';
          ctx.strokeStyle = 'rgba(80,140,255,0.85)';
          ctx.lineWidth   = 1;
          ctx.fillRect(x, y, w, h);
          ctx.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1);
        } else if (drawEmpty) {
          // 빈 셀 - 연한 격자선만
          ctx.fillStyle   = 'rgba(255,255,255,0.04)';
          ctx.strokeStyle = 'rgba(255,255,255,0.22)';
          ctx.lineWidth   = 0.5;
          ctx.fillRect(x, y, w, h);
          ctx.strokeRect(x + 0.25, y + 0.25, w - 0.5, h - 0.5);
        }
      }
    }
  },
  redraw: function() { this._draw(); }
});

var canvasGrid = new CanvasGrid();
canvasGrid.addTo(map);

// ── 건물 위치 레이어 (줌15+, 흰색 테두리 균일 사각형) ─────────
var bldgLayer = L.layerGroup();
var R_LAT = 1 / 111320;
var HALF  = 8; // 고정 8m 반경 → 16×16m 정사각형

PLACES.forEach(function(p) {
  var cosL  = Math.cos(p.lat * Math.PI / 180);
  var R_LNG = 1 / (111320 * cosL);
  var dLat  = HALF * R_LAT;
  var dLng  = HALF * R_LNG;
  var popup = '<div class="p-name">' + p.name + '</div>'
            + '<div class="p-addr">' + p.addr + '</div>'
            + '<div class="p-meta">지상 ' + p.floors + '층 | 바닥면적 약 '
            + Math.round(p.side_m * p.side_m) + ' m²</div>';
  L.rectangle(
    [[p.lat - dLat, p.lng - dLng], [p.lat + dLat, p.lng + dLng]],
    {color: 'rgba(255,255,255,0.9)', weight: 1.5,
     fillColor: 'rgba(255,255,255,0.05)', fillOpacity: 1,
     interactive: true}
  ).bindPopup(popup, {maxWidth:280}).addTo(bldgLayer);
});

// ── 마커 레이어 ───────────────────────────────────────────────
var dotLayer = L.layerGroup();
PLACES.forEach(function(p) {
  L.circleMarker([p.lat, p.lng], {
    radius:3, color:'#ffb432', weight:1,
    fillColor:'#ffb432', fillOpacity:0.85
  }).addTo(dotLayer);
});
dotLayer.addTo(map);

// ── 줌 연동 ──────────────────────────────────────────────────
function updateLayers() {
  var z     = map.getZoom();
  var note  = document.getElementById('zoom-note');
  var showB = document.getElementById('chk-bldg').checked;
  var showD = document.getElementById('chk-dot').checked;
  document.getElementById('zoom-val').textContent = z;
  if (z >= 15) {
    if (showB && !map.hasLayer(bldgLayer)) map.addLayer(bldgLayer);
    if (map.hasLayer(dotLayer)) map.removeLayer(dotLayer);
    note.style.opacity = '0';
  } else {
    if (map.hasLayer(bldgLayer)) map.removeLayer(bldgLayer);
    if (showD && !map.hasLayer(dotLayer)) map.addLayer(dotLayer);
    note.style.opacity = '1';
  }
}
map.on('zoomend', updateLayers);
updateLayers();

// ── 체크박스 ─────────────────────────────────────────────────
document.getElementById('chk-grid').addEventListener('change', function() {
  gridVisible = this.checked;
  canvasGrid.redraw();
});
document.getElementById('chk-bldg').addEventListener('change', function() {
  if (!this.checked && map.hasLayer(bldgLayer)) map.removeLayer(bldgLayer);
  updateLayers();
});
document.getElementById('chk-dot').addEventListener('change', function() {
  if (!this.checked && map.hasLayer(dotLayer)) map.removeLayer(dotLayer);
  updateLayers();
});
</script>
</body>
</html>
""")

out = 'c:/Users/USER/Documents/GitHub/기말공모전/NJT-PJT/숙박밀집도_grid.html'
with open(out, 'w', encoding='utf-8') as fout:
    fout.write(''.join(parts))

print('Done:', os.path.getsize(out)//1024, 'KB')
