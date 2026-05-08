# -*- coding: utf-8 -*-
"""
서울 법정동 단계구분도(Choropleth) 인터랙티브 HTML 생성 스크립트.

목적:
    Leaflet으로 법정동 경계를 그리고, 4가지 밀집도 지표 중 하나를
    드롭다운으로 전환하면서 색칠해 시각화한다.

지원 지표:
    - per_ha : 시설수 / ha (공간 밀집)
    - count  : 숙박시설 수 (절대량)
    - fa_ha  : 숙박 연면적 / ha (규모 밀집)
    - ratio  : 전체 건물 중 숙박 비율(%)

상호작용:
    - 동 클릭 시 우측 상단 패널에 동의 상세 정보 표시
    - 좌측 상단 드롭다운으로 지표 전환
    - 우측 하단 범례는 지표에 맞춰 자동 갱신
    - 체크박스로 개별 숙소 포인트 on/off, 줌 레벨에 따라 반경 자동 조정

데이터:
    NJT-PJT/data/dong_density.json — 동 경계 + 밀집도 속성 (GeoJSON)
    NJT-PJT/data/map_data.json     — 개별 시설 위치 정보
출력:
    NJT-PJT/법정동_숙박밀집도.html
"""

import sys
import json
import os

# Windows 콘솔에서 한글 출력 깨짐 방지
sys.stdout.reconfigure(encoding="utf-8")

# ─── 1. 법정동 GeoJSON 로드 ─────────────────────────────────────
# 파일 내용을 그대로 JS 변수에 주입할 것이므로 파싱하지 않고 문자열로 둔다
with open(
    "c:/Users/USER/Documents/GitHub/기말공모전/NJT-PJT/data/dong_density.json",
    encoding="utf-8",
) as f:
    geojson_str = f.read()

# ─── 2. 개별 숙박시설 위치 로드 ─────────────────────────────────
# 이쪽은 places 키만 추출하면 되므로 JSON 파싱 후 다시 직렬화
with open(
    "c:/Users/USER/Documents/GitHub/기말공모전/NJT-PJT/data/map_data.json",
    encoding="utf-8",
) as f:
    map_data = json.load(f)
# ensure_ascii=False — 한글 그대로 유지 (HTML 내 한국어 표기 위해)
places_json = json.dumps(map_data["places"], ensure_ascii=False)

# HTML 본문을 메모리 효율적으로 누적
parts = []

# ─── 3-1. HTML 헤더 + 좌상단 패널 + 범례 + 정보박스 UI ─────────
parts.append("""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>서울 법정동별 숙박밀집도</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
html,body{width:100%;height:100%;background:#0e0e1a;font-family:'Segoe UI',sans-serif}
#map{width:100%;height:100vh}

#panel{
  position:fixed;top:14px;left:14px;
  background:rgba(10,10,25,0.93);
  border:1px solid rgba(255,255,255,0.13);
  border-radius:10px;padding:14px 16px;
  color:#ddd;font-size:13px;z-index:2000;min-width:200px;
}
#panel h3{color:#ffb432;font-size:15px;margin-bottom:10px;font-weight:700}
#panel .stat{display:flex;justify-content:space-between;margin-bottom:5px;gap:20px}
#panel .val{color:#fff;font-weight:600}
#panel select{
  margin-top:10px;width:100%;padding:5px 8px;
  background:rgba(255,255,255,0.08);color:#fff;
  border:1px solid rgba(255,255,255,0.2);border-radius:6px;font-size:12px;cursor:pointer;
}

#legend{
  position:fixed;bottom:22px;right:14px;
  background:rgba(10,10,25,0.93);
  border:1px solid rgba(255,255,255,0.13);
  border-radius:10px;padding:12px 16px;
  color:#ccc;font-size:12px;z-index:2000;min-width:170px;
}
#legend h4{color:#aaa;font-size:11px;margin-bottom:10px;text-transform:uppercase;letter-spacing:1px}
.lg-row{display:flex;align-items:center;margin-bottom:6px;gap:10px}
.lg-box{width:22px;height:14px;border-radius:2px;flex-shrink:0;border:1px solid rgba(255,255,255,0.15)}
#legend-title{color:#ffb432;font-size:11px;margin-bottom:8px;font-weight:600}

#info-box{
  position:fixed;top:14px;right:14px;
  background:rgba(10,10,25,0.93);
  border:1px solid rgba(255,255,255,0.13);
  border-radius:10px;padding:14px 16px;
  color:#ddd;font-size:13px;z-index:2000;min-width:220px;
  display:none;
}
#info-box .ib-name{font-size:15px;font-weight:700;color:#ffb432;margin-bottom:8px}
#info-box .ib-row{display:flex;justify-content:space-between;margin-bottom:5px;gap:16px}
#info-box .ib-val{color:#fff;font-weight:600}
#info-hint{
  position:fixed;top:14px;right:14px;
  background:rgba(10,10,25,0.88);
  border:1px solid rgba(255,255,255,0.1);
  border-radius:10px;padding:10px 16px;
  color:#888;font-size:12px;z-index:2000;
}

.leaflet-container{background:#0e0e1a}
</style>
</head>
<body>
<div id="map"></div>

<div id="panel">
  <h3>서울 법정동별 숙박밀집도</h3>
  <div class="stat"><span>분석 법정동</span><span class="val">260개</span></div>
  <div class="stat"><span>숙박시설 있는 동</span><span class="val">223개</span></div>
  <div class="stat"><span>총 숙박시설</span><span class="val">4,246개</span></div>
  <select id="metric-select">
    <option value="per_ha">시설수 / ha (공간 밀집)</option>
    <option value="count">숙박시설 수 (절대 수)</option>
    <option value="fa_ha">숙박 연면적 / ha (규모 밀집)</option>
    <option value="ratio">전체 건물 중 숙박 비율 (%)</option>
  </select>
  <label class="toggle" style="margin-top:10px;display:flex;align-items:center;gap:8px;cursor:pointer;color:#ccc;font-size:12px">
    <input type="checkbox" id="chk-dots" checked style="accent-color:#ffb432;cursor:pointer">
    숙소 위치 표시
  </label>
</div>

<div id="legend">
  <div id="legend-title">시설수 / ha</div>
  <div id="legend-items"></div>
  <hr style="border-color:rgba(255,255,255,0.1);margin:8px 0">
  <div class="lg-row"><div class="lg-cell" style="background:#3a3a4a;border:1px solid rgba(255,255,255,0.3)"></div><span>숙박 없는 지역</span></div>
</div>

<div id="info-hint">← 법정동을 클릭하세요</div>
<div id="info-box">
  <div class="ib-name" id="ib-name"></div>
  <div class="ib-row"><span>구</span><span class="ib-val" id="ib-gu"></span></div>
  <div class="ib-row"><span>숙박시설 수</span><span class="ib-val" id="ib-count"></span></div>
  <div class="ib-row"><span>면적</span><span class="ib-val" id="ib-area"></span></div>
  <div class="ib-row"><span>시설수 / ha</span><span class="ib-val" id="ib-per-ha"></span></div>
  <div class="ib-row"><span>연면적 / ha</span><span class="ib-val" id="ib-fa-ha"></span></div>
  <div class="ib-row"><span>숙박 건물 비율</span><span class="ib-val" id="ib-ratio"></span></div>
  <div class="ib-row"><span>숙박 연면적 합</span><span class="ib-val" id="ib-total-fa"></span></div>
</div>

<style>
.leaflet-popup-content-wrapper{background:rgba(10,10,25,0.96);color:#eee;border-radius:8px;border:1px solid rgba(255,180,50,0.6);box-shadow:0 4px 20px rgba(0,0,0,0.7)}
.leaflet-popup-tip{background:rgba(10,10,25,0.96)}
.leaflet-popup-content{margin:10px 14px;font-family:'Segoe UI',sans-serif}
.sp-name{font-size:13px;font-weight:700;color:#ffb432;margin-bottom:4px}
.sp-addr{font-size:11px;color:#999;margin-bottom:4px}
.sp-meta{font-size:11px;color:#bbb}
</style>
""")

# ─── 3-2. JavaScript 데이터 주입 ───────────────────────────────
# Python 측 변수를 JS 상수로 그대로 삽입한다.
parts.append("<script>\n")
parts.append("const GEOJSON = " + geojson_str + ";\n")
parts.append("const PLACES  = " + places_json + ";\n")

# ─── 3-3. JS 본문 ──────────────────────────────────────────────
# 핵심 로직 요약:
#   - METRICS: 4개 지표별 (단위, 구간(breaks), 범례 라벨, 색 팔레트)
#   - getColor: 값에 맞는 색상 산출
#   - buildLayer: 현재 지표 기준으로 GeoJSON 레이어 (재)생성 — 마우스/클릭 핸들러 부착
#   - buildLegend: 지표 변경 시 범례 갱신
#   - 줌 변동 시 마커 반경 자동 조정
parts.append("""
const map = L.map('map', {center:[37.5530,126.9740], zoom:12, zoomControl:true});
L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',{
  attribution:'© CARTO © OSM', subdomains:'abcd', maxZoom:20
}).addTo(map);

// ── 색상 설정 ─────────────────────────────────────────────
const METRICS = {
  per_ha: {
    label: '시설수 / ha',
    unit:  '개/ha',
    breaks:[0, 0.1, 0.5, 1.0, 2.0, 3.0],
    labels:['0', '0.1~0.5', '0.5~1.0', '1.0~2.0', '2.0~3.0', '3.0+'],
    colors:['#1a1a3e','#2a4090','#3070c0','#60b040','#f0a020','#e03010']
  },
  count: {
    label: '숙박시설 수',
    unit:  '개',
    breaks:[0, 1, 10, 30, 60, 100],
    labels:['0', '1~10', '10~30', '30~60', '60~100', '100+'],
    colors:['#1a1a3e','#2a4090','#3070c0','#60b040','#f0a020','#e03010']
  },
  fa_ha: {
    label: '숙박 연면적 / ha',
    unit:  '㎡/ha',
    breaks:[0, 100, 500, 1000, 3000, 6000],
    labels:['0', '100~500', '500~1k', '1k~3k', '3k~6k', '6k+'],
    colors:['#1a1a3e','#2a4090','#3070c0','#60b040','#f0a020','#e03010']
  },
  ratio: {
    label: '숙박 건물 비율',
    unit:  '%',
    breaks:[0, 1, 10, 50, 100, 200],
    labels:['0%', '1~10%', '10~50%', '50~100%', '100~200%', '200%+'],
    colors:['#1a1a3e','#2a4090','#3070c0','#60b040','#f0a020','#e03010']
  }
};

function getColor(val, metric) {
  var m = METRICS[metric];
  for (var i = m.breaks.length - 1; i >= 0; i--) {
    if (val > m.breaks[i]) return m.colors[Math.min(i+1, m.colors.length-1)];
  }
  return m.colors[0];
}

function style(feature, metric) {
  var val = feature.properties[metric];
  return {
    fillColor:   val > 0 ? getColor(val, metric) : '#3a3a4a',
    fillOpacity: val > 0 ? 0.72 : 0.55,
    color:       'rgba(255,255,255,0.45)',
    weight:      0.8
  };
}

// ── GeoJSON 레이어 ─────────────────────────────────────────
var currentMetric = 'per_ha';
var dongLayer;

function buildLayer(metric) {
  if (dongLayer) map.removeLayer(dongLayer);
  dongLayer = L.geoJSON(GEOJSON, {
    style: function(f) { return style(f, metric); },
    onEachFeature: function(feature, layer) {
      var p = feature.properties;
      layer.on({
        mouseover: function(e) {
          e.target.setStyle({weight:2, color:'rgba(255,255,255,0.8)', fillOpacity: p.count>0 ? 0.88 : 0.70});
        },
        mouseout: function(e) {
          dongLayer.resetStyle(e.target);
        },
        click: function(e) {
          document.getElementById('info-hint').style.display = 'none';
          var box = document.getElementById('info-box');
          box.style.display = 'block';
          document.getElementById('ib-name').textContent    = p.name;
          document.getElementById('ib-gu').textContent      = p.gu;
          document.getElementById('ib-count').textContent   = p.count.toLocaleString() + ' 개';
          document.getElementById('ib-area').textContent    = p.area_ha.toFixed(2) + ' ha';
          document.getElementById('ib-per-ha').textContent  = p.per_ha.toFixed(3) + ' 개/ha';
          document.getElementById('ib-fa-ha').textContent   = p.fa_ha.toLocaleString() + ' ㎡/ha';
          document.getElementById('ib-ratio').textContent   = p.ratio.toFixed(1) + ' %';
          document.getElementById('ib-total-fa').textContent= Math.round(p.total_fa).toLocaleString() + ' ㎡';
        }
      });
    }
  }).addTo(map);
}

function buildLegend(metric) {
  var m = METRICS[metric];
  document.getElementById('legend-title').textContent = m.label;
  var html = '';
  m.labels.forEach(function(lbl, i) {
    html += '<div class="lg-row">'
          + '<div class="lg-box" style="background:' + m.colors[i] + '"></div>'
          + '<span>' + lbl + ' ' + m.unit + '</span></div>';
  });
  document.getElementById('legend-items').innerHTML = html;
}

buildLayer(currentMetric);
buildLegend(currentMetric);

document.getElementById('metric-select').addEventListener('change', function() {
  currentMetric = this.value;
  buildLayer(currentMetric);
  buildLegend(currentMetric);
  document.getElementById('info-box').style.display = 'none';
  document.getElementById('info-hint').style.display = 'block';
});

// ── 숙소 포인트 레이어 ────────────────────────────────────
var dotLayer = L.layerGroup();

PLACES.forEach(function(p) {
  var popup = '<div class="sp-name">' + p.name + '</div>'
            + '<div class="sp-addr">' + p.addr + '</div>'
            + '<div class="sp-meta">지상 ' + p.floors + '층 · 연면적 ' + Math.round(p.side_m * p.side_m).toLocaleString() + ' ㎡</div>';
  L.circleMarker([p.lat, p.lng], {
    radius:      4,
    color:       '#ffffff',
    weight:      1.2,
    fillColor:   '#ffb432',
    fillOpacity: 0.85
  }).bindPopup(popup, {maxWidth: 260}).addTo(dotLayer);
});

dotLayer.addTo(map);

// 줌에 따라 반경 조정
map.on('zoomend', function() {
  var z = map.getZoom();
  var r = z <= 12 ? 3 : z <= 14 ? 4 : z <= 16 ? 5 : 6;
  dotLayer.eachLayer(function(l) { l.setRadius(r); });
});

document.getElementById('chk-dots').addEventListener('change', function() {
  this.checked ? map.addLayer(dotLayer) : map.removeLayer(dotLayer);
});

// 법정동 레이어가 항상 포인트 아래 오도록
map.on('overlayadd', function() {
  if (dongLayer) dongLayer.bringToBack();
});
</script>
</body>
</html>
""")

# ─── 4. HTML 파일 저장 ───────────────────────────────────────────
out = "c:/Users/USER/Documents/GitHub/기말공모전/NJT-PJT/법정동_숙박밀집도.html"
with open(out, "w", encoding="utf-8") as f:
    f.write("".join(parts))
# 결과 파일 크기를 KB 단위로 출력
print("Done:", os.path.getsize(out) // 1024, "KB")
