"""
[파일 설명]
서울 집계구별 숙박시설 밀집도와 화재위험도를 종합 대시보드 지도로 시각화하는 HTML을 생성하는 스크립트.

주요 역할:
  Leaflet 라이브러리로 집계구 경계를 3가지 지표로 색상 표현한다:
    - 숙박시설 수 (절대량)
    - 전체 건물 중 숙박 비율(%)
    - 시설수/ha (공간 밀도)
  집계구 클릭 시 상세 패널에 화재위험도 점수와 지표 설명이 표시된다.
  소방서/안전센터 위치도 마커로 함께 표시한다.

입력: data/oa_density.json       (집계구별 분석 데이터)
      data/map_data.json          (개별 숙박시설 위치)
      data/firestation_data.json  (소방서·안전센터 위치)
출력: 집계구_숙박밀집도.html       (Leaflet 집계구 대시보드 맵)
"""

import sys, json, os
sys.stdout.reconfigure(encoding='utf-8')  # 한글 출력 설정

# ─── 1. 집계구 데이터 로드 및 분리 ─────────────────────────────
with open('c:/Users/USER/Documents/GitHub/기말공모전/NJT-PJT/data/oa_density.json', encoding='utf-8') as f:
    raw = json.load(f)

# 숙박시설이 있는 집계구(filled)와 없는 집계구(empty)로 분리
# filled: 색상과 클릭 이벤트 적용, empty: 회색 경계선만 표시
filled = [f for f in raw['features'] if f['properties']['count'] > 0]
empty  = [f for f in raw['features'] if f['properties']['count'] == 0]

# JavaScript에 직접 삽입할 GeoJSON 문자열로 변환
filled_json = json.dumps({'type':'FeatureCollection','features':filled}, ensure_ascii=False)
empty_json  = json.dumps({'type':'FeatureCollection','features':empty},  ensure_ascii=False)

# ─── 2. 개별 숙박시설 위치 로드 ─────────────────────────────────
with open('c:/Users/USER/Documents/GitHub/기말공모전/NJT-PJT/data/map_data.json', encoding='utf-8') as f:
    places_json = json.dumps(json.load(f)['places'], ensure_ascii=False)

# ─── 3. 소방서/안전센터 데이터 로드 ─────────────────────────────
with open('c:/Users/USER/Documents/GitHub/기말공모전/NJT-PJT/data/firestation_data.json', encoding='utf-8') as f:
    stations_json = json.dumps(json.load(f), ensure_ascii=False)

print(f"채워진 집계구: {len(filled)}, 빈 집계구: {len(empty)}")

out_html = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>서울 집계구별 숙박밀집도</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
html,body{width:100%;height:100%;background:#0e0e1a;font-family:'Segoe UI',sans-serif}
#map{width:100%;height:100vh}

/* ─── 왼쪽 패널 ─── */
#panel{position:fixed;top:14px;left:14px;background:rgba(10,10,25,0.95);border:1px solid rgba(255,255,255,0.15);border-radius:12px;padding:14px 16px;color:#ddd;font-size:13px;z-index:2000;min-width:210px}
#panel h3{color:#ffb432;font-size:15px;margin-bottom:10px;font-weight:700;letter-spacing:0.5px}
#panel .stat{display:flex;justify-content:space-between;margin-bottom:5px;gap:20px}
#panel .val{color:#fff;font-weight:600}
#panel select{margin-top:10px;width:100%;padding:5px 8px;background:rgba(255,255,255,0.08);color:#fff;border:1px solid rgba(255,255,255,0.2);border-radius:6px;font-size:12px;cursor:pointer}
.toggle{display:flex;align-items:center;gap:8px;margin-top:7px;cursor:pointer;color:#ccc;font-size:12px}
.toggle input{accent-color:#ffb432;cursor:pointer}
.toggle-dot{width:10px;height:10px;border-radius:50%;flex-shrink:0}

/* ─── 범례 ─── */
#legend{position:fixed;bottom:22px;right:14px;background:rgba(10,10,25,0.95);border:1px solid rgba(255,255,255,0.15);border-radius:12px;padding:14px 18px;color:#ccc;font-size:12px;z-index:2000;min-width:270px;max-width:310px}
#legend-title{color:#ffb432;font-size:13px;margin-bottom:2px;font-weight:700}
#legend-sub{color:#888;font-size:10px;margin-bottom:4px}
#legend-how{font-size:10px;color:#555;margin-bottom:10px;line-height:1.5;padding:5px 7px;background:rgba(255,255,255,0.04);border-radius:5px;border-left:2px solid #ffb43255}
.lg-row{display:flex;align-items:center;margin-bottom:5px;gap:8px}
.lg-bar{width:26px;height:13px;border-radius:3px;flex-shrink:0;border:1px solid rgba(255,255,255,0.12)}
.lg-label{font-size:11px;color:#ddd;min-width:60px}
.lg-tag{font-size:10px;color:#666;flex:1}
.lg-cnt{font-size:10px;color:#555;text-align:right;white-space:nowrap}
.lg-divider{border:none;border-top:1px solid rgba(255,255,255,0.08);margin:7px 0}
#legend-stat{margin-top:8px;padding-top:7px;border-top:1px solid rgba(255,255,255,0.08);font-size:10px;color:#666;line-height:1.8}

/* ─── 범례 하단 마커 범례 ─── */
#legend-markers{margin-top:8px;padding-top:7px;border-top:1px solid rgba(255,255,255,0.08)}
.mk-row{display:flex;align-items:center;gap:8px;margin-bottom:4px;font-size:11px;color:#ccc}
.mk-dot{width:12px;height:12px;border-radius:50%;flex-shrink:0}

/* ─── 힌트 / 정보박스 ─── */
#hint{position:fixed;top:14px;right:14px;background:rgba(10,10,25,0.88);border:1px solid rgba(255,255,255,0.1);border-radius:10px;padding:10px 16px;color:#888;font-size:12px;z-index:2000}
#info-box{position:fixed;top:14px;right:14px;background:rgba(10,10,25,0.96);border:1px solid rgba(255,255,255,0.15);border-radius:12px;padding:14px 16px;color:#ddd;font-size:13px;z-index:2000;min-width:260px;max-width:300px;display:none}

/* info-box 내부 */
#info-box .ib-header{margin-bottom:8px}
#info-box .ib-title{font-size:11px;color:#888;margin-bottom:2px}
#info-box .ib-name{font-size:15px;font-weight:700;color:#ffb432}
#info-box .ib-badge{display:inline-block;padding:2px 9px;border-radius:4px;font-size:11px;font-weight:700;margin-top:5px}
#info-box .ib-section{font-size:10px;color:#555;text-transform:uppercase;letter-spacing:1px;margin:10px 0 5px}
#info-box .ib-row{display:flex;justify-content:space-between;margin-bottom:4px;gap:12px;font-size:12px}
#info-box .ib-val{color:#fff;font-weight:600}

/* 화재위험도 박스 */
#info-box .fire-box{background:rgba(255,80,30,0.07);border:1px solid rgba(255,80,30,0.25);border-radius:8px;padding:10px 12px;margin-top:10px}
#info-box .fire-title{font-size:11px;color:#ff6040;font-weight:700;margin-bottom:8px;display:flex;justify-content:space-between;align-items:center}
#info-box .fire-score-num{font-size:22px;font-weight:700}
#info-box .fire-row{display:flex;justify-content:space-between;margin-bottom:3px;font-size:11px;color:#bbb}
#info-box .fire-val{color:#eee;font-weight:600}
#info-box .fire-bar-bg{background:rgba(255,255,255,0.08);height:6px;border-radius:3px;margin-top:6px;overflow:hidden}
#info-box .fire-bar-fill{height:100%;border-radius:3px;transition:width 0.3s}
#info-box .fire-level{font-size:10px;font-weight:700;padding:1px 6px;border-radius:3px;margin-left:6px}

#info-box .ib-interpret{margin-top:8px;padding-top:8px;border-top:1px solid rgba(255,255,255,0.08);font-size:11px;line-height:1.6;color:#999}

/* 팝업 */
.leaflet-popup-content-wrapper{background:rgba(10,10,25,0.96);color:#eee;border-radius:8px;border:1px solid rgba(255,180,50,0.5);box-shadow:0 4px 20px rgba(0,0,0,0.7)}
.leaflet-popup-tip{background:rgba(10,10,25,0.96)}
.leaflet-popup-content{margin:10px 14px;font-family:'Segoe UI',sans-serif}
.sp-name{font-size:13px;font-weight:700;color:#ffb432;margin-bottom:3px}
.sp-addr{font-size:11px;color:#999}
.st-name{font-size:13px;font-weight:700;margin-bottom:2px}
.st-type-소방서{color:#ff6b6b}
.st-type-안전센터{color:#ff9f43}
.st-meta{font-size:11px;color:#888}
</style>
</head>
<body>
<div id="map"></div>

<!-- 왼쪽 컨트롤 패널 -->
<div id="panel">
  <h3>서울 집계구별 숙박밀집도</h3>
  <div class="stat"><span>전체 집계구</span><span class="val">19,097개</span></div>
  <div class="stat"><span>숙박 있는 집계구</span><span class="val">1,362개</span></div>
  <div class="stat"><span>총 숙박시설</span><span class="val">4,246개</span></div>
  <div class="stat"><span>현재 줌</span><span class="val" id="zoom-val">-</span></div>
  <select id="metric-select">
    <option value="count">숙박시설 수 (절대량)</option>
    <option value="ratio">전체 건물 중 숙박 비율 (%)</option>
    <option value="per_ha">시설수 / ha (공간 밀도)</option>
  </select>
  <label class="toggle"><input type="checkbox" id="chk-dots"    checked>
    <span class="toggle-dot" style="background:#ffb432;border:1px solid #fff"></span>숙소 위치</label>
  <label class="toggle"><input type="checkbox" id="chk-empty"   checked>
    <span class="toggle-dot" style="background:#2a2a3a;border:1px solid rgba(255,255,255,0.3)"></span>빈 집계구 경계</label>
  <label class="toggle"><input type="checkbox" id="chk-station" checked>
    <span class="toggle-dot" style="background:#ff6b6b;border:1px solid #fff"></span>소방서 / 안전센터</label>
</div>

<!-- 범례 -->
<div id="legend">
  <div id="legend-title">숙박시설 수</div>
  <div id="legend-sub"></div>
  <div id="legend-how"></div>
  <div id="legend-items"></div>
  <div class="lg-divider"></div>
  <div class="lg-row">
    <div class="lg-bar" style="background:#2a2a3a;border-color:rgba(255,255,255,0.2)"></div>
    <span class="lg-label">숙박 없음</span>
    <span class="lg-tag">일반 주거·업무지역</span>
    <span class="lg-cnt" id="empty-cnt">17,735구역</span>
  </div>
  <div id="legend-stat"></div>
  <!-- 마커 범례 -->
  <div id="legend-markers">
    <div class="mk-row"><div class="mk-dot" style="background:#ffb432;border:1.5px solid #fff"></div>숙박시설 위치</div>
    <div class="mk-row"><div class="mk-dot" style="background:#ff4444;border:2px solid #fff"></div>소방서 (본서)</div>
    <div class="mk-row"><div class="mk-dot" style="background:#ff9933;border:1.5px solid #fff;transform:rotate(45deg);border-radius:2px"></div>119 안전센터</div>
  </div>
</div>

<div id="hint">← 집계구를 클릭하세요</div>

<!-- 정보 박스 (클릭 시) -->
<div id="info-box">
  <div class="ib-header">
    <div class="ib-title" id="ib-gu-label">구 이름</div>
    <div class="ib-name" id="ib-name">-</div>
    <div id="ib-badge" class="ib-badge"></div>
  </div>

  <div class="ib-section">숙박 현황</div>
  <div class="ib-row"><span>숙박시설 수</span><span class="ib-val" id="ib-count">-</span></div>
  <div class="ib-row"><span>전체 건물 수</span><span class="ib-val" id="ib-total">-</span></div>
  <div class="ib-row"><span>숙박 비율</span><span class="ib-val" id="ib-ratio">-</span></div>
  <div class="ib-row"><span>시설수 / ha</span><span class="ib-val" id="ib-per-ha">-</span></div>
  <div class="ib-row"><span>집계구 면적</span><span class="ib-val" id="ib-area">-</span></div>
  <div class="ib-row"><span>숙박 연면적 합</span><span class="ib-val" id="ib-fa">-</span></div>

  <!-- 화재위험도 박스 -->
  <div class="fire-box" id="fire-box">
    <div class="fire-title">
      🔥 화재위험도
      <span><span class="fire-score-num" id="fire-score-num">-</span><span style="font-size:12px;color:#aaa">/ 100</span></span>
    </div>
    <div class="fire-bar-bg"><div class="fire-bar-fill" id="fire-bar-fill" style="width:0%"></div></div>
    <div style="margin-bottom:8px"></div>
    <div class="fire-row"><span>노후도 (평균 건축연령)</span><span class="fire-val" id="fire-age">-</span></div>
    <div class="fire-row"><span>건폐율 (평면 밀집)</span><span class="fire-val" id="fire-gpye">-</span></div>
    <div class="fire-row"><span>용적률 (입체 밀집)</span><span class="fire-val" id="fire-yong">-</span></div>
    <div class="fire-row"><span>평균 층수 (피난 난이도)</span><span class="fire-val" id="fire-floor">-</span></div>
    <div style="font-size:9px;color:#555;margin-top:6px">* 숙박시설 데이터 기준. 노후도30+건폐율25+용적률25+층수20</div>
  </div>

  <div class="ib-interpret" id="ib-interpret"></div>
</div>
"""

script_part = """
<script>
const FILLED   = """ + filled_json + """;
const EMPTY    = """ + empty_json  + """;
const PLACES   = """ + places_json + """;
const STATIONS = """ + stations_json + """;

const map = L.map('map', {center:[37.5530,126.9740], zoom:12,
  preferCanvas:true, renderer:L.canvas()});
L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',{
  attribution:'© CARTO © OSM', subdomains:'abcd', maxZoom:20
}).addTo(map);

// ── 메트릭 정의 ──────────────────────────────────────────────
const METRICS = {
  count: {
    label:'숙박시설 수', unit:'개',
    desc:'집계구 내 숙박시설 개수 (절대량)',
    how:'숙박업소가 몇 개나 있는지. 숫자가 클수록 숙박업 밀집 지역.',
    breaks:[0,1,3,6,10,20],
    labels:['1개','2~3개','4~6개','7~10개','11~20개','21개+'],
    tags:['희소','소수','중간밀집','높은밀집','매우높음','극밀집'],
    colors:['#2a4090','#3070c0','#60b040','#f0a020','#e03010','#9b0000'],
    stat:'최소 1개 · 최대 66개 · 평균 3.1개'
  },
  ratio: {
    label:'숙박 건물 비율', unit:'%',
    desc:'전체 건물 중 숙박시설이 차지하는 비율',
    how:'동네 성격 지표. 주거지 보통 1% 이하 / 관광특화지역 20%+',
    breaks:[0,1,5,10,20,50],
    labels:['0~1%','1~5%','5~10%','10~20%','20~50%','50%+'],
    tags:['주거·업무','소수혼재','혼재','숙박특화','고농도','전용구역'],
    colors:['#2a4090','#3070c0','#60b040','#f0a020','#e03010','#9b0000'],
    stat:'최소 0.1% · 최대 400% (복합건물 포함)'
  },
  per_ha: {
    label:'시설수 / ha', unit:'개/ha',
    desc:'면적 1ha(100×100m)당 숙박시설 수',
    how:'면적 보정 밀도. 좁은 구역에 밀집될수록 높음. (명동 골목 등)',
    breaks:[0,0.5,2,5,8,11],
    labels:['0~0.5','0.5~2','2~5','5~8','8~11','11+'],
    tags:['저밀도','약간분포','중밀도','고밀도','초고밀도','극밀도'],
    colors:['#2a4090','#3070c0','#60b040','#f0a020','#e03010','#9b0000'],
    stat:'최소 0.0 · 최대 11.9개/ha'
  }
};
var currentMetric = 'count';

// ── 화재위험도 등급 ────────────────────────────────────────────
const FIRE_LEVELS = [
  {max:25, label:'낮음',    color:'#51cf66', bg:'rgba(81,207,102,0.15)'},
  {max:45, label:'보통',    color:'#fcc419', bg:'rgba(252,196,25,0.15)'},
  {max:65, label:'높음',    color:'#ff922b', bg:'rgba(255,146,43,0.15)'},
  {max:100,label:'매우높음',color:'#f03e3e', bg:'rgba(240,62,62,0.15)'}
];
function getFireLevel(score) {
  return FIRE_LEVELS.find(l => score <= l.max) || FIRE_LEVELS[FIRE_LEVELS.length-1];
}

// ── 집계구 성격 해석 ────────────────────────────────────────────
function interpretOA(count, total, ratio, per_ha) {
  var badge, badgeColor, text;
  if      (ratio >= 50) { badge='숙박 전용 구역'; badgeColor='#9b0000';
    text='전체 건물의 절반 이상이 숙박시설. 숙박업이 집중된 특화 구역입니다.'; }
  else if (ratio >= 20) { badge='고농도 숙박 구역'; badgeColor='#e03010';
    text='건물 5개 중 1개 이상이 숙박시설. 관광·상업 인접 지역에서 나타납니다.'; }
  else if (ratio >= 10) { badge='숙박 특화 구역'; badgeColor='#f0a020';
    text='숙박 비율 10%+. 번화가·역세권에서 자주 나타나는 패턴입니다.'; }
  else if (ratio >= 5)  { badge='혼재 지역'; badgeColor='#60b040';
    text='주거·상업시설과 숙박시설이 섞여 있는 구역입니다.'; }
  else if (ratio >= 1)  { badge='소수 혼재'; badgeColor='#3070c0';
    text='일반 주거·업무지역에 숙박시설이 소수 들어와 있습니다.'; }
  else                  { badge='주거·업무 지역'; badgeColor='#2a4090';
    text='숙박 비율 1% 미만. 주거·업무 용도가 지배적인 구역입니다.'; }
  if      (per_ha >= 8) text += ' 면적 대비 밀도가 극히 높아 골목에 밀집된 형태입니다.';
  else if (per_ha >= 3) text += ' 면적 대비 밀도도 높습니다.';
  else if (total <= 20) text += ' 건물 수가 적은 소형 구역입니다.';
  return {badge, color: badgeColor, text};
}

// ── 색상 / 스타일 ────────────────────────────────────────────
function getColor(val, m) {
  var b = METRICS[m].breaks, c = METRICS[m].colors;
  for (var i=b.length-1; i>=0; i--) if(val>b[i]) return c[Math.min(i+1,c.length-1)];
  return c[0];
}
function styleF(f) {
  return {fillColor:getColor(f.properties[currentMetric], currentMetric),
          fillOpacity:0.75, color:'rgba(255,255,255,0.3)', weight:0.5};
}
function styleE() {
  return {fillColor:'#2a2a3a', fillOpacity:0.55,
          color:'rgba(255,255,255,0.15)', weight:0.3};
}

// ── 레이어 구성 ──────────────────────────────────────────────
var filledLayer, emptyLayer;

function buildLayers(metric) {
  if (filledLayer) map.removeLayer(filledLayer);
  if (emptyLayer)  map.removeLayer(emptyLayer);
  emptyLayer = L.geoJSON(EMPTY, {style:styleE, interactive:false});
  if (document.getElementById('chk-empty').checked) emptyLayer.addTo(map);

  filledLayer = L.geoJSON(FILLED, {
    style: styleF,
    onEachFeature: function(feat, layer) {
      var p = feat.properties;
      layer.on({
        mouseover: function(e) {
          e.target.setStyle({weight:1.5, color:'rgba(255,255,255,0.85)', fillOpacity:0.92});
        },
        mouseout: function(e) { filledLayer.resetStyle(e.target); },
        click: function() {
          document.getElementById('hint').style.display = 'none';
          document.getElementById('info-box').style.display = 'block';

          // ① 헤더: 구 이름 + 집계구 번호
          var guName = p.gu_name || '알수없음';
          var oaNo   = p.oa_no  || p.id.slice(8);
          document.getElementById('ib-gu-label').textContent = guName;
          document.getElementById('ib-name').textContent = '집계구 #' + oaNo;

          // ② 숙박 현황
          document.getElementById('ib-count').textContent  = p.count + ' 개';
          document.getElementById('ib-total').textContent  = (p.total||0).toLocaleString() + ' 개';
          document.getElementById('ib-ratio').textContent  = p.ratio.toFixed(1) + ' %';
          document.getElementById('ib-per-ha').textContent = p.per_ha.toFixed(2) + ' 개/ha';
          document.getElementById('ib-area').textContent   = p.area_ha.toFixed(3) + ' ha';
          document.getElementById('ib-fa').textContent     = Math.round(p.fa).toLocaleString() + ' ㎡';

          // ③ 배지
          var interp = interpretOA(p.count, p.total, p.ratio, p.per_ha);
          var badge = document.getElementById('ib-badge');
          badge.textContent = interp.badge;
          badge.style.cssText = 'background:'+interp.color+'22;color:'+interp.color+
            ';border:1px solid '+interp.color+'66;display:inline-block;padding:2px 9px;'+
            'border-radius:4px;font-size:11px;font-weight:700;margin-top:5px';

          // ④ 화재위험도 박스
          var fs = p.fire_score || 0;
          var lv = getFireLevel(fs);
          document.getElementById('fire-score-num').textContent = fs.toFixed(0);
          document.getElementById('fire-score-num').style.color = lv.color;
          var bar = document.getElementById('fire-bar-fill');
          bar.style.width = fs + '%';
          bar.style.background = 'linear-gradient(90deg,'+lv.color+'88,'+lv.color+')';
          document.getElementById('fire-box').style.borderColor = lv.color+'44';

          document.getElementById('fire-age').textContent  =
            p.avg_age  != null ? p.avg_age.toFixed(1)  + '년  (최대 ' + (p.max_age||'-') + '년)' : '데이터 없음';
          document.getElementById('fire-gpye').textContent =
            p.avg_geonpye != null ? p.avg_geonpye.toFixed(1) + ' %' : '데이터 없음';
          document.getElementById('fire-yong').textContent =
            p.avg_yongjuk != null ? p.avg_yongjuk.toFixed(0)+ ' %' : '데이터 없음';
          document.getElementById('fire-floor').textContent=
            p.avg_floors  != null ? p.avg_floors.toFixed(1)  + ' 층' : '데이터 없음';

          // ⑤ 해석 텍스트
          document.getElementById('ib-interpret').textContent = interp.text;
        }
      });
    }
  }).addTo(map);
}

// ── 범례 ────────────────────────────────────────────────────
function buildLegend(metric) {
  var m = METRICS[metric];
  document.getElementById('legend-title').textContent = m.label;
  document.getElementById('legend-sub').textContent   = m.desc;
  document.getElementById('legend-how').innerHTML     =
    '<b style="color:#ffb432cc">▶ 읽는 법:</b> ' + m.how;

  var dist = new Array(m.colors.length).fill(0);
  FILLED.features.forEach(function(f){
    var v = f.properties[metric], placed = false;
    for(var i=m.breaks.length-1; i>=0; i--){
      if(v > m.breaks[i]){ dist[Math.min(i+1,m.colors.length-1)]++; placed=true; break; }
    }
    if(!placed) dist[0]++;
  });

  var html = '';
  m.colors.forEach(function(c,i){
    html += '<div class="lg-row">'
          + '<div class="lg-bar" style="background:'+c+'"></div>'
          + '<span class="lg-label">'+m.labels[i]+'</span>'
          + '<span class="lg-tag">'+m.tags[i]+'</span>'
          + '<span class="lg-cnt">'+dist[i].toLocaleString()+'구역</span>'
          + '</div>';
  });
  document.getElementById('legend-items').innerHTML = html;
  document.getElementById('legend-stat').innerHTML  =
    '▪ ' + m.stat + '<br>▪ 집계구 총 19,097개  숙박있는구역 1,362개';
}

buildLayers(currentMetric);
buildLegend(currentMetric);

// ── 숙소 포인트 레이어 ──────────────────────────────────────
var dotLayer = L.layerGroup();
PLACES.forEach(function(p) {
  L.circleMarker([p.lat,p.lng],{
    radius:3, color:'#fff', weight:1.2, fillColor:'#ffb432', fillOpacity:0.85
  }).bindPopup('<div class="sp-name">'+p.name+'</div><div class="sp-addr">'+p.addr+'</div>',
               {maxWidth:260}).addTo(dotLayer);
});
dotLayer.addTo(map);

// ── 소방서 / 안전센터 레이어 ────────────────────────────────
var stationLayer = L.layerGroup();
STATIONS.forEach(function(s) {
  var isMain = (s.type === '소방서');
  var color  = isMain ? '#ff4444' : '#ff9933';
  var size   = isMain ? 8 : 5;
  var marker = isMain
    ? L.circleMarker([s.lat, s.lng], {
        radius:size, color:'#fff', weight:1.5, fillColor:color, fillOpacity:0.9
      })
    : L.polygon([
        [s.lat + 0.0004, s.lng],
        [s.lat, s.lng + 0.0004],
        [s.lat - 0.0004, s.lng],
        [s.lat, s.lng - 0.0004]
      ], {color:'#fff', weight:1, fillColor:color, fillOpacity:0.88});

  marker.bindPopup(
    '<div class="st-name st-type-'+s.type+'">'+s.name+'</div>'+
    '<div class="st-meta">'+s.type+' · 출동건수 '+s.count+'건</div>',
    {maxWidth:200}
  ).addTo(stationLayer);
});
stationLayer.addTo(map);

// ── 줌 연동 ────────────────────────────────────────────────
map.on('zoomend', function(){
  var z = map.getZoom();
  document.getElementById('zoom-val').textContent = z;
  var r = z<=12?3:z<=14?4:z<=16?5:6;
  dotLayer.eachLayer(function(l){ if(l.setRadius) l.setRadius(r); });
});
document.getElementById('zoom-val').textContent = map.getZoom();

// ── 체크박스 ────────────────────────────────────────────────
document.getElementById('metric-select').addEventListener('change', function(){
  currentMetric = this.value;
  buildLayers(currentMetric);
  buildLegend(currentMetric);
  document.getElementById('info-box').style.display='none';
  document.getElementById('hint').style.display='block';
});
document.getElementById('chk-dots').addEventListener('change', function(){
  this.checked ? map.addLayer(dotLayer) : map.removeLayer(dotLayer);
});
document.getElementById('chk-empty').addEventListener('change', function(){
  this.checked ? map.addLayer(emptyLayer) : map.removeLayer(emptyLayer);
});
document.getElementById('chk-station').addEventListener('change', function(){
  this.checked ? map.addLayer(stationLayer) : map.removeLayer(stationLayer);
});
</script></body></html>
"""

# ─── 4. HTML 파일 저장 ───────────────────────────────────────────
# out_html: HTML 구조(head, body, 스타일, 패널, 범례 등)
# script_part: JavaScript 데이터와 지도 로직
with open('c:/Users/USER/Documents/GitHub/기말공모전/NJT-PJT/집계구_숙박밀집도.html', 'w', encoding='utf-8') as f:
    f.write(out_html + script_part)

out = 'c:/Users/USER/Documents/GitHub/기말공모전/NJT-PJT/집계구_숙박밀집도.html'
print(f"Done: {os.path.getsize(out)//1024} KB")
