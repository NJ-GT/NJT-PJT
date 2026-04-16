import sys, json, os
sys.stdout.reconfigure(encoding='utf-8')

with open('c:/Users/USER/Documents/GitHub/기말공모전/NJT-PJT/data/oa_density.json', encoding='utf-8') as f:
    raw = json.load(f)

filled = [f for f in raw['features'] if f['properties']['count'] > 0]
empty  = [f for f in raw['features'] if f['properties']['count'] == 0]

filled_json = json.dumps({'type':'FeatureCollection','features':filled}, ensure_ascii=False)
empty_json  = json.dumps({'type':'FeatureCollection','features':empty},  ensure_ascii=False)

with open('c:/Users/USER/Documents/GitHub/기말공모전/NJT-PJT/data/firestation_data.json', encoding='utf-8') as f:
    stations_json = json.dumps(json.load(f), ensure_ascii=False)

print(f"채워진 집계구: {len(filled)}, 빈 집계구: {len(empty)}")

html = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>서울 집계구별 숙박시설 3D</title>
<script src="https://unpkg.com/deck.gl@8.9.35/dist.min.js"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
html,body{width:100%;height:100%;background:#0e0e1a;font-family:'Segoe UI',sans-serif;overflow:hidden}
#deckgl-wrapper{width:100%;height:100vh}

#panel{position:fixed;top:16px;left:16px;background:rgba(8,8,20,0.94);border:1px solid rgba(255,255,255,0.15);border-radius:12px;padding:14px 16px;color:#ddd;font-size:13px;z-index:1000;min-width:220px}
#panel h3{color:#ffb432;font-size:14px;font-weight:700;margin-bottom:10px}
.stat{display:flex;justify-content:space-between;margin-bottom:4px;font-size:12px}
.val{color:#fff;font-weight:600}
.divider{border:none;border-top:1px solid rgba(255,255,255,0.08);margin:9px 0}
.mode-btns{display:flex;gap:6px;flex-wrap:wrap;margin-top:4px}
.mode-btn{padding:4px 10px;border-radius:6px;border:1px solid rgba(255,255,255,0.2);background:rgba(255,255,255,0.05);color:#bbb;font-size:11px;cursor:pointer;transition:all .15s}
.mode-btn.active{background:rgba(255,180,50,0.2);border-color:#ffb432;color:#ffb432;font-weight:700}
.toggle{display:flex;align-items:center;gap:8px;margin-top:8px;cursor:pointer;color:#bbb;font-size:12px}
.toggle input{accent-color:#ffb432}

#legend{position:fixed;bottom:20px;right:16px;background:rgba(8,8,20,0.94);border:1px solid rgba(255,255,255,0.15);border-radius:12px;padding:12px 16px;color:#ccc;font-size:12px;z-index:1000;min-width:210px}
#lg-title{color:#ffb432;font-weight:700;margin-bottom:8px;font-size:12px}
.lg-row{display:flex;align-items:center;gap:8px;margin-bottom:5px}
.lg-sw{width:14px;height:14px;border-radius:2px;flex-shrink:0}
.lg-divider{border:none;border-top:1px solid rgba(255,255,255,0.08);margin:7px 0}
.lg-mk-row{display:flex;align-items:center;gap:8px;margin-bottom:4px;font-size:11px}
.lg-dot{width:10px;height:10px;border-radius:50%;flex-shrink:0}

#hover-card{position:fixed;pointer-events:none;background:rgba(8,8,20,0.97);border:1px solid rgba(255,180,50,0.5);border-radius:8px;padding:10px 13px;font-size:12px;z-index:1000;display:none;min-width:200px;color:#eee}
.hc-gu{font-size:10px;color:#888;margin-bottom:2px}
.hc-name{font-size:13px;font-weight:700;color:#ffb432;margin-bottom:6px}
.hc-row{display:flex;justify-content:space-between;gap:12px;margin-bottom:3px}
.hc-val{color:#fff;font-weight:600}
.hc-fire{margin-top:6px;padding-top:6px;border-top:1px solid rgba(255,255,255,0.1);font-size:11px;color:#ff9060;font-weight:600}

#guide{position:fixed;bottom:20px;left:50%;transform:translateX(-50%);background:rgba(8,8,20,0.85);border:1px solid rgba(255,255,255,0.1);border-radius:20px;padding:5px 16px;color:#666;font-size:11px;z-index:1000;white-space:nowrap}
</style>
</head>
<body>
<div id="deckgl-wrapper"></div>

<div id="panel">
  <h3>📐 집계구별 숙박시설 3D</h3>
  <div class="stat"><span>숙박있는 집계구</span><span class="val">1,362개</span></div>
  <div class="stat"><span>평균 건축연령</span><span class="val">31.6년</span></div>
  <div class="stat"><span>평균 층수</span><span class="val">5.1층</span></div>
  <hr class="divider">
  <div style="font-size:11px;color:#777;margin-bottom:5px">높이 기준</div>
  <div class="mode-btns" id="height-btns">
    <button class="mode-btn active" onclick="setHeight('floors')">평균 층수</button>
    <button class="mode-btn" onclick="setHeight('count')">숙박시설 수</button>
    <button class="mode-btn" onclick="setHeight('fire')">화재위험도</button>
  </div>
  <hr class="divider">
  <div style="font-size:11px;color:#777;margin-bottom:5px">색상 기준</div>
  <div class="mode-btns" id="color-btns">
    <button class="mode-btn active" onclick="setColor('age')">노후도</button>
    <button class="mode-btn" onclick="setColor('fire')">화재위험</button>
    <button class="mode-btn" onclick="setColor('count')">숙박밀집</button>
  </div>
  <hr class="divider">
  <label class="toggle"><input type="checkbox" id="chk-station" checked> 소방서 / 안전센터</label>
  <label class="toggle"><input type="checkbox" id="chk-empty" checked> 빈 집계구 (바닥)</label>
</div>

<div id="legend">
  <div id="lg-title">노후도 (평균 건축연령)</div>
  <div id="lg-items"></div>
  <div class="lg-divider"></div>
  <div class="lg-mk-row"><div class="lg-dot" style="background:#ff4444"></div>소방서 (본서)</div>
  <div class="lg-mk-row"><div class="lg-dot" style="background:#ff9933"></div>119 안전센터</div>
</div>

<div id="hover-card">
  <div class="hc-gu" id="hc-gu"></div>
  <div class="hc-name" id="hc-name"></div>
  <div class="hc-row"><span>숙박시설 수</span><span class="hc-val" id="hc-count"></span></div>
  <div class="hc-row"><span>평균 층수</span><span class="hc-val" id="hc-floors"></span></div>
  <div class="hc-row"><span>평균 건축연령</span><span class="hc-val" id="hc-age"></span></div>
  <div class="hc-row"><span>숙박 비율</span><span class="hc-val" id="hc-ratio"></span></div>
  <div class="hc-fire" id="hc-fire"></div>
</div>

<div id="guide">🖱 드래그: 이동 &nbsp;|&nbsp; 우클릭 드래그: 회전/기울기 &nbsp;|&nbsp; 스크롤: 줌</div>

<script>
const FILLED   = """ + filled_json + """;
const EMPTY    = """ + empty_json  + """;
const STATIONS = """ + stations_json + """;

// ── 색상 정의 ─────────────────────────────────────────────────
const COLOR_MODES = {
  age: {
    title: '노후도 (평균 건축연령)',
    items: [
      {label:'0~10년 (신축)',   color:[200,230,255]},
      {label:'10~20년',         color:[160,210,150]},
      {label:'20~30년',         color:[255,215,80]},
      {label:'30~40년',         color:[255,140,30]},
      {label:'40~60년',         color:[210,60,30]},
      {label:'60년+ (노후건물)', color:[150,20,20]},
    ],
    fn: function(p) {
      var a = p.avg_age || 0;
      if(a < 10)  return [200,230,255,220];
      if(a < 20)  return [160,210,150,220];
      if(a < 30)  return [255,215,80,220];
      if(a < 40)  return [255,140,30,220];
      if(a < 60)  return [210,60,30,220];
      return [150,20,20,220];
    }
  },
  fire: {
    title: '화재위험도 점수',
    items: [
      {label:'0~25 (낮음)',   color:[80,200,100]},
      {label:'25~45 (보통)',  color:[255,200,50]},
      {label:'45~65 (높음)',  color:[255,120,30]},
      {label:'65+ (매우높음)',color:[220,30,30]},
    ],
    fn: function(p) {
      var s = p.fire_score || 0;
      if(s < 25) return [80,200,100,220];
      if(s < 45) return [255,200,50,220];
      if(s < 65) return [255,120,30,220];
      return [220,30,30,220];
    }
  },
  count: {
    title: '숙박시설 수',
    items: [
      {label:'1개',        color:[42,64,144]},
      {label:'2~3개',      color:[48,112,192]},
      {label:'4~6개',      color:[96,176,64]},
      {label:'7~10개',     color:[240,160,32]},
      {label:'11~20개',    color:[224,48,16]},
      {label:'21개 이상',  color:[155,0,0]},
    ],
    fn: function(p) {
      var c = p.count || 0;
      if(c <= 1)  return [42,64,144,220];
      if(c <= 3)  return [48,112,192,220];
      if(c <= 6)  return [96,176,64,220];
      if(c <= 10) return [240,160,32,220];
      if(c <= 20) return [224,48,16,220];
      return [155,0,0,220];
    }
  }
};

var heightMode = 'floors';
var colorMode  = 'age';
var showStation= true;
var showEmpty  = true;

function getHeight(p) {
  if(heightMode === 'floors') return (p.avg_floors || 1) * 3.5 * 8;   // ×8 배율로 시각화
  if(heightMode === 'count')  return (p.count || 1) * 20;
  if(heightMode === 'fire')   return (p.fire_score || 1) * 8;
  return 20;
}

// ── deck.gl 초기화 ────────────────────────────────────────────
var deckInstance = new deck.Deck({
  container: 'deckgl-wrapper',
  initialViewState: {
    longitude: 126.9740,
    latitude:  37.5530,
    zoom: 12.5,
    pitch: 55,
    bearing: -15
  },
  controller: true,
  layers: []
});

function refresh() {
  var layers = [];

  // 타일 베이스맵
  layers.push(new deck.TileLayer({
    id: 'basemap',
    data: 'https://a.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png',
    minZoom:0, maxZoom:19, tileSize:256,
    renderSubLayers: function(props) {
      var b = props.tile.bbox;
      return new deck.BitmapLayer(props, {
        data: null, image: props.data,
        bounds: [b.west, b.south, b.east, b.north]
      });
    }
  }));

  // 빈 집계구 (바닥 레이어)
  if(showEmpty) {
    layers.push(new deck.GeoJsonLayer({
      id: 'empty-oa',
      data: EMPTY,
      filled: true,
      stroked: true,
      extruded: false,
      getFillColor: [30,30,50,80],
      getLineColor: [80,80,100,100],
      lineWidthMinPixels: 0.3
    }));
  }

  // 숙박 있는 집계구 — 3D 압출
  layers.push(new deck.GeoJsonLayer({
    id: 'filled-oa',
    data: FILLED,
    filled: true,
    stroked: true,
    extruded: true,
    wireframe: false,
    getFillColor: function(f) { return COLOR_MODES[colorMode].fn(f.properties); },
    getLineColor: [255,255,255,40],
    lineWidthMinPixels: 0.5,
    getElevation: function(f) { return getHeight(f.properties); },
    elevationScale: 1,
    pickable: true,
    autoHighlight: true,
    highlightColor: [255,220,80,60],
    material: {
      ambient: 0.4,
      diffuse: 0.6,
      shininess: 20,
      specularColor: [255,255,255]
    },
    onHover: function(info) {
      var card = document.getElementById('hover-card');
      if(info.object) {
        var p = info.object.properties;
        card.style.display = 'block';
        card.style.left = (info.x + 16) + 'px';
        card.style.top  = Math.max(10, info.y - 160) + 'px';
        document.getElementById('hc-gu').textContent    = p.gu_name || '';
        document.getElementById('hc-name').textContent  = '집계구 #' + (p.oa_no || p.id.slice(8));
        document.getElementById('hc-count').textContent = p.count + ' 개';
        document.getElementById('hc-floors').textContent= (p.avg_floors != null ? p.avg_floors.toFixed(1) + '층' : '-');
        document.getElementById('hc-age').textContent   = (p.avg_age   != null ? p.avg_age.toFixed(1)   + '년' : '-');
        document.getElementById('hc-ratio').textContent = p.ratio.toFixed(1) + ' %';
        var lv = p.fire_score >= 65 ? '매우높음' : p.fire_score >= 45 ? '높음' : p.fire_score >= 25 ? '보통' : '낮음';
        document.getElementById('hc-fire').textContent  = '🔥 화재위험도 ' + (p.fire_score || 0).toFixed(0) + '점 (' + lv + ')';
      } else {
        card.style.display = 'none';
      }
    }
  }));

  // 소방서 / 안전센터
  if(showStation) {
    var fs = STATIONS.filter(function(s){ return s.type==='소방서'; });
    var sc = STATIONS.filter(function(s){ return s.type==='안전센터'; });
    layers.push(new deck.ColumnLayer({
      id:'fs', data:fs,
      getPosition:function(d){return[d.lng,d.lat];},
      getElevation: 200, radius:50,
      getFillColor:[255,50,50,230],
      getLineColor:[255,255,255,200],
      lineWidthMinPixels:1, extruded:true, pickable:true,
      onHover:function(info){
        var card=document.getElementById('hover-card');
        if(info.object){
          card.style.display='block';
          card.style.left=(info.x+16)+'px'; card.style.top=(info.y-60)+'px';
          document.getElementById('hc-gu').textContent='소방서';
          document.getElementById('hc-name').textContent='🚒 '+info.object.name;
          document.getElementById('hc-count').textContent=info.object.count+'건';
          document.getElementById('hc-floors').textContent='';
          document.getElementById('hc-age').textContent='';
          document.getElementById('hc-ratio').textContent='';
          document.getElementById('hc-fire').textContent='';
        } else { card.style.display='none'; }
      }
    }));
    layers.push(new deck.ColumnLayer({
      id:'sc', data:sc,
      getPosition:function(d){return[d.lng,d.lat];},
      getElevation:100, radius:30,
      getFillColor:[255,150,30,220],
      getLineColor:[255,255,255,160],
      lineWidthMinPixels:0.5, extruded:true, pickable:true,
      onHover:function(info){
        var card=document.getElementById('hover-card');
        if(info.object){
          card.style.display='block';
          card.style.left=(info.x+16)+'px'; card.style.top=(info.y-60)+'px';
          document.getElementById('hc-gu').textContent='119 안전센터';
          document.getElementById('hc-name').textContent='🔶 '+info.object.name;
          document.getElementById('hc-count').textContent=info.object.count+'건';
          document.getElementById('hc-floors').textContent='';
          document.getElementById('hc-age').textContent='';
          document.getElementById('hc-ratio').textContent='';
          document.getElementById('hc-fire').textContent='';
        } else { card.style.display='none'; }
      }
    }));
  }

  deckInstance.setProps({layers: layers});
  renderLegend();
}

// ── 범례 ─────────────────────────────────────────────────────
function renderLegend() {
  var m = COLOR_MODES[colorMode];
  document.getElementById('lg-title').textContent = m.title;
  var html = '';
  m.items.forEach(function(item) {
    var c = item.color;
    html += '<div class="lg-row">'
          + '<div class="lg-sw" style="background:rgb('+c[0]+','+c[1]+','+c[2]+')"></div>'
          + '<span>'+item.label+'</span></div>';
  });
  // 높이 기준 설명
  var hDesc = heightMode==='floors' ? '높이 = 평균 층수 × 3.5m (×8 배율)'
            : heightMode==='count'  ? '높이 = 숙박시설 수 × 20m'
            : '높이 = 화재위험점수 × 8m';
  html += '<div style="margin-top:6px;font-size:10px;color:#555;border-top:1px solid rgba(255,255,255,0.07);padding-top:5px">'+hDesc+'</div>';
  document.getElementById('lg-items').innerHTML = html;
}

// ── 버튼 토글 ─────────────────────────────────────────────────
function setHeight(mode) {
  heightMode = mode;
  document.querySelectorAll('#height-btns .mode-btn').forEach(function(b, i) {
    var modes = ['floors','count','fire'];
    b.classList.toggle('active', modes[i] === mode);
  });
  refresh();
}
function setColor(mode) {
  colorMode = mode;
  document.querySelectorAll('#color-btns .mode-btn').forEach(function(b, i) {
    var modes = ['age','fire','count'];
    b.classList.toggle('active', modes[i] === mode);
  });
  refresh();
}
document.getElementById('chk-station').addEventListener('change', function(){
  showStation = this.checked; refresh();
});
document.getElementById('chk-empty').addEventListener('change', function(){
  showEmpty = this.checked; refresh();
});

refresh();
</script>
</body>
</html>
"""

out = 'c:/Users/USER/Documents/GitHub/기말공모전/NJT-PJT/숙박시설_3D.html'
with open(out, 'w', encoding='utf-8') as f:
    f.write(html)
print(f'Done: {os.path.getsize(out)//1024} KB')
