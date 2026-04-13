# -*- coding: utf-8 -*-
"""
=============================================================
[목적] 사상자 발생 화재 시각화 지도 생성
       사망/부상 규모별 마커 + 히트맵 + 소방시설 레이어

[입력]  data/화재출동/화재출동_사상자발생.csv  (filter_casualties.py 출력)

[출력]  data/화재출동/화재출동_사상자발생_지도.html

[레이어]
  - 히트맵: 사망자 가중치 3, 부상자 가중치 1
  - 빨간 원: 사망자 발생 (크기 = 사망자수)
  - 주황 원: 부상자만 발생
  - 소방시설: 소방서/안전센터/구조대 (Kakao API 조회)

[사용 라이브러리]  pandas, folium, folium.plugins.HeatMap, requests
[외부 API]  Kakao Local API (소방시설 검색)
=============================================================
"""
import pandas as pd, os, folium, requests, time
from folium.plugins import HeatMap
from collections import Counter

BASE = r'C:\Users\USER\Documents\GitHub\기말공모전\NJT-PJT'
OUT  = os.path.join(BASE, 'data', '화재출동')

df = pd.read_csv(os.path.join(OUT, '화재출동_사상자발생.csv'), encoding='utf-8-sig', low_memory=False)
print(f'로드: {len(df)}행')

# 좌표 정제
df['위도'] = pd.to_numeric(df['위도'], errors='coerce')
df['경도'] = pd.to_numeric(df['경도'], errors='coerce')
df = df.dropna(subset=['위도','경도'])
df = df[(df['위도']>37.4) & (df['위도']<37.7) & (df['경도']>126.7) & (df['경도']<127.3)]
print(f'유효 좌표: {len(df)}건')

# 소방시설 가져오기
KAKAO_KEY = '96172db4c3b086f76853ed89242acefa'

def search_all(query):
    results = []
    for page in range(1, 4):
        try:
            r = requests.get('https://dapi.kakao.com/v2/local/search/keyword.json',
                headers={'Authorization': f'KakaoAK {KAKAO_KEY}'},
                params={'query': query, 'size': 15, 'page': page}, timeout=5)
            docs = r.json().get('documents', [])
            results.extend(docs)
            if r.json().get('meta', {}).get('is_end', True): break
            time.sleep(0.1)
        except: break
    return results

all_places = {}
for type_name, query in [('소방서','서울특별시 소방서'),('안전센터','서울특별시 119안전센터'),('구조대','서울특별시 구조대')]:
    for d in search_all(query):
        if not d.get('address_name','').startswith('서울'): continue
        pid = d['id']
        if pid not in all_places:
            all_places[pid] = {'name':d['place_name'],'lat':float(d['y']),'lon':float(d['x']),
                               'addr':d.get('road_address_name',d.get('address_name','')),'type':type_name}
stations = list(all_places.values())
print(f'소방시설: {len(stations)}개')

# ── 지도 생성 ──────────────────────────────────────────────────────
m = folium.Map(location=[37.555, 126.977], zoom_start=12, tiles='CartoDB positron')

fg_heat    = folium.FeatureGroup(name='🔥 히트맵', show=True)
fg_death   = folium.FeatureGroup(name='💀 사망자 발생', show=True)
fg_injured = folium.FeatureGroup(name='🩹 부상자 발생 (사망없음)', show=False)
fg_station = folium.FeatureGroup(name='🚒 소방시설', show=True)

# 히트맵 (사망자는 가중치 높게)
heat_data = []
for _, row in df.iterrows():
    w = 3 if row['사망자수'] >= 1 else 1
    heat_data.append([row['위도'], row['경도'], w])
HeatMap(heat_data, radius=18, blur=14, max_zoom=14,
        gradient={0.2:'blue',0.5:'lime',0.8:'yellow',1.0:'red'}).add_to(fg_heat)

# 마커: 사망자 있는 건 (빨간 별)
death_df = df[df['사망자수'] >= 1]
for _, row in death_df.iterrows():
    fclt = str(row.get('발화장소_대분류',''))
    popup = (f"<b style='color:#c0392b'>사망 {int(row['사망자수'])}명 / 부상 {int(row['부상자수'])}명</b><br>"
             f"발생: {str(row.get('발생일자',''))[:10]}<br>"
             f"장소: {fclt} &gt; {row.get('발화장소_소분류','')}<br>"
             f"요인: {row.get('발화요인_대분류','')}<br>"
             f"열원: {row.get('발화열원_대분류','')}<br>"
             f"구: {row.get('발생시군구','')}")
    size = 6 + int(row['사망자수']) * 3
    folium.CircleMarker(
        location=[row['위도'], row['경도']],
        radius=size, color='#c0392b', fill=True, fill_color='#e74c3c',
        fill_opacity=0.85, weight=1.5,
        popup=folium.Popup(popup, max_width=300)
    ).add_to(fg_death)

# 마커: 부상자만 있는 건 (주황)
injured_df = df[(df['사망자수'] == 0) & (df['부상자수'] >= 1)]
for _, row in injured_df.iterrows():
    fclt = str(row.get('발화장소_대분류',''))
    popup = (f"<b style='color:#e67e22'>부상 {int(row['부상자수'])}명</b><br>"
             f"발생: {str(row.get('발생일자',''))[:10]}<br>"
             f"장소: {fclt} &gt; {row.get('발화장소_소분류','')}<br>"
             f"요인: {row.get('발화요인_대분류','')}<br>"
             f"구: {row.get('발생시군구','')}")
    size = 4 + min(int(row['부상자수']), 5)
    folium.CircleMarker(
        location=[row['위도'], row['경도']],
        radius=size, color='#e67e22', fill=True, fill_color='#f39c12',
        fill_opacity=0.75, weight=1,
        popup=folium.Popup(popup, max_width=280)
    ).add_to(fg_injured)

# 소방시설
type_cfg = {'소방서':{'emoji':'🚒','color':'#c0392b','size':28},
            '구조대':{'emoji':'⛑️','color':'#2980b9','size':24},
            '안전센터':{'emoji':'🏥','color':'#e67e22','size':24}}
for s in sorted(stations, key=lambda x: ['소방서','구조대','안전센터'].index(x['type'])):
    cfg = type_cfg[s['type']]
    icon_html = (f'<div style="background:{cfg["color"]};width:{cfg["size"]}px;height:{cfg["size"]}px;'
                 f'border-radius:50%;display:flex;align-items:center;justify-content:center;'
                 f'font-size:{cfg["size"]-10}px;border:2px solid white;box-shadow:0 2px 5px rgba(0,0,0,0.5)">{cfg["emoji"]}</div>')
    folium.Marker(location=[s['lat'],s['lon']],
        icon=folium.DivIcon(html=icon_html, icon_size=(cfg['size'],cfg['size']),
                            icon_anchor=(cfg['size']//2,cfg['size']//2)),
        tooltip=s['name'],
        popup=folium.Popup(f"<b>{s['name']}</b><br>{s['addr']}", max_width=250)).add_to(fg_station)

fg_heat.add_to(m); fg_death.add_to(m); fg_injured.add_to(m); fg_station.add_to(m)
folium.LayerControl(collapsed=False).add_to(m)

# 통계
cnt_구 = Counter(df['발생시군구'].dropna())
cnt_장소 = Counter(df['발화장소_대분류'].dropna())
total_death = int(df['사망자수'].sum())
total_injured = int(df['부상자수'].sum())

top_장소 = sorted(cnt_장소.items(), key=lambda x:-x[1])[:5]

legend_html = f'''
<div style="position:fixed;bottom:30px;left:30px;z-index:1000;background:white;
            padding:14px 18px;border-radius:10px;box-shadow:2px 2px 8px rgba(0,0,0,0.3);font-size:12px;min-width:220px">
  <b>🔥 사상자 발생 화재 (2021~2024)</b><br>
  <div style="margin:6px 0;font-size:11px">
    총 <b>{len(df)}건</b> &nbsp;|&nbsp;
    사망 <b style="color:#c0392b">{total_death}명</b> &nbsp;|&nbsp;
    부상 <b style="color:#e67e22">{total_injured}명</b>
  </div>
  <hr style="margin:8px 0">
  <b>구별 발생</b><br>
  {''.join([f'<span style="font-size:11px">{k}: {v}건</span><br>' for k,v in sorted(cnt_구.items(), key=lambda x:-x[1])])}
  <hr style="margin:8px 0">
  <b>발화장소 TOP5</b><br>
  {''.join([f'<span style="font-size:11px">{k}: {v}건</span><br>' for k,v in top_장소])}
  <hr style="margin:8px 0">
  <b>마커 범례</b><br>
  <span style="color:#e74c3c;font-size:13px">●</span> 사망자 발생 (크기=사망자수)<br>
  <span style="color:#f39c12;font-size:13px">●</span> 부상자만 발생
</div>'''
m.get_root().html.add_child(folium.Element(legend_html))

out_html = os.path.join(OUT, '화재출동_사상자발생_지도.html')
m.save(out_html)
print(f'저장: {out_html}')
