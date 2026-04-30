# -*- coding: utf-8 -*-
"""
[파일 설명]
연도별(2021~2024) 화재출동 CSV 파일들을 하나로 합치고, 서울 7개구 화재 현황 히트맵을 생성하는 스크립트.

주요 역할:
  1. 2021~2024년 화재출동 CSV를 합쳐 data/화재출동/화재출동_2021_2024.csv로 저장한다.
  2. 카카오 API로 서울 소방시설 위치를 검색한다.
  3. Folium으로 출동 위치 히트맵 + 소방시설 마커를 합친 HTML 지도를 생성한다.

입력: 화재출동_2021/화재출동_2021.csv  (및 2022, 2023, 2024 동일 구조)
출력: data/화재출동/화재출동_2021_2024.csv   (통합 출동 데이터 → 화재위험도_계산.py 입력)
      data/화재출동/화재출동_서울_지도.html   (Folium 히트맵 시각화)

실행 순서: merge_fire.py → (filter_casualties.py) → 화재위험도_계산.py
"""
import pandas as pd, os, folium, requests, time
from pyproj import Transformer
from folium.plugins import HeatMap

BASE = r'C:\Users\USER\Documents\GitHub\기말공모전\NJT-PJT'
OUT  = os.path.join(BASE, 'data', '화재출동')
os.makedirs(OUT, exist_ok=True)

# ── 1. 화재출동 2021~2024 합치기 ──────────────────────────────────
files = [os.path.join(BASE, f'화재출동_{y}', f'화재출동_{y}.csv') for y in range(2021, 2025)]
dfs = []
for f in files:
    for enc in ['utf-8-sig', 'cp949', 'euc-kr']:
        try:
            df = pd.read_csv(f, encoding=enc, low_memory=False)
            print(f'로드: {os.path.basename(f)} ({len(df)}행)')
            dfs.append(df)
            break
        except Exception as e:
            continue

merged = pd.concat(dfs, ignore_index=True)
merged.to_csv(os.path.join(OUT, '화재출동_2021_2024.csv'), index=False, encoding='utf-8-sig')
print(f'합계: {len(merged)}행 x {len(merged.columns)}컬럼')

# ── 2. 서울 + 7개구 필터 ──────────────────────────────────────────
seoul_7 = ['강남구','마포구','서초구','송파구','용산구','종로구','중구']
서울 = merged[(merged['GRNDS_CTPV_NM']=='서울특별시') &
              (merged['GRNDS_SGG_NM'].isin(seoul_7))].copy()
서울['DAMG_RGN_LAT'] = pd.to_numeric(서울['DAMG_RGN_LAT'], errors='coerce')
서울['DAMG_RGN_LOT'] = pd.to_numeric(서울['DAMG_RGN_LOT'], errors='coerce')
서울 = 서울.dropna(subset=['DAMG_RGN_LAT','DAMG_RGN_LOT'])
서울 = 서울[(서울['DAMG_RGN_LAT']>37.4) & (서울['DAMG_RGN_LAT']<37.7)]
print(f'서울 7개구 화재: {len(서울)}건 (좌표있음)')

# ── 3. 소방시설 가져오기 ──────────────────────────────────────────
KAKAO_KEY = '96172db4c3b086f76853ed89242acefa'

def search_all(query):
    results = []
    for page in range(1, 4):
        r = requests.get('https://dapi.kakao.com/v2/local/search/keyword.json',
            headers={'Authorization': f'KakaoAK {KAKAO_KEY}'},
            params={'query': query, 'size': 15, 'page': page}, timeout=5)
        docs = r.json().get('documents', [])
        results.extend(docs)
        if r.json().get('meta', {}).get('is_end', True): break
        time.sleep(0.1)
    return results

all_places = {}
for type_name, query in [('소방서','서울특별시 소방서'),'안전센터','서울특별시 119안전센터'),'구조대','서울특별시 구조대')]:
    for d in search_all(query):
        if not d.get('address_name','').startswith('서울'): continue
        pid = d['id']
        if pid not in all_places:
            all_places[pid] = {'name':d['place_name'],'lat':float(d['y']),'lon':float(d['x']),
                               'addr':d.get('road_address_name',d.get('address_name','')),'type':type_name}
stations = list(all_places.values())
print(f'소방시설: {len(stations)}개')

# ── 4. 지도 생성 ──────────────────────────────────────────────────
m = folium.Map(location=[37.555, 126.977], zoom_start=12, tiles='CartoDB positron')

fg_heat   = folium.FeatureGroup(name='🔥 화재출동 히트맵', show=True)
fg_fire   = folium.FeatureGroup(name='📍 화재출동 위치', show=False)
fg_station= folium.FeatureGroup(name='🚒 소방시설', show=True)

# 히트맵
heat_data = [[r['DAMG_RGN_LAT'], r['DAMG_RGN_LOT']] for _, r in 서울.iterrows()]
HeatMap(heat_data, radius=15, blur=12, max_zoom=14,
        gradient={0.2:'blue',0.5:'lime',0.8:'yellow',1.0:'red'}).add_to(fg_heat)

# 화재 개별 마커 (건물구분별 색상)
bldg_colors = {'주거':'#e74c3c','업무시설':'#3498db','판매/영업':'#e67e22',
               '공장/창고':'#8e44ad','기타':'#95a5a6'}
for _, row in 서울.iterrows():
    fclt = str(row.get('FCLT_PLC_LCLSF_NM',''))
    color = '#e74c3c' if '주거' in fclt else ('#3498db' if '업무' in fclt else
            '#e67e22' if '판매' in fclt or '영업' in fclt else
            '#8e44ad' if '공장' in fclt or '창고' in fclt else '#95a5a6')
    popup = (f"<b>{row.get('FCLT_PLC_SCLSF_NM','')}</b><br>"
             f"발생: {str(row.get('OCRN_YMD',''))[:8]}<br>"
             f"구분: {fclt}<br>"
             f"사망: {row.get('DTH_CNT',0)}명 / 부상: {row.get('INJPSN_CNT',0)}명<br>"
             f"재산피해: {row.get('PRPT_DAM_AMT',0):,.0f}천원")
    folium.CircleMarker(
        location=[row['DAMG_RGN_LAT'], row['DAMG_RGN_LOT']],
        radius=4, color=color, fill=True, fill_color=color,
        fill_opacity=0.75, weight=0.5,
        popup=folium.Popup(popup, max_width=280)
    ).add_to(fg_fire)

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

fg_heat.add_to(m); fg_fire.add_to(m); fg_station.add_to(m)
folium.LayerControl(collapsed=False).add_to(m)

from collections import Counter
cnt_구 = Counter(서울['GRNDS_SGG_NM'])
cnt_sta = Counter(s['type'] for s in stations)

legend_html = f'''
<div style="position:fixed;bottom:30px;left:30px;z-index:1000;background:white;
            padding:14px 18px;border-radius:10px;box-shadow:2px 2px 8px rgba(0,0,0,0.3);font-size:12px;min-width:210px">
  <b>🔥 화재출동 현황 (2021~2024)</b><br>
  <div style="margin:6px 0;font-size:11px">총 <b>{len(서울)}건</b></div>
  {''.join([f'<span style="font-size:11px">{k}: {v}건</span><br>' for k,v in sorted(cnt_구.items(), key=lambda x:-x[1])])}
  <hr style="margin:8px 0">
  <b>히트맵 강도</b>
  <div style="width:185px;height:12px;background:linear-gradient(to right,blue,lime,yellow,red);border-radius:3px;margin:5px 0 2px"></div>
  <div style="display:flex;justify-content:space-between;font-size:10px;width:185px"><span>낮음</span><span>높음</span></div>
  <hr style="margin:8px 0">
  <b>소방시설 ({len(stations)}개)</b><br>
  🚒 소방서 {cnt_sta["소방서"]}개 &nbsp; ⛑️ 구조대 {cnt_sta["구조대"]}개<br>
  🏥 안전센터 {cnt_sta["안전센터"]}개
</div>'''
m.get_root().html.add_child(folium.Element(legend_html))

out_html = os.path.join(OUT, '화재출동_서울_지도.html')
m.save(out_html)
print(f'저장: {out_html}')
