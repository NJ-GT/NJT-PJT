# -*- coding: utf-8 -*-
"""
=============================================================
[목적] 숙박업 건물 종합 시각화 지도 생성 (3개 레이어 통합)

[입력]  data/등기부등본_숙박업_핵심피처.csv   (숙박업 건물 + EPSG:5174 좌표)
        data/화재출동/화재출동_사상자발생.csv  (filter_casualties.py 출력)
        (관광특구 좌표는 스크립트 내 하드코딩)

[출력]  data/숙박업_종합지도.html

[레이어]
  - 숙박업 건물: 사용승인일 기준 색상 그라데이션 (오래될수록 붉음)
  - 관광특구: 중심점 기준 1km 반투명 원
  - 사상자 화재: 사망(빨강) / 부상(주황) 마커

[좌표 변환]  EPSG:5174 → WGS84 (pyproj)

[사용 라이브러리]  pandas, folium, folium.plugins.HeatMap, numpy, pyproj
=============================================================
"""
import pandas as pd, os, folium, numpy as np
from folium.plugins import HeatMap
from pyproj import Transformer
from collections import Counter

BASE = r'C:\Users\USER\Documents\GitHub\기말공모전\NJT-PJT'

# ── 1. 숙박업 핵심피처 로드 ───────────────────────────────────────────
feat = pd.read_csv(os.path.join(BASE, 'data', '등기부등본_숙박업_핵심피처.csv'),
                   encoding='utf-8-sig', low_memory=False)
print(f'숙박업 피처: {len(feat)}행')
print('컬럼:', feat.columns.tolist())

# 좌표 컬럼 찾기
x_col = [c for c in feat.columns if '좌표' in c and 'X' in c][0]
y_col = [c for c in feat.columns if '좌표' in c and 'Y' in c][0]
print(f'좌표 컬럼: {x_col}, {y_col}')

feat['_x'] = pd.to_numeric(feat[x_col], errors='coerce')
feat['_y'] = pd.to_numeric(feat[y_col], errors='coerce')
feat = feat.dropna(subset=['_x','_y'])

# EPSG:5174 → WGS84 변환
transformer = Transformer.from_crs('EPSG:5174', 'EPSG:4326', always_xy=True)
lats, lons = [], []
for _, row in feat.iterrows():
    x, y = row['_x'], row['_y']
    if 125 < x < 130 and 36 < y < 38:
        lats.append(y); lons.append(x)
    else:
        lon, lat = transformer.transform(x, y)
        lats.append(lat); lons.append(lon)
feat['lat'] = lats
feat['lon'] = lons
feat = feat[(feat['lat']>37.4)&(feat['lat']<37.7)&(feat['lon']>126.7)&(feat['lon']<127.3)]
print(f'유효 좌표 숙박업: {len(feat)}건')

# 사용승인일 → 연도 파싱
def parse_year(v):
    try:
        s = str(int(float(v)))
        if len(s) == 8:
            return int(s[:4])
    except: pass
    return None

feat['_year'] = feat['사용승인일'].apply(parse_year)
valid = feat.dropna(subset=['_year'])
print(f'사용승인일 유효: {len(valid)}건, 범위: {int(valid["_year"].min())}~{int(valid["_year"].max())}')

# ── 2. 사상자 발생 화재 로드 ──────────────────────────────────────────
fire = pd.read_csv(os.path.join(BASE, 'data', '화재출동', '화재출동_사상자발생.csv'),
                   encoding='utf-8-sig', low_memory=False)
fire['위도'] = pd.to_numeric(fire['위도'], errors='coerce')
fire['경도'] = pd.to_numeric(fire['경도'], errors='coerce')
fire = fire.dropna(subset=['위도','경도'])
fire = fire[(fire['위도']>37.4)&(fire['위도']<37.7)&(fire['경도']>126.7)&(fire['경도']<127.3)]
print(f'사상자 화재: {len(fire)}건')

# ── 3. 관광특구 (7개구 관련) ──────────────────────────────────────────
관광특구 = [
    {'name': '명동·남대문·북창 관광특구',  'lat': 37.5635, 'lon': 126.9826},
    {'name': '이태원 관광특구',             'lat': 37.5344, 'lon': 126.9946},
    {'name': '동대문 패션타운 관광특구',    'lat': 37.5666, 'lon': 127.0092},
    {'name': '종로·청계 관광특구',          'lat': 37.5700, 'lon': 126.9826},
    {'name': '잠실 관광특구',               'lat': 37.5133, 'lon': 127.1000},
    {'name': '강남 마이스 관광특구',        'lat': 37.5117, 'lon': 127.0590},
    {'name': '홍대 관광특구',               'lat': 37.5563, 'lon': 126.9238},
]

# ── 4. 사용승인일 → 색상 매핑 ────────────────────────────────────────
def year_to_color(year):
    if year is None or (isinstance(year, float) and np.isnan(year)): return '#aaaaaa'
    year = int(year)
    if year < 1970:   return '#2c3e50'
    elif year < 1980: return '#8e44ad'
    elif year < 1990: return '#2980b9'
    elif year < 2000: return '#27ae60'
    elif year < 2010: return '#f39c12'
    elif year < 2020: return '#e67e22'
    else:             return '#e74c3c'

# ── 5. 지도 생성 ──────────────────────────────────────────────────────
m = folium.Map(location=[37.555, 126.977], zoom_start=12, tiles='CartoDB positron')

fg_tour    = folium.FeatureGroup(name='🏖️ 관광특구 (1km)', show=True)
fg_hotel   = folium.FeatureGroup(name='🏨 숙박업 (사용승인일)', show=True)
fg_fire    = folium.FeatureGroup(name='🔥 사상자 발생 화재', show=True)
fg_heat    = folium.FeatureGroup(name='🌡️ 화재 히트맵', show=False)

# 관광특구 원
for t in 관광특구:
    folium.Circle(
        location=[t['lat'], t['lon']],
        radius=1000,
        color='#e74c3c', fill=True, fill_color='#e74c3c',
        fill_opacity=0.12, weight=2, opacity=0.6,
        tooltip=t['name']
    ).add_to(fg_tour)
    folium.Marker(
        location=[t['lat'], t['lon']],
        icon=folium.DivIcon(
            html=f'<div style="font-size:11px;font-weight:bold;color:#c0392b;'
                 f'white-space:nowrap;text-shadow:1px 1px 2px white,-1px -1px 2px white">{t["name"]}</div>',
            icon_anchor=(0, 10)
        )
    ).add_to(fg_tour)

# 숙박업 마커 (사용승인일 색상)
for _, row in feat.iterrows():
    year = row.get('_year')
    color = year_to_color(year)
    year_str = str(int(year)) if (year and not (isinstance(year, float) and np.isnan(year))) else '미상'
    popup = (f"<b>{row.get('도로명대지위치','')}</b><br>"
             f"사용승인일: {year_str}년<br>"
             f"연면적: {row.get('연면적(㎡)','')}㎡<br>"
             f"지상층수: {row.get('지상층수','')}층<br>"
             f"주용도: {row.get('주용도코드명','')}<br>"
             f"구조: {row.get('구조코드명','')}")
    folium.CircleMarker(
        location=[row['lat'], row['lon']],
        radius=5, color=color, fill=True, fill_color=color,
        fill_opacity=0.8, weight=0.8,
        popup=folium.Popup(popup, max_width=280)
    ).add_to(fg_hotel)

# 화재 히트맵
heat_data = [[r['위도'], r['경도'], 2 if r['사망자수']>=1 else 1] for _, r in fire.iterrows()]
HeatMap(heat_data, radius=18, blur=14, max_zoom=14,
        gradient={0.2:'blue',0.5:'lime',0.8:'yellow',1.0:'red'}).add_to(fg_heat)

# 사상자 마커
for _, row in fire.iterrows():
    is_death = row['사망자수'] >= 1
    color = '#c0392b' if is_death else '#e67e22'
    size = 5 + int(row['사망자수'])*3 if is_death else 4 + min(int(row['부상자수']),4)
    popup = (f"<b style='color:{color}'>사망 {int(row['사망자수'])}명 / 부상 {int(row['부상자수'])}명</b><br>"
             f"발생: {str(row.get('발생일자',''))[:10]}<br>"
             f"장소: {row.get('발화장소_대분류','')} &gt; {row.get('발화장소_소분류','')}<br>"
             f"구: {row.get('발생시군구','')}")
    folium.CircleMarker(
        location=[row['위도'], row['경도']],
        radius=size, color=color, fill=True, fill_color=color,
        fill_opacity=0.85, weight=1.5,
        popup=folium.Popup(popup, max_width=280)
    ).add_to(fg_fire)

fg_tour.add_to(m); fg_hotel.add_to(m); fg_fire.add_to(m); fg_heat.add_to(m)
folium.LayerControl(collapsed=False).add_to(m)

# ── 6. 범례 ──────────────────────────────────────────────────────────
yr_legend = [
    ('#2c3e50','~1969'),('#8e44ad','1970s'),('#2980b9','1980s'),
    ('#27ae60','1990s'),('#f39c12','2000s'),('#e67e22','2010s'),('#e74c3c','2020~'),
    ('#aaaaaa','미상'),
]
yr_rows = ''.join([
    f'<span style="display:inline-block;width:12px;height:12px;background:{c};border-radius:50%;margin-right:4px"></span>'
    f'<span style="font-size:11px;margin-right:8px">{l}</span>'
    for c,l in yr_legend
])

cnt_year = Counter(valid['_year'].apply(lambda y: f'{int(y)//10*10}s' if y else '미상'))

legend_html = f'''
<div style="position:fixed;bottom:30px;left:30px;z-index:1000;background:white;
            padding:14px 18px;border-radius:10px;box-shadow:2px 2px 8px rgba(0,0,0,0.3);font-size:12px;min-width:240px">
  <b>🏨 숙박업 사용승인일 × 🔥 사상자 화재</b>
  <hr style="margin:8px 0">
  <b>사용승인일 (숙박업 {len(feat)}개)</b><br>
  <div style="margin:6px 0;line-height:2">{yr_rows}</div>
  <hr style="margin:8px 0">
  <b>🔥 사상자 발생 화재 ({len(fire)}건)</b><br>
  <span style="color:#c0392b;font-size:13px">●</span> 사망자 발생 &nbsp;
  <span style="color:#e67e22;font-size:13px">●</span> 부상자만 발생<br>
  <span style="font-size:11px">마커 크기 = 사상자수 비례</span>
  <hr style="margin:8px 0">
  <b>🏖️ 관광특구</b> 반경 1km 표시
</div>'''
m.get_root().html.add_child(folium.Element(legend_html))

out_html = os.path.join(BASE, 'data', '화재출동', '숙박업_사용승인일_사상자_지도.html')
m.save(out_html)
print(f'저장: {out_html}')
