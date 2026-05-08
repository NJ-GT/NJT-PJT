# -*- coding: utf-8 -*-
"""
숙박업 건물 종합 시각화 지도 (3개 레이어 통합) — folium 인터랙티브 HTML.

목적:
    - 등기부등본 핵심 피처와 사상자 발생 화재, 관광특구 위치를 한 지도 위에 결합해
      시각적으로 위험·노후·관광권 중첩을 한 번에 확인.

레이어:
    - 숙박업 건물 : 사용승인일 기준 색상 그라데이션 (오래될수록 짙은 색)
    - 관광특구    : 중심점 기준 1km 반투명 원 + 라벨
    - 사상자 화재 : 사망(빨강) / 부상(주황) 마커 + 토글 가능한 히트맵

좌표 변환:
    - EPSG:5174 (구 GRS80 중부원점) → WGS84 (위경도) — pyproj 사용
    - 단, x 가 이미 WGS84 경도 범위(125~130)면 변환하지 않고 그대로 사용

입력:
    - data/등기부등본_숙박업_핵심피처.csv          (숙박업 건물 + 좌표/사용승인일/주용도)
    - data/화재출동/화재출동_사상자발생.csv        (filter_casualties.py 출력)

출력:
    - data/화재출동/숙박업_사용승인일_사상자_지도.html
"""
import pandas as pd
import os
import folium
import numpy as np
from folium.plugins import HeatMap  # 화재 히트맵 레이어
from pyproj import Transformer  # 좌표계 변환
from collections import Counter  # 연도 분포 카운트


BASE = r"C:\Users\USER\Documents\GitHub\기말공모전\NJT-PJT"


# ── 1. 숙박업 핵심피처 로드 ───────────────────────────────────────────
feat = pd.read_csv(
    os.path.join(BASE, "data", "등기부등본_숙박업_핵심피처.csv"),
    encoding="utf-8-sig",
    low_memory=False,
)
print(f"숙박업 피처: {len(feat)}행")
print("컬럼:", feat.columns.tolist())

# 컬럼명에 'X'/'Y'와 '좌표'가 포함된 컬럼을 동적으로 탐지
# (원본 데이터의 정확한 컬럼명을 모를 때도 안전하게 작동)
x_col = [c for c in feat.columns if "좌표" in c and "X" in c][0]
y_col = [c for c in feat.columns if "좌표" in c and "Y" in c][0]
print(f"좌표 컬럼: {x_col}, {y_col}")

# 숫자형 변환 후 결측 제거
feat["_x"] = pd.to_numeric(feat[x_col], errors="coerce")
feat["_y"] = pd.to_numeric(feat[y_col], errors="coerce")
feat = feat.dropna(subset=["_x", "_y"])

# EPSG:5174 → WGS84 (위경도) 좌표 변환기
# 단, x 가 이미 WGS84 경도 범위(125~130)면 변환하지 않고 그대로 사용
transformer = Transformer.from_crs("EPSG:5174", "EPSG:4326", always_xy=True)
lats, lons = [], []
for _, row in feat.iterrows():
    x, y = row["_x"], row["_y"]
    # 이미 WGS84 위경도인 데이터는 그대로 사용
    if 125 < x < 130 and 36 < y < 38:
        lats.append(y)
        lons.append(x)
    else:
        # EPSG:5174 평면좌표 → WGS84 변환
        lon, lat = transformer.transform(x, y)
        lats.append(lat)
        lons.append(lon)
feat["lat"] = lats
feat["lon"] = lons

# 서울 범위 밖의 이상 좌표 필터링 (대략적 bounding box)
feat = feat[
    (feat["lat"] > 37.4)
    & (feat["lat"] < 37.7)
    & (feat["lon"] > 126.7)
    & (feat["lon"] < 127.3)
]
print(f"유효 좌표 숙박업: {len(feat)}건")


# 사용승인일(8자리 숫자: YYYYMMDD) → 건축 연도(4자리 정수) 추출
def parse_year(v):
    """YYYYMMDD 형식의 사용승인일을 4자리 연도(int)로 변환. 실패 시 None."""
    try:
        s = str(int(float(v)))
        if len(s) == 8:
            return int(s[:4])
    except:
        pass
    return None


feat["_year"] = feat["사용승인일"].apply(parse_year)
valid = feat.dropna(subset=["_year"])
print(
    f"사용승인일 유효: {len(valid)}건, 범위: {int(valid['_year'].min())}~{int(valid['_year'].max())}"
)


# ── 2. 사상자 발생 화재 로드 ──────────────────────────────────────────
fire = pd.read_csv(
    os.path.join(BASE, "data", "화재출동", "화재출동_사상자발생.csv"),
    encoding="utf-8-sig",
    low_memory=False,
)
fire["위도"] = pd.to_numeric(fire["위도"], errors="coerce")
fire["경도"] = pd.to_numeric(fire["경도"], errors="coerce")
fire = fire.dropna(subset=["위도", "경도"])
# 서울 범위 필터
fire = fire[
    (fire["위도"] > 37.4)
    & (fire["위도"] < 37.7)
    & (fire["경도"] > 126.7)
    & (fire["경도"] < 127.3)
]
print(f"사상자 화재: {len(fire)}건")


# ── 3. 관광특구 (7개구 관련) ──────────────────────────────────────────
# 서울시 지정 관광특구 7개의 중심점 — 1km 반경 원으로 시각화
관광특구 = [
    {"name": "명동·남대문·북창 관광특구", "lat": 37.5635, "lon": 126.9826},
    {"name": "이태원 관광특구", "lat": 37.5344, "lon": 126.9946},
    {"name": "동대문 패션타운 관광특구", "lat": 37.5666, "lon": 127.0092},
    {"name": "종로·청계 관광특구", "lat": 37.5700, "lon": 126.9826},
    {"name": "잠실 관광특구", "lat": 37.5133, "lon": 127.1000},
    {"name": "강남 마이스 관광특구", "lat": 37.5117, "lon": 127.0590},
    {"name": "홍대 관광특구", "lat": 37.5563, "lon": 126.9238},
]


# ── 4. 사용승인일 → 색상 매핑 ────────────────────────────────────────
def year_to_color(year):
    """사용승인일 연도를 10년 단위 색상으로 매핑 — 1970년대 미만은 가장 짙은 색."""
    if year is None or (isinstance(year, float) and np.isnan(year)):
        return "#aaaaaa"  # 미상 — 회색
    year = int(year)
    if year < 1970:
        return "#2c3e50"  # 가장 오래된 건물 — 짙은 남색
    elif year < 1980:
        return "#8e44ad"  # 1970년대 — 보라
    elif year < 1990:
        return "#2980b9"  # 1980년대 — 파랑
    elif year < 2000:
        return "#27ae60"  # 1990년대 — 초록
    elif year < 2010:
        return "#f39c12"  # 2000년대 — 노랑
    elif year < 2020:
        return "#e67e22"  # 2010년대 — 주황
    else:
        return "#e74c3c"  # 2020년 이후 신축 — 빨강


# ── 5. 지도 생성 ──────────────────────────────────────────────────────
# CartoDB positron — 옅은 회색 베이스맵 (마커/원이 잘 보이도록)
m = folium.Map(location=[37.555, 126.977], zoom_start=12, tiles="CartoDB positron")

# 4개 레이어 그룹 — LayerControl 로 토글 가능
fg_tour = folium.FeatureGroup(name="🏖️ 관광특구 (1km)", show=True)
fg_hotel = folium.FeatureGroup(name="🏨 숙박업 (사용승인일)", show=True)
fg_fire = folium.FeatureGroup(name="🔥 사상자 발생 화재", show=True)
fg_heat = folium.FeatureGroup(name="🌡️ 화재 히트맵", show=False)  # 기본 비활성


# 관광특구 원 + 라벨
for t in 관광특구:
    # 1km 반투명 원 — fill_opacity 낮게 해서 안쪽 마커가 보이게
    folium.Circle(
        location=[t["lat"], t["lon"]],
        radius=1000,
        color="#e74c3c",
        fill=True,
        fill_color="#e74c3c",
        fill_opacity=0.12,
        weight=2,
        opacity=0.6,
        tooltip=t["name"],
    ).add_to(fg_tour)

    # 관광특구명 텍스트 — DivIcon 으로 마커 위치에 직접 라벨 표시
    folium.Marker(
        location=[t["lat"], t["lon"]],
        icon=folium.DivIcon(
            html=f'<div style="font-size:11px;font-weight:bold;color:#c0392b;'
            f'white-space:nowrap;text-shadow:1px 1px 2px white,-1px -1px 2px white">{t["name"]}</div>',
            icon_anchor=(0, 10),
        ),
    ).add_to(fg_tour)


# 숙박업 마커 (사용승인일 색상으로 시각화)
for _, row in feat.iterrows():
    year = row.get("_year")
    color = year_to_color(year)
    # 연도 표시 — 미상은 '미상' 으로
    year_str = (
        str(int(year))
        if (year and not (isinstance(year, float) and np.isnan(year)))
        else "미상"
    )
    # 팝업 — 주소/사용승인일/연면적/층수/주용도/구조 정보
    popup = (
        f"<b>{row.get('도로명대지위치', '')}</b><br>"
        f"사용승인일: {year_str}년<br>"
        f"연면적: {row.get('연면적(㎡)', '')}㎡<br>"
        f"지상층수: {row.get('지상층수', '')}층<br>"
        f"주용도: {row.get('주용도코드명', '')}<br>"
        f"구조: {row.get('구조코드명', '')}"
    )
    # 작은 원형 마커 — 반경 5px
    folium.CircleMarker(
        location=[row["lat"], row["lon"]],
        radius=5,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.8,
        weight=0.8,
        popup=folium.Popup(popup, max_width=280),
    ).add_to(fg_hotel)


# 화재 히트맵 — 사망자가 있는 화재는 가중치 2, 없으면 1
heat_data = [
    [r["위도"], r["경도"], 2 if r["사망자수"] >= 1 else 1] for _, r in fire.iterrows()
]
HeatMap(
    heat_data,
    radius=18,
    blur=14,
    max_zoom=14,
    gradient={0.2: "blue", 0.5: "lime", 0.8: "yellow", 1.0: "red"},
).add_to(fg_heat)


# 사상자 마커 — 사망 발생은 빨강, 부상만은 주황
for _, row in fire.iterrows():
    is_death = row["사망자수"] >= 1
    color = "#c0392b" if is_death else "#e67e22"
    # 마커 크기 — 사망자수 또는 부상자수에 비례 (최대 4명까지만 가산)
    size = (
        5 + int(row["사망자수"]) * 3 if is_death else 4 + min(int(row["부상자수"]), 4)
    )
    popup = (
        f"<b style='color:{color}'>사망 {int(row['사망자수'])}명 / 부상 {int(row['부상자수'])}명</b><br>"
        f"발생: {str(row.get('발생일자', ''))[:10]}<br>"
        f"장소: {row.get('발화장소_대분류', '')} &gt; {row.get('발화장소_소분류', '')}<br>"
        f"구: {row.get('발생시군구', '')}"
    )
    folium.CircleMarker(
        location=[row["위도"], row["경도"]],
        radius=size,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.85,
        weight=1.5,
        popup=folium.Popup(popup, max_width=280),
    ).add_to(fg_fire)


# 4개 그룹을 지도에 추가 + 레이어 컨트롤
fg_tour.add_to(m)
fg_hotel.add_to(m)
fg_fire.add_to(m)
fg_heat.add_to(m)
folium.LayerControl(collapsed=False).add_to(m)


# ── 6. 범례 ──────────────────────────────────────────────────────────
# 사용승인일 색상 범례 데이터 — 색상 / 라벨
yr_legend = [
    ("#2c3e50", "~1969"),
    ("#8e44ad", "1970s"),
    ("#2980b9", "1980s"),
    ("#27ae60", "1990s"),
    ("#f39c12", "2000s"),
    ("#e67e22", "2010s"),
    ("#e74c3c", "2020~"),
    ("#aaaaaa", "미상"),
]

# 색상 점 + 라벨을 한 줄에 가로로 늘어놓는 HTML
yr_rows = "".join(
    [
        f'<span style="display:inline-block;width:12px;height:12px;background:{c};border-radius:50%;margin-right:4px"></span>'
        f'<span style="font-size:11px;margin-right:8px">{l}</span>'
        for c, l in yr_legend
    ]
)

# 10년 단위 카운트 (참고용 — 향후 범례 확장 시 사용 가능)
cnt_year = Counter(
    valid["_year"].apply(lambda y: f"{int(y) // 10 * 10}s" if y else "미상")
)


# 좌하단 고정 범례 박스 — 숙박업/화재/관광특구 안내
legend_html = f"""
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
</div>"""

m.get_root().html.add_child(folium.Element(legend_html))


# 최종 HTML 저장
out_html = os.path.join(BASE, "data", "화재출동", "숙박업_사용승인일_사상자_지도.html")
m.save(out_html)
print(f"저장: {out_html}")
