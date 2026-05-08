# -*- coding: utf-8 -*-
"""
버퍼(50m) 건물수 기반 KDE(가우시안 커널 밀도) 맵을 재생성하는 스크립트.

목적:
    - 숙박시설 좌표를 표본으로, 반경 50m 건물수를 가중치로 KDE 곡면을 생성한다.
    - Folium 지도 위에 ① KDE 곡면 이미지, ② 동적 HeatMap, ③ 개별 포인트 3개 레이어를 올린다.

입력:
    - NJT-PJT/data/서울10구_숙소_소방거리_유클리드.csv (위경도, 반경_50m_건물수, 노후도/소방위험도/단속수 등)

출력:
    - NJT-PJT/data/Map_Buffer_KDE.html

처리 흐름:
    1) CSV 로딩 → 위경도 결측/이상치 제거 (서울 범위로 클립)
    2) 가우시안 KDE 계산 (반경_50m_건물수 가중치 적용, bw=0.04)
    3) 200×200 그리드 위에 KDE 평가 → 0~1 정규화
    4) Matplotlib 으로 KDE 등고면 PNG → base64 인코딩
    5) Folium 지도에 KDE 오버레이 + HeatMap + CircleMarker 레이어 추가
    6) 범례 HTML 추가 후 HTML 파일로 저장
"""
import base64
import sys
from io import BytesIO

# 수치/데이터프레임
import numpy as np
import pandas as pd
# Matplotlib (PNG 생성, GUI 미사용)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
# 가우시안 KDE
from scipy.stats import gaussian_kde
# 인터랙티브 지도
import folium
from folium.plugins import HeatMap

# 콘솔 UTF-8 출력
sys.stdout.reconfigure(encoding="utf-8")

# 경로 상수
BASE = "c:/Users/USER/Documents/GitHub/기말공모전/NJT-PJT"
INPUT_PATH = f"{BASE}/data/서울10구_숙소_소방거리_유클리드.csv"
OUTPUT_PATH = f"{BASE}/data/Map_Buffer_KDE.html"

# CSV 로딩
df = pd.read_csv(INPUT_PATH, encoding="utf-8-sig")
# KDE 계산에 필요한 컬럼 결측 제거
df = df.dropna(subset=["위도", "경도", "반경_50m_건물수"])
# 서울 영역 클립 (위경도 이상치 제거)
df = df[
    (df["위도"] > 37.4)
    & (df["위도"] < 37.7)
    & (df["경도"] > 126.7)
    & (df["경도"] < 127.3)
]
print(f"숙박시설: {len(df)}개")

# ── KDE 계산 ─────────────────────────────────────────────────────────
# 가중치(반경 50m 건물수) → 합 0이면 균등 가중치
weights = df["반경_50m_건물수"].values.astype(float)
if weights.sum() == 0:
    weights = np.ones(len(df))
# 가중치 정규화 (합=1)
weights = weights / weights.sum()

# (경도, 위도) 좌표 행렬
xy = np.vstack([df["경도"].values, df["위도"].values])
# 가우시안 KDE (bandwidth scott/silverman 대신 고정 0.04 사용 → 더 세밀한 곡면)
kde = gaussian_kde(xy, weights=weights, bw_method=0.04)

# KDE 평가용 그리드 범위 (입력 좌표 범위 ± 0.02도)
lat_min = df["위도"].min() - 0.02
lat_max = df["위도"].max() + 0.02
lon_min = df["경도"].min() - 0.02
lon_max = df["경도"].max() + 0.02

# 200×200 격자 생성
GRID_N = 200
grid_lon, grid_lat = np.mgrid[
    lon_min : lon_max : GRID_N * 1j, lat_min : lat_max : GRID_N * 1j
]
# 각 격자점에서 KDE 값 평가 후 (200,200) 행렬로 reshape
kde_values = kde(np.vstack([grid_lon.ravel(), grid_lat.ravel()])).reshape(
    GRID_N, GRID_N
)
# 0~1 정규화
kde_norm = (kde_values - kde_values.min()) / (kde_values.max() - kde_values.min())

# ── KDE 이미지 생성 ───────────────────────────────────────────────────
# 정사각 캔버스에 KDE 등고면 그리기
fig, ax = plt.subplots(figsize=(10, 10), dpi=150)
# 노랑→빨강 컬러맵
cmap = plt.get_cmap("YlOrRd")
# 알파 채널을 추가해 낮은 밀도 영역은 투명하게 (지도 베이스를 보이게)
cmap_alpha = cmap(np.linspace(0, 1, 256))
cmap_alpha[:25, 3] = 0  # 하위 ~10% 완전 투명
cmap_alpha[25:50, 3] = np.linspace(0, 0.4, 25)  # 점진적 페이드인
custom_cmap = mcolors.LinearSegmentedColormap.from_list("YlOrRd_alpha", cmap_alpha)
# 등고면(contourf) 으로 그리기
ax.contourf(grid_lon, grid_lat, kde_norm, levels=20, cmap=custom_cmap, alpha=0.85)
# 좌표 범위 고정
ax.set_xlim(lon_min, lon_max)
ax.set_ylim(lat_min, lat_max)
# 축/배경 모두 투명
ax.axis("off")
fig.patch.set_alpha(0)
ax.patch.set_alpha(0)

# PNG → base64 인코딩 (Folium ImageOverlay 에 dataURL 로 삽입)
buf = BytesIO()
fig.savefig(
    buf, format="png", bbox_inches="tight", pad_inches=0, transparent=True, dpi=150
)
buf.seek(0)
img_b64 = base64.b64encode(buf.read()).decode("utf-8")
plt.close(fig)
print("KDE 이미지 생성 완료")

# ── Folium 지도 ───────────────────────────────────────────────────────
# 지도 중심: 그리드 중심점, CartoDB Positron 타일
m = folium.Map(
    location=[(lat_min + lat_max) / 2, (lon_min + lon_max) / 2],
    zoom_start=12,
    tiles="CartoDB positron",
)

# 레이어 1: KDE 곡면 (PNG 오버레이)
fg_kde = folium.FeatureGroup(name="KDE 건물 밀집도 곡면", show=True)
folium.raster_layers.ImageOverlay(
    image=f"data:image/png;base64,{img_b64}",
    bounds=[[lat_min, lon_min], [lat_max, lon_max]],
    opacity=0.75,
).add_to(fg_kde)
fg_kde.add_to(m)

# 레이어 2: HeatMap (Folium 내장 동적 KDE) — 비교용, 기본 숨김
fg_heat = folium.FeatureGroup(name="HeatMap (동적 KDE)", show=False)
heat_data = [
    [r["위도"], r["경도"], float(r["반경_50m_건물수"])]
    for _, r in df.iterrows()
    if r["반경_50m_건물수"] > 0
]
HeatMap(
    heat_data,
    radius=20,
    blur=25,
    max_zoom=15,
    gradient={0.2: "blue", 0.4: "lime", 0.6: "yellow", 0.8: "orange", 1.0: "red"},
).add_to(fg_heat)
fg_heat.add_to(m)

# 레이어 3: 개별 포인트 (CircleMarker, 기본 숨김)
fg_pts = folium.FeatureGroup(name="숙박시설 개별 포인트", show=False)
for _, r in df.iterrows():
    cnt = int(r["반경_50m_건물수"])
    # 건물수 구간별 색상: >30 빨강, 15~30 주황, <15 초록
    color = "#c0392b" if cnt > 30 else ("#e67e22" if cnt >= 15 else "#27ae60")
    # 팝업 HTML (이름/주변 정보 요약)
    popup_html = (
        f"<div style='font-family:Malgun Gothic;width:200px'>"
        f"<b>[{r['구']}] {r['업소명']}</b><hr style='margin:4px 0'>"
        f"반경 50m 건물수: <b>{cnt}개</b><br>"
        f"집중도: {r['집중도(%)']:.1f}%<br>"
        f"노후도 점수: {r['노후도_점수']:.3f}<br>"
        f"소방위험도 점수: {r['소방위험도_점수']:.3f}<br>"
        f"불법주정차 단속수: {int(r['불법주정차_단속수_50m']) if pd.notna(r.get('불법주정차_단속수_50m', None)) else '-'}"
        f"</div>"
    )
    folium.CircleMarker(
        [r["위도"], r["경도"]],
        radius=4,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.85,
        weight=0.8,
        popup=folium.Popup(popup_html, max_width=220),
        tooltip=f"[{r['구']}] {r['업소명']} | 주변건물 {cnt}개",
    ).add_to(fg_pts)
fg_pts.add_to(m)

# 레이어 컨트롤 (펼친 상태)
folium.LayerControl(collapsed=False).add_to(m)

# ── 범례 HTML ─────────────────────────────────────────────────────────
legend_html = f"""
<div style="position:fixed;bottom:30px;left:30px;z-index:1000;
            background:white;padding:14px 18px;border-radius:10px;
            box-shadow:2px 2px 8px rgba(0,0,0,.3);font-size:12px;
            font-family:'Malgun Gothic',sans-serif;min-width:240px">
  <b>버퍼(50m) + KDE 건물 밀집도</b>
  <hr style="margin:8px 0">
  <b>KDE 밀도 (곡면 레이어)</b>
  <div style="width:200px;height:14px;
              background:linear-gradient(to right,rgba(255,255,204,0.2),#fecc5c,#fd8d3c,#e31a1c);
              border-radius:3px;margin:5px 0 2px"></div>
  <div style="display:flex;justify-content:space-between;font-size:10px;width:200px">
    <span>낮은 밀도</span><span>높은 밀도</span>
  </div>
  <hr style="margin:8px 0">
  <b>개별 포인트</b><br>
  <span style="color:#c0392b">●</span> &gt;30개 &nbsp;
  <span style="color:#e67e22">●</span> 15~30개 &nbsp;
  <span style="color:#27ae60">●</span> &lt;15개<br>
  <hr style="margin:8px 0">
  <small>총 {len(df):,}개 숙박시설 | 서울 10개구</small>
</div>"""
m.get_root().html.add_child(folium.Element(legend_html))

# 최종 HTML 저장
m.save(OUTPUT_PATH)
print(f"[저장 완료] {OUTPUT_PATH}")
