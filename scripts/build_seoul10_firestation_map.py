# -*- coding: utf-8 -*-
"""
서울 10개구 숙박시설 ↔ 가장 가까운 소방서/안전센터 거리(Haversine) + Folium 지도 생성.

목적:
    - 통합숙박시설최종안0415 에서 10개구 숙박시설 추출
    - 안전센터/소방서별 최근접 거리·예상 도착시간(30km/h 가정) 계산
    - 도로명/지번주소에서 동 추출 후 관할구역 매칭으로 담당 기관 표기
    - CSV + Folium HTML 두 가지 산출물 저장

입력:
    - data/통합숙박시설최종안0415.csv (전체 숙박시설)
    - data/소방서_안전센터_구조대_위치정보_2025_wgs84.csv (소방시설 좌표/관할구역)

출력:
    - data/서울10구_숙소_소방거리_유클리드.csv   (행 단위 결과)
    - data/Map_Seoul10_Firestation.html           (인터랙티브 지도)

처리 흐름:
    1) 숙박시설 CSV 로드 → 컬럼명 위치 기반 지정 → 10개구만 필터 + 동 추출
    2) 소방시설 CSV 로드, 시설유형 분리(안전센터/구조대 vs 소방서)
    3) Haversine 으로 시설별 최근접 안전센터/소방서 거리 계산
    4) 관할구역 문자열 매칭으로 담당 안전센터/소방서 도출
    5) CSV 저장 + 구별 거리 통계 출력
    6) Folium 지도: 구별 FeatureGroup + 소방시설 마커 + 범례 HTML
"""
import os
import numpy as np
import pandas as pd
import folium

# 가정 상수: 출동 평균속도 30km/h, 거리 등급 임계
SPEED_KMH = 30
SAFE_DIST = 1000  # 안전 (~3분)
GOLDEN_DIST = 2000  # 골든타임 경계 (~5분)
DANGER_DIST = 3000  # 위험 (~7분)

# 경로 상수 (스크립트 위치 기준)
BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
LODGING = os.path.join(BASE, "data", "통합숙박시설최종안0415.csv")
FIRE_CSV = os.path.join(BASE, "data", "소방서_안전센터_구조대_위치정보_2025_wgs84.csv")
OUT_CSV = os.path.join(BASE, "data", "서울10구_숙소_소방거리_유클리드.csv")
OUT_HTML = os.path.join(BASE, "data", "Map_Seoul10_Firestation.html")

# 분석 대상 10개구
GUGUS = [
    "마포구",
    "중구",
    "종로구",
    "용산구",
    "강남구",
    "영등포구",
    "강서구",
    "송파구",
    "성동구",
    "서초구",
]

# ── 1. 숙박시설 로드 & 필터 ──────────────────────────────────────────
df_raw = pd.read_csv(LODGING, encoding="utf-8-sig")
# CSV 의 헤더 인코딩 깨짐 대비, 위치 기반 컬럼명을 직접 지정
df_raw.columns = [
    "X좌표",
    "Y좌표",
    "위도",
    "경도",
    "도로명소재지",
    "시군구코드",
    "행정동코드",
    "법정동코드",
    "층",
    "호",
    "업소명",
    "건축물대장PK",
    "지번소재지",
    "연면적",
    "건폐율",
    "사용코드",
    "기타용도",
    "주용도코드",
    "기타용도2",
    "지상층수",
    "지하층수",
    "건물층수",
    "사용승인일",
    "증가일",
    "상용층수",
    "판매층수",
    "사용연한여부",
    "소방청_매칭결과",
    "소방청_스프링클러설치",
    "소방청_미이소클러설치",
    "소방청_건물명",
    "소방청_도로명주소_매칭",
    "소방청_지번주소_매칭",
    "소방청_매칭점수",
    "소방청_매칭방법",
    "면적",
    "건폐율2",
]

# 도로명소재지에서 'OO구' 정규식 추출 → 10개구만 유지
addr = df_raw["도로명소재지"].fillna("")
df_raw["구"] = addr.str.extract(r"서울특별시\s+(\S+구)")[0]
target = df_raw[df_raw["구"].isin(GUGUS)].copy().reset_index(drop=True)

# 동 추출 (관할구역 매칭용)
target["동"] = target["도로명소재지"].fillna("").str.extract(r"(\S+동)\s")[0]
# 도로명에 동이 없으면 지번소재지에서 추출
target["동"].fillna(
    target["지번소재지"].fillna("").str.extract(r"(\S+동)")[0], inplace=True
)

print(f"대상 숙박시설: {len(target)}개")
print(target["구"].value_counts().to_string())

# ── 2. 소방시설 로드 ──────────────────────────────────────────────────
fire = pd.read_csv(FIRE_CSV, encoding="utf-8-sig")
# 안전센터/구조대 vs 소방서 분리 (최근접 매칭 따로)
fire_sc = fire[fire["시설유형"] == "안전센터/구조대"].reset_index(drop=True)
fire_fs = fire[fire["시설유형"] == "소방서"].reset_index(drop=True)

# ── 3. 유클리드 거리 (Haversine) ──────────────────────────────────────
R = 6371000  # 지구 반지름 m


def nearest_euclidean(lats, lons, station_df):
    """각 (lat, lon) 점에서 가장 가까운 station 의 (인덱스, 거리m) 리스트 반환."""
    s_lats = np.radians(station_df["위도"].values)
    s_lons = np.radians(station_df["경도"].values)
    results = []
    for lat, lon in zip(lats, lons):
        dlat = s_lats - np.radians(lat)
        dlon = s_lons - np.radians(lon)
        a = (
            np.sin(dlat / 2) ** 2
            + np.cos(np.radians(lat)) * np.cos(s_lats) * np.sin(dlon / 2) ** 2
        )
        dists = R * 2 * np.arcsin(np.sqrt(a))
        idx = dists.argmin()
        results.append((idx, round(dists[idx])))
    return results


# 모든 숙박시설에 대해 안전센터/소방서 최근접 매칭
nearest_sc = nearest_euclidean(target["위도"], target["경도"], fire_sc)
nearest_fs = nearest_euclidean(target["위도"], target["경도"], fire_fs)


# ── 4. 관할구역 매칭 ──────────────────────────────────────────────────
def find_responsible(gu_kw, dong_kw, ftype):
    """관할구역 문자열에 (구, 동) 둘 다 포함된 시설명을 ' / ' 로 결합해 반환."""
    if not dong_kw or pd.isna(dong_kw):
        return None
    mask = (
        (fire["시설유형"] == ftype)
        & fire["관할구역"].fillna("").str.contains(str(gu_kw), regex=False)
        & fire["관할구역"].fillna("").str.contains(str(dong_kw), regex=False)
    )
    hits = fire[mask]
    return " / ".join(hits["시설명"].tolist()) if not hits.empty else None


# ── 5. 결과 DataFrame ─────────────────────────────────────────────────
rows = []
for i, row in target.iterrows():
    sc_idx, sc_dist = nearest_sc[i]
    fs_idx, fs_dist = nearest_fs[i]
    sc = fire_sc.iloc[sc_idx]
    fs = fire_fs.iloc[fs_idx]
    # 두 시설 중 가까운 거리로 출동 시간 산출
    best = min(sc_dist, fs_dist)
    move_sec = best / (SPEED_KMH / 3.6)
    # 출동 준비 60초 + 이동시간 → 분 단위(소수1)
    total_min = round((60 + move_sec) / 60, 1)
    rows.append(
        {
            "구": row["구"],
            "동": row["동"],
            "업소명": row["업소명"],
            "주소": row["도로명소재지"],
            "위도": row["위도"],
            "경도": row["경도"],
            "최근접_안전센터": sc["시설명"],
            "안전센터_유클리드m": sc_dist,
            "최근접_소방서": fs["시설명"],
            "소방서_유클리드m": fs_dist,
            "최근접_거리m": best,
            "이동시간초": round(move_sec),
            "예상도착분": total_min,
            "담당_안전센터": find_responsible(row["구"], row["동"], "안전센터/구조대"),
            "담당_소방서": find_responsible(row["구"], row["동"], "소방서"),
        }
    )

result_df = pd.DataFrame(rows)
result_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

# ── 6. 구별 통계 출력 ────────────────────────────────────────────────
stats = (
    result_df.groupby("구")["최근접_거리m"]
    .agg(["mean", "min", "max", "count"])
    .round(0)
    .astype(int)
)
print("\n=== 구별 평균 유클리드 거리 ===")
print(stats.to_string())
# 거리 등급별 카운트 (초록/주황/빨강/진빨강)
d = result_df["최근접_거리m"]
g = (d <= SAFE_DIST).sum()
o = ((d > SAFE_DIST) & (d <= GOLDEN_DIST)).sum()
r_ = ((d > GOLDEN_DIST) & (d <= DANGER_DIST)).sum()
rr = (d > DANGER_DIST).sum()
print(f"\n전체 {len(result_df)}개 | 평균 {d.mean():.0f}m")
print(
    f"초록(≤{SAFE_DIST}m): {g}개 | 주황(≤{GOLDEN_DIST}m): {o}개 | 빨강(≤{DANGER_DIST}m): {r_}개 | 진빨강: {rr}개"
)


# ── 7. 지도 ──────────────────────────────────────────────────────────
def marker_color(dist):
    """거리 등급별 색 (초록/주황/빨강/진빨강)."""
    if dist <= SAFE_DIST:
        return "#27ae60"
    if dist <= GOLDEN_DIST:
        return "#e67e22"
    if dist <= DANGER_DIST:
        return "#c0392b"
    return "#7f0000"


def fmt_sec(sec):
    """초 → 'M분 S초' 포맷."""
    return f"{int(sec) // 60}분 {int(sec) % 60}초"


# 서울 중심부 기준 지도
m = folium.Map(location=[37.555, 126.970], zoom_start=12, tiles="CartoDB positron")

# 구별 FeatureGroup 으로 묶어 토글 가능하게
for gu in GUGUS:
    sub = result_df[result_df["구"] == gu]
    if sub.empty:
        continue
    avg = sub["최근접_거리m"].mean()
    fg = folium.FeatureGroup(name=f"{gu} ({len(sub)}개 · 평균{avg:.0f}m)", show=True)
    for _, r in sub.iterrows():
        dv = r["최근접_거리m"]
        move_sec = r["이동시간초"]
        total_min = r["예상도착분"]
        c = marker_color(dv)
        # 담당 기관 우선, 없으면 최근접 기관 표시
        담당sc = (
            r["담당_안전센터"] if pd.notna(r["담당_안전센터"]) else r["최근접_안전센터"]
        )
        담당fs = r["담당_소방서"] if pd.notna(r["담당_소방서"]) else r["최근접_소방서"]
        # 팝업 HTML
        popup = (
            f"<div style='font-family:Malgun Gothic;width:270px'>"
            f"<b>[{r['구']}] {r['업소명']}</b>"
            f"<hr style='margin:4px 0'>"
            f"<b style='color:{c}'>최근접 거리: {dv:.0f}m<br>"
            f"이동: {fmt_sec(move_sec)} → 총 {total_min}분</b>"
            f"<hr style='margin:4px 0'>"
            f"<span style='color:#e67e22'>&#128680; 담당 안전센터:</span> <b>{담당sc}</b><br>"
            f"<span style='color:#c0392b'>&#128658; 담당 소방서:</span> <b>{담당fs}</b>"
            f"<hr style='margin:4px 0'>"
            f"<small>최근접 안전센터: {r['최근접_안전센터']} {r['안전센터_유클리드m']}m<br>"
            f"최근접 소방서: {r['최근접_소방서']} {r['소방서_유클리드m']}m</small>"
            f"</div>"
        )
        folium.CircleMarker(
            [r["위도"], r["경도"]],
            radius=5,
            color=c,
            fill=True,
            fill_color=c,
            fill_opacity=0.8,
            weight=1.2,
            popup=folium.Popup(popup, max_width=290),
            tooltip=f"[{r['구']}] {r['업소명']} | {dv:.0f}m",
        ).add_to(fg)
    fg.add_to(m)

# 소방시설 마커 (소방서=빨강 / 안전센터=주황)
fg_fire = folium.FeatureGroup(name="소방시설", show=True)
for _, r in fire.iterrows():
    ic = "red" if r["시설유형"] == "소방서" else "orange"
    # 관할구역 텍스트가 너무 길면 60자로 자름
    district = (
        (str(r["관할구역"])[:60] + "...")
        if pd.notna(r["관할구역"]) and len(str(r["관할구역"])) > 60
        else (str(r["관할구역"]) if pd.notna(r["관할구역"]) else "-")
    )
    popup_html = (
        f"<div style='font-family:Malgun Gothic;width:250px'>"
        f"<b>{r['시설명']}</b> ({r['시설유형']})"
        f"<hr style='margin:4px 0'>"
        f"<small>관할: {district}</small>"
        f"</div>"
    )
    folium.Marker(
        [r["위도"], r["경도"]],
        tooltip=f"{r['시설명']} ({r['시설유형']})",
        popup=folium.Popup(popup_html, max_width=260),
        icon=folium.Icon(color=ic, icon="fire", prefix="fa"),
    ).add_to(fg_fire)
fg_fire.add_to(m)

folium.LayerControl(collapsed=False).add_to(m)

# 구별 통계 HTML 조각 (범례 안에 삽입)
gu_stats = result_df.groupby("구")["최근접_거리m"].agg(["mean", "count"])
gu_stats_html = "".join(
    f"<b>{gu}:</b> 평균 {int(gu_stats.loc[gu, 'mean'])}m ({int(gu_stats.loc[gu, 'count'])}개)<br>"
    for gu in GUGUS
    if gu in gu_stats.index
)

# 범례 HTML
legend_html = f"""
<div style="position:fixed;bottom:30px;left:30px;z-index:1000;
            background:white;padding:14px 18px;border-radius:10px;
            box-shadow:2px 2px 8px rgba(0,0,0,.3);font-size:12px;
            font-family:'Malgun Gothic',sans-serif;min-width:280px">
  <b>서울 10개구 숙소 ↔ 소방시설 (유클리드)</b>
  <hr style='margin:8px 0'>
  {gu_stats_html}
  <hr style='margin:8px 0'>
  <span style='color:#27ae60'>●</span> ~{SAFE_DIST}m · 총3분 이내 (안전): <b>{g}개</b><br>
  <span style='color:#e67e22'>●</span> ~{GOLDEN_DIST}m · 총5분 골든타임 경계: <b>{o}개</b><br>
  <span style='color:#c0392b'>●</span> ~{DANGER_DIST}m · 총7분 (위험): <b>{r_}개</b><br>
  <span style='color:#7f0000'>●</span> {DANGER_DIST}m+ · 7분 초과 (매우위험): <b>{rr}개</b><br>
  <hr style='margin:8px 0'>
  전체 <b>{len(result_df)}</b>개 | 평균 <b>{d.mean():.0f}m</b><br>
  <small>마커 클릭 시 담당 안전센터·소방서 표시<br>
  유클리드 거리 · 소방차 30km/h · 골든타임 5분 기준</small>
</div>"""
m.get_root().html.add_child(folium.Element(legend_html))
m.save(OUT_HTML)
print(f"\n[저장] {OUT_HTML}")
