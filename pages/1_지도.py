# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import pydeck as pdk

st.set_page_config(page_title="위험도 지도", page_icon="🗺️", layout="wide")
st.title("🗺️ 위험도 지도")

pdk.settings.mapbox_api_key = st.secrets["mapbox"]["token"]

CLUSTER_COLORS = {
    0: [45,  91,  227],
    1: [0,   196, 159],
    2: [255, 107, 107],
    3: [255, 209, 102],
    4: [168, 85,  247],
}
DEFAULT_COLOR = [150, 150, 150]

@st.cache_data
def load_core():
    df = pd.read_csv('data/핵심서울0424.csv', encoding='utf-8-sig')
    df = df.dropna(subset=['위도', '경도']).copy()
    mn, mx = df['위험점수_AHP'].min(), df['위험점수_AHP'].max()
    df['ahp_norm'] = (df['위험점수_AHP'] - mn) / (mx - mn + 1e-9)
    df['군집_int'] = df['군집'].fillna(-1).astype(int)
    for i, ch in enumerate(['r', 'g', 'b']):
        df[ch] = df['군집_int'].map(lambda x, i=i: CLUSTER_COLORS.get(x, DEFAULT_COLOR)[i])
    return df

@st.cache_data
def load_stations():
    return pd.read_csv(
        'data/소방서_안전센터_구조대_위치정보_2025_wgs84.csv', encoding='utf-8-sig'
    ).dropna(subset=['위도', '경도'])

core     = load_core()
stations = load_stations()

st.sidebar.header("레이어 & 필터")
all_gu   = sorted(core['구'].dropna().unique())
sel_gu   = st.sidebar.multiselect("구 선택", all_gu, default=all_gu)
show_scatter  = st.sidebar.checkbox("숙소 위치 (군집색)", value=True)
show_heat     = st.sidebar.checkbox("AHP 위험도 히트맵",  value=True)
show_station  = st.sidebar.checkbox("소방서 / 안전센터",  value=True)
map_style_key = st.sidebar.selectbox(
    "지도 스타일",
    ["dark-v10", "light-v10", "satellite-v9"],
    format_func=lambda x: {"dark-v10": "다크", "light-v10": "라이트", "satellite-v9": "위성"}[x],
)

filt = core[core['구'].isin(sel_gu)] if sel_gu else core

layers = []
if show_heat and len(filt):
    layers.append(pdk.Layer(
        "HeatmapLayer", data=filt,
        get_position=["경도", "위도"],
        get_weight="ahp_norm",
        opacity=0.75, threshold=0.05, radiusPixels=40,
    ))
if show_scatter and len(filt):
    layers.append(pdk.Layer(
        "ScatterplotLayer", data=filt,
        get_position=["경도", "위도"],
        get_color=["r", "g", "b", 200],
        get_radius=25,
        pickable=True, auto_highlight=True,
    ))
if show_station:
    layers.append(pdk.Layer(
        "ScatterplotLayer", data=stations,
        get_position=["경도", "위도"],
        get_color=[255, 40, 40, 230],
        get_radius=80, pickable=True,
    ))

view = pdk.ViewState(
    latitude=filt['위도'].mean()  if len(filt) else 37.53,
    longitude=filt['경도'].mean() if len(filt) else 126.98,
    zoom=11, pitch=45,
)

st.pydeck_chart(pdk.Deck(
    layers=layers,
    initial_view_state=view,
    map_style=f"mapbox://styles/mapbox/{map_style_key}",
    tooltip={"text": "{업소명}\n{구} {동}\nAHP 위험점수: {위험점수_AHP}\n소방도달: {이동시간초}초"},
))

st.markdown("**군집 색상 범례**")
leg_cols = st.columns(len(CLUSTER_COLORS) + 1)
for i, (k, rgb) in enumerate(CLUSTER_COLORS.items()):
    hex_c = '#{:02X}{:02X}{:02X}'.format(*rgb)
    leg_cols[i].markdown(
        f'<span style="color:{hex_c};font-size:18px">■</span> 군집 {k}',
        unsafe_allow_html=True,
    )
leg_cols[-1].markdown('<span style="color:#FF2828;font-size:18px">■</span> 소방서', unsafe_allow_html=True)

if len(filt):
    st.markdown("---")
    m1, m2, m3 = st.columns(3)
    m1.metric("선택 시설 수",       f"{len(filt):,}개")
    m2.metric("평균 AHP 위험점수",  f"{filt['위험점수_AHP'].mean():.3f}")
    m3.metric("평균 소방도달(초)",  f"{filt['이동시간초'].mean():.0f}초")
