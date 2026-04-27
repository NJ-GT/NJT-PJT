# -*- coding: utf-8 -*-
import streamlit as st
import streamlit.components.v1 as components
import os

st.set_page_config(page_title="지도 아카이브", page_icon="🌐", layout="wide")
st.title("🌐 분석 지도 아카이브")
st.caption("기존 생성된 인터랙티브 지도 파일 — 레이어 토글, 줌 등 직접 조작 가능")

MAPS = {
    "🏨 소방서 · 숙박시설 통합 지도": "data/Map_Seoul10_Firestation.html",
    "🔥 화재 KDE 밀도 지도":          "data/Map_Fire_KDE.html",
    "📍 소방거리 버퍼 KDE":            "data/Map_Buffer_KDE.html",
    "🚒 소방서 위치 (2025)":           "data/Map_Firestation_2025.html",
}

sel = st.radio("지도 선택", list(MAPS.keys()), horizontal=True)
fpath = MAPS[sel]

if not os.path.exists(fpath):
    st.error(f"파일 없음: {fpath}")
    st.stop()

size_mb = os.path.getsize(fpath) / 1024 / 1024
st.caption(f"파일 크기: {size_mb:.1f} MB — 로딩에 잠시 시간이 걸릴 수 있습니다.")

with open(fpath, 'r', encoding='utf-8') as f:
    html = f.read()

components.html(html, height=680, scrolling=False)
