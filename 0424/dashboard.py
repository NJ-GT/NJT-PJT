# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import plotly.express as px
import glob

st.set_page_config(
    page_title="서울 숙박시설 화재위험 대쉬보드",
    page_icon="🏨",
    layout="wide",
)

# ── 전역 스타일 ──
st.markdown("""
<style>
/* 사이드바 너비 */
[data-testid="stSidebar"] { min-width: 220px; max-width: 260px; }
/* 메트릭 카드 */
[data-testid="metric-container"] {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 14px 18px;
}
/* 구분선 여백 */
hr { margin: 1rem 0 !important; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_main():
    f = glob.glob('data/*0423*.csv')[0]
    return pd.read_csv(f, encoding='utf-8-sig')

@st.cache_data
def load_core():
    return pd.read_csv('data/핵심서울0424.csv', encoding='utf-8-sig')

df   = load_main()
core = load_core()

# ── 헤더 ──
st.title("서울 숙박시설 화재위험 분석 대쉬보드")
st.markdown(
    "<p style='color:#64748B; margin-top:-12px;'>"
    "외국인관광도시민박업 · 관광숙박업 · 숙박업 &nbsp;|&nbsp; 서울 10개구 &nbsp;|&nbsp; 2020–2025</p>",
    unsafe_allow_html=True,
)

st.markdown("---")

# ── 핵심 메트릭 ──
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("총 분석 시설",    f"{len(df):,}개")
c2.metric("분석 구역",       f"{df['구'].nunique()}개구")
c3.metric("외국인민박 비중", f"{df['업종'].value_counts(normalize=True).get('외국인관광도시민박업',0)*100:.1f}%")
c4.metric("군집 수",         f"{core['군집'].nunique()}개")
c5.metric("GWR R²",         "0.40", help="Y=실제화재수 기준 bisquare GWR")

st.markdown("---")

# ── 차트 ──
col_a, col_b = st.columns([1, 1])

with col_a:
    st.subheader("업종별 시설 분포")
    cnt = df['업종'].value_counts().reset_index()
    cnt.columns = ['업종', '수']
    fig = px.pie(
        cnt, names='업종', values='수',
        color_discrete_sequence=['#2563EB', '#10B981', '#F59E0B'],
        hole=0.5,
    )
    fig.update_traces(textposition='outside', textinfo='percent+label')
    fig.update_layout(
        margin=dict(t=10, b=10, l=0, r=0),
        legend=dict(orientation='h', y=-0.12),
        paper_bgcolor='white', plot_bgcolor='white',
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

with col_b:
    st.subheader("구별 시설 수")
    gu = df['구'].value_counts().reset_index()
    gu.columns = ['구', '수']
    fig2 = px.bar(
        gu.sort_values('수'), x='수', y='구', orientation='h',
        color='수', color_continuous_scale=['#BFDBFE', '#1D4ED8'],
        text='수',
    )
    fig2.update_traces(textposition='outside')
    fig2.update_layout(
        margin=dict(t=10, b=10, l=0, r=20),
        coloraxis_showscale=False,
        yaxis_title=None, xaxis_title='시설 수',
        paper_bgcolor='white', plot_bgcolor='#FAFAFA',
    )
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# ── 분석 체인 안내 ──
st.subheader("분석 구조")
cols = st.columns(5)
steps = [
    ("1_지도",        "3D 위험도 지도",      "AHP 위험점수 히트맵 + 군집 색상"),
    ("6_위험순위",    "위험 순위 & 골든타임", "상위 N개 시설 순위 + 소방도달시간"),
    ("9_GWR",         "GWR 공간분석",        "지역별 위험계수 지도 (R²=0.40)"),
    ("11_공간분석",   "공간분석 근거",        "Moran's I → Spatial Lag → GWR 체인"),
    ("10_분석보고서", "분석 보고서",          "OLS·PCA·LISA 등 전체 시각화 아카이브"),
]
for col, (_, title, desc) in zip(cols, steps):
    with col:
        st.markdown(
            f"<div style='background:#F1F5F9; border-radius:10px; padding:14px; "
            f"min-height:100px;'>"
            f"<b style='color:#1E293B'>{title}</b><br>"
            f"<span style='color:#64748B; font-size:0.85rem'>{desc}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

st.markdown("<br><p style='color:#94A3B8; text-align:center'>👈 왼쪽 사이드바에서 분석 페이지를 선택하세요</p>", unsafe_allow_html=True)
