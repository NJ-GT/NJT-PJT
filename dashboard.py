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

@st.cache_data
def load_main():
    f = glob.glob('data/*0423*.csv')[0]
    return pd.read_csv(f, encoding='utf-8-sig')

@st.cache_data
def load_core():
    return pd.read_csv('data/핵심서울0424.csv', encoding='utf-8-sig')

df   = load_main()
core = load_core()

st.title("🏨 서울 숙박시설 화재위험 분석 대쉬보드")
st.caption("외국인관광도시민박업 · 관광숙박업 · 숙박업  |  서울 10개구  |  2020–2025")
st.markdown("---")

c1, c2, c3, c4 = st.columns(4)
c1.metric("총 분석 시설",   f"{len(df):,}개")
c2.metric("분석 구역",      f"{df['구'].nunique()}개구")
c3.metric("외국인민박 비중", f"{df['업종'].value_counts(normalize=True).get('외국인관광도시민박업',0)*100:.1f}%")
c4.metric("군집 수",        f"{core['군집'].nunique()}개")

st.markdown("---")
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("업종별 시설 분포")
    cnt = df['업종'].value_counts().reset_index()
    cnt.columns = ['업종', '수']
    fig = px.pie(cnt, names='업종', values='수',
                 color_discrete_sequence=['#2D5BE3', '#00C49F', '#FF6B6B'],
                 hole=0.45)
    fig.update_layout(margin=dict(t=10, b=10), legend=dict(orientation='h', y=-0.1))
    st.plotly_chart(fig, use_container_width=True)

with col_b:
    st.subheader("구별 시설 수")
    gu = df['구'].value_counts().reset_index()
    gu.columns = ['구', '수']
    fig2 = px.bar(gu.sort_values('수'), x='수', y='구', orientation='h',
                  color='수', color_continuous_scale='Blues')
    fig2.update_layout(margin=dict(t=10, b=10), coloraxis_showscale=False,
                       yaxis_title=None, xaxis_title='시설 수')
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")
st.info("👈 왼쪽 사이드바에서 분석 페이지를 선택하세요")
