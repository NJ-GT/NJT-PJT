# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="클러스터 분석", page_icon="🧩", layout="wide")
st.title("🧩 클러스터 분석")

PALETTE    = ['#2D5BE3', '#00C49F', '#FF6B6B', '#FFD166', '#A855F7']
RADAR_VARS = ['소방위험도_점수', '위험점수_AHP', '건물나이', '집중도(%)', '공식도로폭m']

@st.cache_data
def load_core():
    df = pd.read_csv('data/핵심서울0424.csv', encoding='utf-8-sig')
    df = df.dropna(subset=['군집']).copy()
    df['군집'] = df['군집'].astype(int)
    return df

core     = load_core()
clusters = sorted(core['군집'].unique())

st.markdown(f"총 **{len(clusters)}개** 군집  ·  **{len(core):,}개** 시설")

tab1, tab2, tab3 = st.tabs(["레이더 차트", "구별 군집 구성", "군집별 통계"])

with tab1:
    avail = [v for v in RADAR_VARS if v in core.columns]
    radar_df = core[['군집'] + avail].copy()
    for v in avail:
        mn, mx = radar_df[v].min(), radar_df[v].max()
        radar_df[v + '_n'] = (radar_df[v] - mn) / (mx - mn + 1e-9)

    norm_vars = [v + '_n' for v in avail]
    means     = radar_df.groupby('군집')[norm_vars].mean()

    fig = go.Figure()
    for i, cl in enumerate(clusters):
        vals  = means.loc[cl].tolist()
        cats  = avail + [avail[0]]
        vals += [vals[0]]
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=cats, fill='toself', name=f'군집 {cl}',
            line_color=PALETTE[i % len(PALETTE)],
            fillcolor=PALETTE[i % len(PALETTE)],
            opacity=0.35,
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        height=520, showlegend=True,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**군집별 시설 수**")
    sz = core['군집'].value_counts().sort_index().reset_index()
    sz.columns = ['군집', '수']
    sz['군집명'] = '군집 ' + sz['군집'].astype(str)
    fig2 = px.pie(sz, names='군집명', values='수',
                  color_discrete_sequence=PALETTE, hole=0.4)
    fig2.update_layout(margin=dict(t=10, b=10))
    st.plotly_chart(fig2, use_container_width=True)

with tab2:
    gu_cl = core.groupby(['구', '군집']).size().reset_index(name='수')
    gu_cl['군집명'] = '군집 ' + gu_cl['군집'].astype(str)
    fig3 = px.bar(gu_cl, x='구', y='수', color='군집명', barmode='stack',
                  color_discrete_sequence=PALETTE, title='구별 군집 구성')
    fig3.update_layout(xaxis_title=None, yaxis_title='시설 수', legend_title='군집')
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")
    pct = core.groupby(['구', '군집']).size().unstack(fill_value=0)
    pct = pct.div(pct.sum(axis=1), axis=0).mul(100).round(1)
    pct.columns = [f'군집 {c}' for c in pct.columns]
    st.subheader("구별 군집 비율 (%)")
    st.dataframe(pct, use_container_width=True)

with tab3:
    show_cols = [c for c in ['위험점수_AHP', '소방위험도_점수', '이동시간초', '건물나이', '공식도로폭m']
                 if c in core.columns]
    stats = core.groupby('군집')[show_cols].mean().round(3)
    stats.index = ['군집 ' + str(i) for i in stats.index]
    st.subheader("군집별 주요 변수 평균")
    st.dataframe(stats, use_container_width=True)

    st.markdown("---")
    sel_col = st.selectbox("변수 선택 (박스플롯)", show_cols)
    core_plot = core.copy()
    core_plot['군집명'] = '군집 ' + core_plot['군집'].astype(str)
    fig4 = px.box(core_plot, x='군집명', y=sel_col, color='군집명',
                  color_discrete_sequence=PALETTE, points=False)
    fig4.update_layout(showlegend=False, xaxis_title=None)
    st.plotly_chart(fig4, use_container_width=True)
