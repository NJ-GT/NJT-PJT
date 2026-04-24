# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import plotly.express as px
import glob

st.set_page_config(page_title="위험 변수 탐색", page_icon="🔍", layout="wide")
st.title("🔍 위험 변수 탐색")

RISK_VARS = ['구조노후도', '단속위험도', '도로폭위험도', '집중도', '주변건물수']
COLOR_MAP  = {
    '외국인관광도시민박업': '#2D5BE3',
    '숙박업':              '#00C49F',
    '관광숙박업':          '#FF6B6B',
}

@st.cache_data
def load_df():
    f = glob.glob('data/*0423*.csv')[0]
    return pd.read_csv(f, encoding='utf-8-sig')

df = load_df()

st.sidebar.header("필터")
all_gu     = sorted(df['구'].dropna().unique())
sel_gu     = st.sidebar.multiselect("구 선택",   all_gu,                         default=all_gu)
sel_upjong = st.sidebar.multiselect("업종 선택", df['업종'].dropna().unique().tolist(),
                                    default=df['업종'].dropna().unique().tolist())

filt = df.copy()
if sel_gu:     filt = filt[filt['구'].isin(sel_gu)]
if sel_upjong: filt = filt[filt['업종'].isin(sel_upjong)]

st.markdown(f"**선택된 시설: {len(filt):,}개**")
if len(filt) == 0:
    st.warning("필터 조건에 맞는 데이터가 없습니다.")
    st.stop()

tab1, tab2, tab3 = st.tabs(["분포 비교", "업종별 평균", "상관관계"])

with tab1:
    sel_var = st.selectbox("변수 선택", RISK_VARS)
    fig = px.violin(filt, x='업종', y=sel_var, color='업종', box=True, points=False,
                    color_discrete_map=COLOR_MAP,
                    labels={'업종': '', sel_var: sel_var})
    fig.update_layout(showlegend=False, xaxis_title=None)
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.histogram(filt, x=sel_var, color='업종', barmode='overlay',
                        opacity=0.7, nbins=40, color_discrete_map=COLOR_MAP)
    fig2.update_layout(bargap=0.05, xaxis_title=sel_var, yaxis_title='시설 수')
    st.plotly_chart(fig2, use_container_width=True)

with tab2:
    mean_df   = filt.groupby('업종')[RISK_VARS].mean().reset_index()
    mean_melt = mean_df.melt(id_vars='업종', var_name='변수', value_name='평균값')
    fig3 = px.bar(mean_melt, x='변수', y='평균값', color='업종', barmode='group',
                  color_discrete_map=COLOR_MAP, text_auto='.3f')
    fig3.update_traces(textposition='outside', textfont_size=10)
    fig3.update_layout(xaxis_title=None, uniformtext_minsize=8)
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("요약 통계")
    st.dataframe(filt[RISK_VARS].describe().T.round(3), use_container_width=True)

with tab3:
    corr = filt[RISK_VARS].corr()
    fig4 = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r',
                     zmin=-1, zmax=1, aspect='auto')
    fig4.update_layout(margin=dict(t=10))
    st.plotly_chart(fig4, use_container_width=True)

    st.subheader("구 × 업종별 변수 평균 히트맵")
    sel_var2 = st.selectbox("변수 선택 (히트맵)", RISK_VARS, key='hm_var')
    pivot = filt.pivot_table(index='구', columns='업종', values=sel_var2, aggfunc='mean').round(3)
    fig5  = px.imshow(pivot, text_auto='.3f', color_continuous_scale='YlOrRd', aspect='auto')
    st.plotly_chart(fig5, use_container_width=True)
