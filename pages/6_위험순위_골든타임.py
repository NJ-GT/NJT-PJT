# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="위험순위 & 골든타임", page_icon="🏆", layout="wide")
st.title("🏆 위험 시설 순위 & 소방 골든타임")

GOLDEN_DEFAULT = 210  # 유클리드 기준 ~3.5분 ≈ 실도로 7분(골든타임)

@st.cache_data
def load_data():
    df = pd.read_csv('data/data_with_fire_targets.csv', encoding='utf-8-sig')
    return df.dropna(subset=['위도','경도','위험점수_AHP'])

df = load_data()

st.sidebar.header("필터")
all_gu  = sorted(df['구'].dropna().unique())
sel_gu  = st.sidebar.multiselect("구 선택", all_gu, default=all_gu)
top_n   = st.sidebar.slider("상위 N개 시설", 10, 200, 50)
GOLDEN  = st.sidebar.slider("골든타임 기준(초)", 60, 420, GOLDEN_DEFAULT,
                             help="이동시간초는 유클리드(직선) 기반. 210초≈실도로 7분 상당")

filt = df[df['구'].isin(sel_gu)] if sel_gu else df

tab1, tab2, tab3 = st.tabs(["위험 시설 순위", "골든타임 초과 현황", "소방거리 분석"])

with tab1:
    top = (filt.nlargest(top_n, '위험점수_AHP')
           [['구','동','업소명','위험점수_AHP','소방위험도_점수','건물나이',
             '공식도로폭m','이동시간초','반경100m_화재수']]
           .reset_index(drop=True))
    top.index += 1
    top.columns = ['구','동','시설명','AHP위험점수','소방위험도','건물나이(년)',
                   '인근도로폭(m)','소방도달(초)','반경100m화재수']

    st.dataframe(
        top.style
           .background_gradient(subset=['AHP위험점수'], cmap='Reds')
           .background_gradient(subset=['소방도달(초)'], cmap='Oranges')
           .background_gradient(subset=['반경100m화재수'], cmap='YlOrRd'),
        use_container_width=True, height=450,
    )

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.scatter(filt, x='건물나이', y='위험점수_AHP', color='구',
                         hover_data=['업소명','동'],
                         title='건물나이 vs AHP 위험점수',
                         color_discrete_sequence=px.colors.qualitative.Bold,
                         opacity=0.6)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig2 = px.box(filt, x='구', y='위험점수_AHP', color='구',
                      title='구별 AHP 위험점수 분포', points=False,
                      color_discrete_sequence=px.colors.qualitative.Bold)
        fig2.update_layout(showlegend=False, xaxis_title=None)
        st.plotly_chart(fig2, use_container_width=True)

with tab2:
    over   = filt[filt['이동시간초'] > GOLDEN]
    pct    = len(over) / len(filt) * 100 if len(filt) else 0
    excess = (over['이동시간초'] - GOLDEN).mean()

    c1, c2, c3 = st.columns(3)
    c1.metric("골든타임 초과 시설", f"{len(over):,}개")
    c2.metric("전체 대비",         f"{pct:.1f}%")
    c3.metric("평균 초과 시간",    f"{excess:.0f}초 ({excess/60:.1f}분)" if len(over) else "—")

    # 구별 초과율 bar
    gu_rate = (filt.groupby('구')
               .apply(lambda x: (x['이동시간초'] > GOLDEN).mean() * 100)
               .round(1).reset_index())
    gu_rate.columns = ['구', '초과율(%)']
    avg = gu_rate['초과율(%)'].mean()

    fig3 = px.bar(gu_rate.sort_values('초과율(%)', ascending=False),
                  x='구', y='초과율(%)', color='초과율(%)',
                  color_continuous_scale='Reds', title='구별 골든타임 초과 비율')
    fig3.add_hline(y=avg, line_dash='dash', line_color='gray',
                   annotation_text=f'평균 {avg:.1f}%')
    fig3.update_layout(coloraxis_showscale=False, xaxis_title=None)
    st.plotly_chart(fig3, use_container_width=True)

    # 초과 시설 리스트
    st.subheader(f"골든타임 초과 시설 목록 ({len(over):,}개)")
    show = (over[['구','동','업소명','이동시간초','위험점수_AHP','건물나이','공식도로폭m']]
            .copy().sort_values('이동시간초', ascending=False))
    show['초과(초)'] = (show['이동시간초'] - GOLDEN).astype(int)
    st.dataframe(show.reset_index(drop=True), use_container_width=True, height=400)

with tab3:
    c1, c2 = st.columns(2)
    with c1:
        fig4 = px.scatter(filt, x='이동시간초', y='도로폭_보정이동시간초',
                          color='구', opacity=0.5,
                          hover_data=['업소명'],
                          title='소방도달시간: 원래 vs 도로폭 보정',
                          labels={'이동시간초':'원래(초)','도로폭_보정이동시간초':'도로폭 보정(초)'},
                          color_discrete_sequence=px.colors.qualitative.Bold)
        mx = max(filt['이동시간초'].max(), filt['도로폭_보정이동시간초'].max())
        fig4.add_shape(type='line', x0=0, y0=0, x1=mx, y1=mx,
                       line=dict(dash='dot', color='gray', width=1))
        fig4.add_vline(x=GOLDEN, line_dash='dash', line_color='red')
        fig4.add_hline(y=GOLDEN, line_dash='dash', line_color='red',
                       annotation_text='골든타임 7분')
        st.plotly_chart(fig4, use_container_width=True)
    with c2:
        fig5 = px.histogram(filt, x='이동시간초', color='구', barmode='overlay',
                            opacity=0.6, nbins=50,
                            title='소방도달시간 분포',
                            color_discrete_sequence=px.colors.qualitative.Bold)
        fig5.add_vline(x=GOLDEN, line_dash='dash', line_color='red',
                       annotation_text='7분')
        st.plotly_chart(fig5, use_container_width=True)

    fig6 = px.scatter(filt, x='공식도로폭m', y='위험점수_AHP',
                      color='구', opacity=0.5,
                      hover_data=['업소명','동'],
                      title='인근 도로폭 vs AHP 위험점수',
                      labels={'공식도로폭m':'도로폭(m)','위험점수_AHP':'AHP 위험점수'},
                      color_discrete_sequence=px.colors.qualitative.Bold)
    st.plotly_chart(fig6, use_container_width=True)
