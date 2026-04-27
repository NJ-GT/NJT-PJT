# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="화재출동 분석", page_icon="🚒", layout="wide")
st.title("🚒 화재출동 분석 (2017–2024)")

GU_10       = ['강남구','강서구','마포구','서초구','성동구','송파구','영등포구','용산구','종로구','중구']
WDAY_ORDER  = ['월요일','화요일','수요일','목요일','금요일','토요일','일요일']

@st.cache_data
def load_fire():
    df = pd.read_csv('data/화재출동/화재출동_2021_2024.csv', encoding='utf-8-sig', low_memory=False)
    df = df[df['발생시군구'].isin(GU_10)].copy()
    df['발생연도']  = df['발생연도'].astype(int)
    df['발생시']    = pd.to_numeric(df['발생시'], errors='coerce')
    df['요일']      = df['발생요일'].str.strip()
    df['발화요인']  = df['발화요인_대분류'].str.strip()
    df['발화장소중'] = df['발화장소_중분류'].str.strip()
    return df

fire = load_fire()

st.sidebar.header("필터")
sel_gu   = st.sidebar.multiselect("구 선택", GU_10, default=GU_10)
yr_min   = int(fire['발생연도'].min())
yr_range = st.sidebar.slider("연도 범위", yr_min, 2024, (2021, 2024))

filt = fire[fire['발생시군구'].isin(sel_gu) & fire['발생연도'].between(*yr_range)]

총건수  = len(filt)
총피해액 = filt['재산피해액(천원)'].sum()
c1, c2, c3 = st.columns(3)
c1.metric("총 화재 건수",   f"{총건수:,}건")
c2.metric("총 재산피해액",  f"{총피해액/1000:.0f}백만원")
c3.metric("건당 평균피해액", f"{총피해액/총건수/1000:.1f}백만원" if 총건수 else "—")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["연도별 추이", "요일·시간 히트맵", "발화요인", "출동시간", "숙박 화재"])

with tab1:
    by_yr_gu = (filt.groupby(['발생연도', '발생시군구'])
                .agg(건수=('화재번호','count'), 피해액=('재산피해액(천원)','sum'))
                .reset_index())

    fig = px.bar(by_yr_gu, x='발생연도', y='건수', color='발생시군구',
                 barmode='stack', title='구별 연도별 화재 건수',
                 color_discrete_sequence=px.colors.qualitative.Bold)
    fig.update_xaxes(tickvals=list(range(yr_range[0], yr_range[1]+1)), tickformat='d')
    st.plotly_chart(fig, use_container_width=True)

    total_yr = by_yr_gu.groupby('발생연도')[['건수','피해액']].sum().reset_index()
    total_yr['피해액(백만원)'] = (total_yr['피해액'] / 1000).round(1)
    fig2 = px.line(total_yr, x='발생연도', y=['건수', '피해액(백만원)'],
                   markers=True, title='전체 화재 건수 & 피해액 추이')
    fig2.update_xaxes(tickvals=list(range(yr_range[0], yr_range[1]+1)), tickformat='d')
    st.plotly_chart(fig2, use_container_width=True)

with tab2:
    avail_days = [d for d in WDAY_ORDER if d in filt['요일'].values]
    pivot = (filt.groupby(['요일','발생시']).size()
             .unstack(fill_value=0)
             .reindex(avail_days)
             .reindex(columns=range(0, 24), fill_value=0))

    fig3 = px.imshow(
        pivot, labels=dict(x='시간(시)', y='요일', color='건수'),
        color_continuous_scale='YlOrRd', title='요일 × 시간대 화재 발생 히트맵',
        aspect='auto', text_auto=True,
    )
    fig3.update_layout(height=320)
    st.plotly_chart(fig3, use_container_width=True)

with tab3:
    cause = (filt.groupby('발화요인')['화재번호'].count()
             .sort_values(ascending=False).head(10).reset_index())
    cause.columns = ['발화요인', '건수']

    fig4 = px.bar(cause, x='건수', y='발화요인', orientation='h',
                  color='건수', color_continuous_scale='Reds',
                  title='상위 10 발화요인')
    fig4.update_layout(coloraxis_showscale=False, yaxis_title=None)
    st.plotly_chart(fig4, use_container_width=True)

    place = (filt.groupby('발화장소중')['화재번호'].count()
             .sort_values(ascending=False).head(12).reset_index())
    place.columns = ['발화장소', '건수']
    fig5 = px.pie(place, names='발화장소', values='건수',
                  title='발화장소(중분류) 비중', hole=0.35,
                  color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig5, use_container_width=True)

with tab4:
    fig6 = px.histogram(filt, x='출동소요시간', nbins=60,
                        color_discrete_sequence=['#2D5BE3'],
                        title='출동소요시간 분포 (초)')
    fig6.add_vline(x=420, line_dash='dash', line_color='red',
                   annotation_text='골든타임 7분(420초)', annotation_position='top right')
    st.plotly_chart(fig6, use_container_width=True)

    by_gu_t = (filt.groupby('발생시군구')['출동소요시간']
               .agg(['mean','median']).round(0).reset_index())
    by_gu_t.columns = ['구','평균(초)','중앙값(초)']
    fig7 = px.bar(by_gu_t.sort_values('평균(초)'), x='평균(초)', y='구',
                  orientation='h', color='평균(초)', color_continuous_scale='RdYlGn_r',
                  title='구별 평균 출동소요시간')
    fig7.add_vline(x=420, line_dash='dash', line_color='red',
                   annotation_text='7분', annotation_position='top right')
    fig7.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig7, use_container_width=True)

with tab5:
    sukbak = filt[filt['발화장소중'] == '숙박시설']

    c1, c2, c3 = st.columns(3)
    c1.metric("숙박시설 화재 건수", f"{len(sukbak):,}건")
    c2.metric("전체 화재 중 비중", f"{len(sukbak)/len(filt)*100:.1f}%" if len(filt) else "—")
    c3.metric("재산피해액 합계", f"{sukbak['재산피해액(천원)'].sum()/1000:.1f}백만원")

    by_yr_s = sukbak.groupby('발생연도').size().reset_index(name='건수')
    fig8 = px.bar(by_yr_s, x='발생연도', y='건수',
                  color='건수', color_continuous_scale='Reds',
                  title='숙박시설 화재 연도별 건수')
    fig8.update_layout(coloraxis_showscale=False)
    fig8.update_xaxes(tickformat='d')
    st.plotly_chart(fig8, use_container_width=True)

    by_gu_s = sukbak.groupby('발생시군구').size().reset_index(name='건수')
    fig9 = px.bar(by_gu_s.sort_values('건수', ascending=False), x='발생시군구', y='건수',
                  color='건수', color_continuous_scale='Oranges',
                  title='구별 숙박시설 화재 건수')
    fig9.update_layout(coloraxis_showscale=False, xaxis_title=None)
    st.plotly_chart(fig9, use_container_width=True)

    if len(sukbak):
        cause_s = (sukbak.groupby('발화요인')['화재번호'].count()
                   .sort_values(ascending=False).head(8).reset_index())
        cause_s.columns = ['발화요인', '건수']
        fig10 = px.pie(cause_s, names='발화요인', values='건수',
                       title='숙박시설 화재 발화요인', hole=0.35,
                       color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig10, use_container_width=True)
