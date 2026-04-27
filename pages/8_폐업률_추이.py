# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="폐업률 추이", page_icon="📉", layout="wide")
st.title("📉 인허가 폐업률 & 생존 분석")

GU_10 = ['강남구','강서구','마포구','서초구','성동구','송파구','영등포구','용산구','종로구','중구']
LABEL = {'민박':'외국인관광도시민박업', '관광':'관광숙박업', '숙박':'숙박업'}
PAL   = {'외국인관광도시민박업':'#2D5BE3', '관광숙박업':'#00C49F', '숙박업':'#FF6B6B'}

@st.cache_data
def load_license():
    base = '원본데이터'
    dfs  = {}
    for key, fname in [
        ('민박', '서울시 외국인관광도시민박업 인허가 정보.csv'),
        ('관광', '서울시 관광숙박업 인허가 정보.csv'),
        ('숙박', '서울시 숙박업 인허가 정보.csv'),
    ]:
        df = pd.read_csv(f'{base}/{fname}', encoding='utf-8-sig', low_memory=False)
        df['인허가연도'] = pd.to_datetime(df['인허가일자'], errors='coerce').dt.year
        df['구']       = df['지번주소'].str.extract(r'서울특별시\s+(\S+구)')
        dfs[key]       = df[df['구'].isin(GU_10)].copy()
    return dfs

data = load_license()

st.sidebar.header("필터")
sel_gu   = st.sidebar.multiselect("구 선택", GU_10, default=GU_10)
yr_range = st.sidebar.slider("기준 연도 범위 (인허가연도)", 2010, 2025, (2015, 2025))

tab1, tab2, tab3 = st.tabs(["현재 영업상태", "구별 폐업률", "누적 생존 추이"])

with tab1:
    rows = []
    for key, df in data.items():
        g = df[df['구'].isin(sel_gu)]
        for state, cnt in g['영업상태명'].value_counts().items():
            rows.append({'업종': LABEL[key], '영업상태': state, '수': cnt})
    state_df = pd.DataFrame(rows)

    fig = px.bar(state_df, x='업종', y='수', color='영업상태', barmode='stack',
                 title='업종별 현재 영업상태 전체 현황',
                 color_discrete_sequence=px.colors.qualitative.Set2)
    st.plotly_chart(fig, use_container_width=True)

    # 폐업률 요약 메트릭
    st.markdown("---")
    cols = st.columns(3)
    for i, (key, df) in enumerate(data.items()):
        g    = df[df['구'].isin(sel_gu)]
        rate = (g['영업상태명'] == '폐업').mean() * 100
        cols[i].metric(LABEL[key], f"폐업률 {rate:.1f}%", f"총 {len(g):,}개")

with tab2:
    rows = []
    for key, df in data.items():
        g = df[df['구'].isin(sel_gu) & df['인허가연도'].between(*yr_range)]
        if len(g) == 0:
            continue
        total  = g.groupby('구').size()
        closed = g[g['영업상태명'] == '폐업'].groupby('구').size()
        rate   = (closed / total * 100).fillna(0).round(1)
        for gu, r in rate.items():
            rows.append({'구': gu, '업종': LABEL[key], '폐업률(%)': r})
    rate_df = pd.DataFrame(rows)

    if not rate_df.empty:
        fig2 = px.bar(rate_df, x='구', y='폐업률(%)', color='업종', barmode='group',
                      title=f'구별 폐업률 ({yr_range[0]}–{yr_range[1]}년 인허가 기준)',
                      color_discrete_map=PAL)
        fig2.update_layout(xaxis_title=None)
        st.plotly_chart(fig2, use_container_width=True)

        fig3 = px.imshow(
            rate_df.pivot_table(index='구', columns='업종', values='폐업률(%)', aggfunc='mean').round(1),
            text_auto='.1f', color_continuous_scale='RdYlGn_r',
            title='구 × 업종 폐업률 히트맵 (%)', aspect='auto',
        )
        st.plotly_chart(fig3, use_container_width=True)

with tab3:
    YEARS = list(range(yr_range[0], yr_range[1] + 1))
    rows  = []
    for key, df in data.items():
        g     = df[df['구'].isin(sel_gu)]
        label = LABEL[key]
        for yr in YEARS:
            sub    = g[g['인허가연도'] <= yr]
            active = (sub['영업상태명'] == '영업/정상').sum()
            closed = (sub['영업상태명'] == '폐업').sum()
            rows.append({'업종': label, '연도': yr, '영업중': int(active), '폐업': int(closed)})
    cum = pd.DataFrame(rows)

    fig4 = px.line(cum, x='연도', y='영업중', color='업종', markers=True,
                   title='누적 영업중 시설 수 추이',
                   color_discrete_map=PAL)
    st.plotly_chart(fig4, use_container_width=True)

    # 업종별 영업중+폐업 스택 area
    fig5 = go.Figure()
    for key, label in LABEL.items():
        sub  = cum[cum['업종'] == label]
        col  = PAL[label]
        fig5.add_trace(go.Scatter(
            x=sub['연도'], y=sub['영업중'], name=f'{label} 영업중',
            stackgroup='one', line_color=col, fillcolor=col, opacity=0.6,
        ))
    fig5.update_layout(title='업종별 누적 영업중 시설 (스택)', xaxis_title='연도', yaxis_title='누적 시설 수')
    st.plotly_chart(fig5, use_container_width=True)

    # 신규 vs 폐업 연도별 흐름
    st.markdown("---")
    st.subheader("연도별 신규 인허가 vs 폐업 건수")
    rows2 = []
    for key, df in data.items():
        g = df[df['구'].isin(sel_gu) & df['인허가연도'].between(*yr_range)]
        for yr in YEARS:
            신규 = (g['인허가연도'] == yr).sum()
            폐업 = ((g['인허가연도'] == yr) & (g['영업상태명'] == '폐업')).sum()
            rows2.append({'업종': LABEL[key], '연도': yr, '신규': int(신규), '폐업': int(폐업)})
    flow = pd.DataFrame(rows2)

    for label in LABEL.values():
        sub = flow[flow['업종'] == label]
        fig6 = go.Figure()
        fig6.add_trace(go.Bar(x=sub['연도'], y=sub['신규'], name='신규', marker_color='#2D5BE3'))
        fig6.add_trace(go.Bar(x=sub['연도'], y=-sub['폐업'], name='폐업', marker_color='#FF6B6B'))
        fig6.update_layout(
            title=f'{label} — 신규 vs 폐업',
            barmode='relative', xaxis_title=None, yaxis_title='건수',
            legend=dict(orientation='h'),
            xaxis=dict(tickvals=YEARS, tickformat='d'),
        )
        st.plotly_chart(fig6, use_container_width=True)
