# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="인허가 트렌드", page_icon="📊", layout="wide")
st.title("📊 인허가 트렌드 (2020–2025)")

GU_10 = ['강남구','강서구','마포구','서초구','성동구','송파구','영등포구','용산구','종로구','중구']
YEARS = list(range(2020, 2026))
PAL   = px.colors.qualitative.Bold

@st.cache_data
def load_license():
    base = '원본데이터'
    try:
        f1 = pd.read_csv(f'{base}/서울시 외국인관광도시민박업 인허가 정보.csv', encoding='utf-8-sig', low_memory=False)
        f2 = pd.read_csv(f'{base}/서울시 관광숙박업 인허가 정보.csv',          encoding='utf-8-sig', low_memory=False)
        f3 = pd.read_csv(f'{base}/서울시 숙박업 인허가 정보.csv',              encoding='utf-8-sig', low_memory=False)
    except FileNotFoundError as e:
        return None, str(e)
    out = {}
    for key, df in [('민박', f1), ('관광', f2), ('숙박', f3)]:
        df = df.copy()
        df['인허가연도'] = pd.to_datetime(df['인허가일자'], errors='coerce').dt.year
        df['구'] = df['지번주소'].str.extract(r'서울특별시\s+(\S+구)')
        out[key] = df[df['구'].isin(GU_10)]
    return out, None

data, err = load_license()
if err:
    st.error(f"원본 데이터 로드 실패: {err}")
    st.stop()

st.sidebar.header("필터")
sel_gu   = st.sidebar.multiselect("구 선택", GU_10, default=GU_10[:5])
yr_range = st.sidebar.slider("연도 범위", 2020, 2025, (2020, 2025))

if not sel_gu:
    st.warning("구를 하나 이상 선택하세요.")
    st.stop()

yrs = list(range(yr_range[0], yr_range[1] + 1))

def build_trend(df, sel_gu, yrs):
    rows = []
    for gu in sel_gu:
        g = df[df['구'] == gu]
        for yr in yrs:
            rows.append({'구': gu, '연도': yr, '신규인허가': int((g['인허가연도'] == yr).sum())})
    return pd.DataFrame(rows)

tab1, tab2, tab3 = st.tabs(["🏠 외국인관광도시민박업", "🏨 숙박업", "🏩 관광숙박업"])

with tab1:
    df_plot = build_trend(data['민박'], sel_gu, yrs)
    fig = px.line(df_plot, x='연도', y='신규인허가', color='구', markers=True,
                  title='외국인관광도시민박업 — 구별 신규 인허가 추이',
                  color_discrete_sequence=PAL)
    fig.update_xaxes(tickvals=yrs, ticktext=[str(y) for y in yrs])
    st.plotly_chart(fig, use_container_width=True)

    total = df_plot.groupby('구')['신규인허가'].sum().reset_index().rename(columns={'신규인허가': '합계'})
    fig2 = px.bar(total.sort_values('합계', ascending=False), x='구', y='합계',
                  color='합계', color_continuous_scale='Blues',
                  title=f'{yr_range[0]}–{yr_range[1]} 구별 누적 신규 인허가')
    fig2.update_layout(coloraxis_showscale=False, xaxis_title=None)
    st.plotly_chart(fig2, use_container_width=True)

with tab2:
    df_plot = build_trend(data['숙박'], sel_gu, yrs)
    fig = px.bar(df_plot, x='연도', y='신규인허가', color='구', barmode='group',
                 title='숙박업 — 구별 신규 인허가',
                 color_discrete_sequence=PAL)
    fig.update_xaxes(tickvals=yrs, ticktext=[str(y) for y in yrs])
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    df_plot = build_trend(data['관광'], sel_gu, yrs)
    fig = px.bar(df_plot, x='연도', y='신규인허가', color='구', barmode='group',
                 title='관광숙박업 — 구별 신규 인허가',
                 color_discrete_sequence=PAL)
    fig.update_xaxes(tickvals=yrs, ticktext=[str(y) for y in yrs])
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.subheader("전체 업종 비교 (선택 기간·구)")
rows_all = []
for key, label in [('민박','외국인관광도시민박업'), ('숙박','숙박업'), ('관광','관광숙박업')]:
    g = data[key][data[key]['구'].isin(sel_gu)]
    for yr in yrs:
        rows_all.append({'업종': label, '연도': yr, '신규인허가': int((g['인허가연도'] == yr).sum())})
df_all = pd.DataFrame(rows_all)
fig_all = px.bar(df_all, x='연도', y='신규인허가', color='업종', barmode='stack',
                 color_discrete_sequence=['#2D5BE3','#00C49F','#FF6B6B'],
                 title='업종별 신규 인허가 합계 (선택 구 합산)')
fig_all.update_xaxes(tickvals=yrs, ticktext=[str(y) for y in yrs])
st.plotly_chart(fig_all, use_container_width=True)
