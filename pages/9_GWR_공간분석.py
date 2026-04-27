# -*- coding: utf-8 -*-
import glob
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="GWR 공간분석", page_icon="🌡️", layout="wide")
st.title("🌡️ GWR 지역별 위험 계수 분석")
st.caption("Geographically Weighted Regression — Y=log(화재수+1), 주변건물수=통제변수 포함, 6변수")

@st.cache_data
def load_gwr():
    return pd.read_csv('data/gwr_results.csv', encoding='utf-8-sig')

@st.cache_data
def load_base():
    f    = glob.glob('data/*0423*.csv')[0]
    main = pd.read_csv(f, encoding='utf-8-sig')
    core = pd.read_csv('data/data_with_fire_targets.csv', encoding='utf-8-sig')
    cc   = core.columns
    core_key = core[[cc[4], cc[5], cc[17], cc[22], cc[45]]].copy()
    core_key.columns = ['위도', '경도', '소방위험도_점수', '위험점수_AHP', '반경100m_화재수']
    df = pd.merge(main, core_key, on=['위도', '경도'], how='left')
    for v in ['위험점수_AHP', '반경100m_화재수', '주변건물수']:
        df[v] = pd.to_numeric(df[v], errors='coerce')
    return df.dropna(subset=['위험점수_AHP', '반경100m_화재수', '위도', '경도']).reset_index(drop=True)

gwr  = load_gwr()
base = load_base()

RISK_VARS = [c.replace('coef_', '') for c in gwr.columns if c.startswith('coef_')]

# ── MGWR 변수별 BW (Y=log(화재수+1), 6변수 실행 결과) ──
MGWR_BW = {
    'Intercept':    70,
    '구조노후도':   498,
    '단속위험도':   498,
    '도로폭위험도': 498,
    '집중도':       484,
    '주변건물수':   44,
    '소방위험도_점수': 498,
}

# ── 모델 요약 메트릭 ──
st.markdown("### 모델 성능 비교 (6변수 + BW 최적화, Y=실제 화재수)")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("GWR R²",    "0.5703", "adj 0.4448")
c2.metric("Bandwidth", "68 NN",  help="AICc 최소화 기준 최적 BW")
c3.metric("MGWR R²",   "0.3640", "adj 0.2897")
c4.metric("VIF 최대",   "1.55",   help="모든 변수 VIF < 2 → 다중공선성 없음")
c5.metric("Y 변수",     "화재수", help="log(반경100m 화재수+1): 실제 발생 기록")

st.info(
    "**최종 모델 (6변수, Y=log(화재수+1)):**\n\n"
    "- **주변건물수 = 통제변수**: X에 포함하여 다른 계수들이 건물 수 보정 후 순수 위험 효과를 추정\n"
    "- Y = log(반경100m 화재수+1) — 실제 발생 기록 기준 (AHP 순환 제거)\n"
    "- VIF 최대 1.55 → 다중공선성 없음\n"
    "- MGWR: 주변건물수 BW=44(국소), 나머지 광역 → 건물 밀집 효과만 공간적으로 이질적"
)

st.markdown("---")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📍 유의 계수 분포", "📊 계수 지도", "📈 계수 분포",
    "🔬 MGWR 변수별 BW", "🔗 STEP 3 — AHP·화재 연결", "🔍 해석 가이드"
])

# ════════════════════════════════════
# TAB 1 │ 변수별 유의 지점 분포
# ════════════════════════════════════
with tab1:
    st.subheader("변수별 통계적 유의 지점 분포 (|t| > 1.96)")
    st.caption(
        "Local R² 지도는 Y=화재수 데이터 희소성으로 수치 불안정 → "
        "대신 '어느 지역에서 어떤 변수가 통계적으로 유의한가'로 공간 이질성을 확인합니다."
    )

    st.info(
        "**왜 Local R² 지도를 쓰지 않나?**  \n"
        "화재수=0인 시설이 대부분이라 근방 Y 분산이 0에 가까운 지점에서 "
        "local R² = 1 - SS_res/SS_tot 계산이 수치 불안정(-inf, 극단 음수)합니다.  \n"
        "Global R²=0.570은 전체 적합도 기준으로 유효합니다."
    )

    # 변수별 유의 비율 바 차트
    sig_rows = []
    for v in RISK_VARS:
        tval_col = f'tval_{v}'
        sig_pct  = float((gwr[tval_col].abs() > 1.96).mean() * 100)
        sig_rows.append({'변수': v, '유의 비율(%)': sig_pct})
    sig_df = pd.DataFrame(sig_rows).sort_values('유의 비율(%)', ascending=False)

    fig_sig = px.bar(
        sig_df, x='변수', y='유의 비율(%)',
        color='유의 비율(%)', color_continuous_scale=['#DBEAFE', '#1D4ED8'],
        text='유의 비율(%)', title='변수별 |t|>1.96 유의 지점 비율',
    )
    fig_sig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig_sig.add_hline(y=50, line_dash='dash', line_color='gray',
                      annotation_text='50% 기준선')
    fig_sig.update_layout(height=380, plot_bgcolor='#FAFAFA', paper_bgcolor='white',
                          coloraxis_showscale=False, margin=dict(t=40, b=10))
    st.plotly_chart(fig_sig, use_container_width=True)

    # 모든 변수가 동시에 유의한 지점 지도
    st.subheader("복수 변수 동시 유의 지점 지도")
    st.caption("두 개 이상 변수에서 동시에 |t|>1.96인 지점 — 위험 요인이 복합 작용하는 핫스팟")

    gwr_plot2 = gwr.copy()
    gwr_plot2['유의_수'] = sum(
        (gwr[f'tval_{v}'].abs() > 1.96).astype(int) for v in RISK_VARS
    )
    gwr_plot2['색상'] = pd.cut(
        gwr_plot2['유의_수'], bins=[-1, 0, 1, 2, 10],
        labels=['0개(비유의)', '1개', '2개', '3개+']
    )
    cat_colors = {'0개(비유의)': '#CBD5E1', '1개': '#93C5FD', '2개': '#F59E0B', '3개+': '#EF4444'}

    fig_multi = px.scatter_mapbox(
        gwr_plot2, lat='위도', lon='경도',
        color='색상', color_discrete_map=cat_colors,
        size_max=8, mapbox_style='carto-positron',
        zoom=10.5, center=dict(lat=37.53, lon=126.99),
        opacity=0.8, labels={'색상': '유의 변수 수'},
        title='복수 위험변수 동시 유의 지점 (핫스팟)',
    )
    fig_multi.update_layout(height=500, margin=dict(t=40, b=0))
    st.plotly_chart(fig_multi, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Global GWR R²",  "0.5703", "adj 0.4448")
        st.metric("3개+ 동시유의", f"{(gwr_plot2['유의_수']>=3).mean()*100:.1f}%")
    with col2:
        st.metric("BW",             "68 NN",  "최적 적응형 이웃수")
        st.metric("1개↑ 유의",      f"{(gwr_plot2['유의_수']>=1).mean()*100:.1f}%")

    # ── 지배 변수 지도 (마포구 등 비유의 지역도 색상 표시) ──
    st.markdown("---")
    st.subheader("지배 위험변수 지도 (전 지점)")
    st.caption(
        "각 지점에서 |t|값이 가장 높은 변수를 색상으로 표시합니다. "
        "유의하지 않은 지점도 색상이 부여되어 마포구 등 '어느 위험요인이 주도적인가'를 파악할 수 있습니다."
    )

    dom_df = gwr.copy()
    tval_cols = {v: f'tval_{v}' for v in RISK_VARS}
    tval_abs  = pd.DataFrame({v: dom_df[col].abs() for v, col in tval_cols.items()})
    dom_df['지배변수'] = tval_abs.idxmax(axis=1)
    dom_df['최대_t']   = tval_abs.max(axis=1)

    var_colors = {
        '구조노후도':      '#E53E3E',
        '단속위험도':      '#DD6B20',
        '도로폭위험도':    '#D69E2E',
        '집중도':          '#38A169',
        '주변건물수':      '#3182CE',
        '소방위험도_점수': '#805AD5',
    }

    fig_dom = px.scatter_mapbox(
        dom_df, lat='위도', lon='경도',
        color='지배변수',
        color_discrete_map=var_colors,
        hover_data=['지배변수', '최대_t'],
        size_max=8, mapbox_style='carto-positron',
        zoom=10.5, center=dict(lat=37.53, lon=126.99),
        opacity=0.75, labels={'지배변수': '주도 위험변수'},
        title='지점별 주도 위험변수 (|t| 최대 기준)',
    )
    fig_dom.update_layout(height=520, margin=dict(t=40, b=0))
    st.plotly_chart(fig_dom, use_container_width=True)

    # 구별 지배변수 분포 (마포구 포함)
    if '구' in gwr.columns or '구' in base.columns:
        gu_col_src = dom_df if '구' in dom_df.columns else None
        if gu_col_src is None:
            # 위도·경도로 base와 merge해서 '구' 컬럼 가져오기
            if '구' in base.columns:
                gu_map = base[['위도', '경도', '구']].drop_duplicates()
                dom_df2 = dom_df.merge(gu_map, on=['위도', '경도'], how='left')
            else:
                dom_df2 = dom_df.copy()
        else:
            dom_df2 = gu_col_src

        if '구' in dom_df2.columns:
            focus_gu = ['마포구', '용산구', '강남구', '종로구', '중구']
            dom_focus = dom_df2[dom_df2['구'].isin(focus_gu)]
            if len(dom_focus) > 0:
                gu_dom = (
                    dom_focus.groupby(['구', '지배변수'])
                    .size().reset_index(name='count')
                )
                gu_tot = dom_focus.groupby('구').size().reset_index(name='total')
                gu_dom = gu_dom.merge(gu_tot, on='구')
                gu_dom['비율(%)'] = gu_dom['count'] / gu_dom['total'] * 100

                fig_gu_dom = px.bar(
                    gu_dom, x='구', y='비율(%)', color='지배변수',
                    color_discrete_map=var_colors,
                    title='주요 구별 지배 위험변수 분포',
                    barmode='stack',
                )
                fig_gu_dom.update_layout(height=380, margin=dict(t=40, b=0))
                st.plotly_chart(fig_gu_dom, use_container_width=True)

# ════════════════════════════════════
# TAB 2 │ 계수 지도
# ════════════════════════════════════
with tab2:
    st.subheader("변수별 GWR 계수 지도")
    st.caption("양수(붉은색) = 해당 지역에서 화재율 높이는 방향, 음수(파란색) = 낮추는 방향")

    var_labels = {
        '구조노후도':    '구조노후도 (건물 노후화)',
        '단속위험도':    '단속위험도 (불법주정차)',
        '도로폭위험도':  '도로폭위험도 (좁은 도로)',
        '집중도':        '집중도 (주변 시설 밀집)',
        '소방위험도_점수': '소방위험도_점수 (소방 접근성)',
    }
    sel_var  = st.selectbox("변수 선택", RISK_VARS, format_func=lambda x: var_labels.get(x, x))
    col_name = f'coef_{sel_var}'
    tval_col = f'tval_{sel_var}'
    abs_max  = float(gwr[col_name].abs().quantile(0.95))

    gwr_plot        = gwr.copy()
    gwr_plot['유의'] = gwr_plot[tval_col].abs() > 1.96
    gwr_plot['size'] = gwr_plot['유의'].map({True: 8, False: 4})

    fig2 = px.scatter_mapbox(
        gwr_plot, lat='위도', lon='경도',
        color=col_name, color_continuous_scale='RdBu_r',
        range_color=[-abs_max, abs_max],
        size='size', size_max=10,
        mapbox_style='carto-positron',
        zoom=10.5, center=dict(lat=37.53, lon=126.99),
        opacity=0.8, labels={col_name: f'{sel_var} 계수'},
        title=f'GWR 계수 지도 — {sel_var}  (큰 점 = |t|>1.96 유의)',
    )
    fig2.update_layout(height=550, margin=dict(t=30, b=0))
    st.plotly_chart(fig2, use_container_width=True)

    sig_pct = float((gwr_plot[tval_col].abs() > 1.96).mean() * 100)
    st.caption(f"통계적으로 유의한 지점: {sig_pct:.1f}%  (|t| > 1.96)")

# ════════════════════════════════════
# TAB 3 │ 계수 분포
# ════════════════════════════════════
with tab3:
    st.subheader("변수별 계수 분포 비교")

    rows = []
    for v in RISK_VARS:
        col = gwr[f'coef_{v}']
        rows.append({'변수': v, '평균': col.mean(), '중앙값': col.median(),
                     '표준편차': col.std(), '최솟값': col.min(), '최댓값': col.max(),
                     'IQR': col.quantile(0.75) - col.quantile(0.25)})
    stat_df = pd.DataFrame(rows).round(3)
    st.dataframe(stat_df.set_index('변수'), use_container_width=True)

    melt = pd.DataFrame({v: gwr[f'coef_{v}'] for v in RISK_VARS}).melt(var_name='변수', value_name='계수')
    fig3 = px.violin(melt, x='변수', y='계수', box=True, points=False,
                     color='변수', color_discrete_sequence=px.colors.qualitative.Bold,
                     title='변수별 GWR 계수 분포 (바이올린)')
    fig3.add_hline(y=0, line_dash='dash', line_color='gray')
    fig3.update_layout(showlegend=False, xaxis_title=None)
    st.plotly_chart(fig3, use_container_width=True)

    coef_cols = [f'coef_{v}' for v in RISK_VARS]
    corr = gwr[coef_cols].rename(columns={f'coef_{v}': v for v in RISK_VARS}).corr()
    fig4 = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r',
                     zmin=-1, zmax=1, title='GWR 계수 간 공간 상관관계')
    st.plotly_chart(fig4, use_container_width=True)

# ════════════════════════════════════
# TAB 4 │ MGWR 변수별 BW
# ════════════════════════════════════
with tab4:
    st.subheader("MGWR — 변수별 최적 Bandwidth (Y=log(화재수+1), 6변수)")
    st.caption("500샘플 backfitting 수렴 | 주변건물수 BW=44(국소), 나머지 BW≈500(전역)")

    bw_df = pd.DataFrame([
        {'변수': k, 'BW': v,
         '영역': '국소' if v < 80 else ('중간' if v < 200 else '광역')}
        for k, v in MGWR_BW.items()
    ])
    color_map = {'광역': '#2563EB', '중간': '#F59E0B', '국소': '#10B981'}
    fig_bw = px.bar(bw_df, x='변수', y='BW', color='영역',
                    color_discrete_map=color_map, text='BW',
                    title='MGWR 변수별 최적 Bandwidth (offset 보정)')
    fig_bw.update_traces(textposition='outside')
    fig_bw.update_layout(yaxis_title='Bandwidth (NN)', xaxis_title=None, height=420)
    st.plotly_chart(fig_bw, use_container_width=True)

    interp = {
        'Intercept':     ('70',  '국소', '기준값 — Intercept만 지역 특성 반영'),
        '주변건물수':    ('44',  '국소', '건물 밀집 효과는 골목 단위로 극히 국소적'),
        '집중도':        ('484', '광역', '시설 밀집도는 서울 전역에 균일한 효과'),
        '구조노후도':    ('498', '광역', '노후 건물 효과는 전역 동질적'),
        '단속위험도':    ('498', '광역', '불법주정차 효과는 전역 동질적'),
        '도로폭위험도':  ('498', '광역', '도로폭 효과는 전역 동질적'),
        '소방위험도_점수': ('498','광역', '소방 접근성 효과는 전역 동질적'),
    }
    rows_i = [{'변수': k, 'BW': v[0], '범위': v[1], '해석': v[2]} for k, v in interp.items()]
    st.dataframe(pd.DataFrame(rows_i).set_index('변수'), use_container_width=True)

    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("MGWR R²",  "0.3640", "adj 0.2897")
    col_m2.metric("국소 BW", "44 NN",  "주변건물수 — 골목 단위")
    col_m3.metric("광역 BW", "≈500 NN","나머지 5변수 — 전역 균일")

    st.info(
        "**MGWR 핵심 해석**\n\n"
        "- **주변건물수 BW=44** (유일하게 국소): 인접 건물 수의 화재 영향은 골목 단위로 급격히 변함\n"
        "- **나머지 5변수 BW≈500**: 구조노후도·단속위험도 등은 서울 전역에서 동일한 방향으로 작용\n"
        "- 해석: '화재 발생에 가장 공간적으로 이질적인 요인은 건물 밀도(주변건물수)뿐'\n"
        "- 나머지 위험 요인들은 지역과 무관하게 일관된 효과 → OLS/Spatial Error로 충분히 설명 가능"
    )

# ════════════════════════════════════
# TAB 5 │ STEP 3 — AHP·화재 연결
# ════════════════════════════════════
with tab5:
    st.subheader("STEP 3 — 이론적 위험지수(AHP)와 실제 화재수 연결 분석")
    st.caption("위험지수가 높은 지역에서 실제 화재도 많이 발생하는지 검증")

    st.markdown("""
    <div style='background:#EFF6FF; border-left:4px solid #2563EB; padding:10px 16px;
    border-radius:0 8px 8px 0; margin-bottom:12px;'>
    <b>분석 전략</b>: AHP(이론 위험) ↔ 화재수(실제 발생) 두 모델은 서로 다른 관점.<br>
    이 탭은 두 결과가 <b>공간적으로 일치하는지</b> 확인하여 AHP 위험지수의 현실 타당성을 검증합니다.
    </div>
    """, unsafe_allow_html=True)

    # ── 데이터 준비 ──
    plot_df = base[['위도', '경도', '위험점수_AHP', '반경100m_화재수', '주변건물수']].copy()
    plot_df['화재율'] = np.log1p(plot_df['반경100m_화재수']) - np.log1p(plot_df['주변건물수'].fillna(0))
    plot_df = plot_df.dropna()

    corr_val = plot_df['위험점수_AHP'].corr(plot_df['반경100m_화재수'])
    corr_rate = plot_df['위험점수_AHP'].corr(plot_df['화재율'])

    c1, c2, c3 = st.columns(3)
    c1.metric("AHP ↔ 화재수 상관", f"{corr_val:.3f}")
    c2.metric("AHP ↔ 화재율 상관", f"{corr_rate:.3f}")
    c3.metric("분석 시설 수", f"{len(plot_df):,}개")

    col_l, col_r = st.columns(2)

    # ── 산점도 ──
    with col_l:
        st.subheader("① AHP 위험점수 vs 실제 화재수")
        st.caption(f"상관계수 r={corr_val:.3f} — 위험지수 높을수록 화재 발생 경향")

        samp = plot_df.sample(min(2000, len(plot_df)), random_state=42)
        slope = np.polyfit(samp['위험점수_AHP'], samp['반경100m_화재수'], 1)
        x_line = np.linspace(samp['위험점수_AHP'].min(), samp['위험점수_AHP'].max(), 100)

        fig_sc = px.scatter(
            samp, x='위험점수_AHP', y='반경100m_화재수',
            opacity=0.3, color_discrete_sequence=['#3B82F6'],
            labels={'위험점수_AHP': 'AHP 위험점수', '반경100m_화재수': '실제 화재수(100m)'},
            title=f'AHP vs 화재수  (r={corr_val:.3f})',
        )
        fig_sc.add_trace(go.Scatter(
            x=x_line, y=np.polyval(slope, x_line), mode='lines',
            line=dict(color='#EF4444', width=2), name='추세선',
        ))
        fig_sc.update_layout(height=400, plot_bgcolor='#FAFAFA', paper_bgcolor='white',
                             margin=dict(t=40, b=10))
        st.plotly_chart(fig_sc, use_container_width=True)

    # ── AHP 분위별 화재수 박스 ──
    with col_r:
        st.subheader("② AHP 위험 분위별 평균 화재수")
        st.caption("위험지수를 5분위로 나눠서 실제 화재수 비교")

        plot_df['AHP_분위'] = pd.qcut(plot_df['위험점수_AHP'], q=5,
                                      labels=['최저위험(1)', '저위험(2)', '중간(3)', '고위험(4)', '최고위험(5)'])
        grp = plot_df.groupby('AHP_분위', observed=True)['반경100m_화재수'].mean().reset_index()

        fig_box = px.bar(
            grp, x='AHP_분위', y='반경100m_화재수',
            color='반경100m_화재수', color_continuous_scale=['#DBEAFE', '#1D4ED8'],
            text='반경100m_화재수',
            title='AHP 위험 분위별 평균 화재수',
            labels={'반경100m_화재수': '평균 화재수(100m)', 'AHP_분위': 'AHP 위험 분위'},
        )
        fig_box.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig_box.update_layout(height=400, plot_bgcolor='#FAFAFA', paper_bgcolor='white',
                              coloraxis_showscale=False, margin=dict(t=40, b=10))
        st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("---")

    # ── 지도 겹치기 ──
    st.subheader("③ 공간 지도 비교 — AHP 위험점수 vs 실제 화재율")
    st.caption("두 지도의 고위험 지역이 겹치는지 확인 → AHP 지수의 공간 타당성 검증")

    col_m1, col_m2 = st.columns(2)

    with col_m1:
        ahp_max = float(plot_df['위험점수_AHP'].quantile(0.95))
        fig_ahp = px.scatter_mapbox(
            plot_df.sample(min(2000, len(plot_df)), random_state=1),
            lat='위도', lon='경도',
            color='위험점수_AHP',
            color_continuous_scale='YlOrRd',
            range_color=[plot_df['위험점수_AHP'].quantile(0.05), ahp_max],
            size_max=7, mapbox_style='carto-positron',
            zoom=10.5, center=dict(lat=37.53, lon=126.99),
            opacity=0.75, labels={'위험점수_AHP': 'AHP 위험점수'},
            title='이론적 위험지수 (AHP)',
        )
        fig_ahp.update_layout(height=430, margin=dict(t=40, b=10))
        st.plotly_chart(fig_ahp, use_container_width=True)

    with col_m2:
        rate_max = float(plot_df['화재율'].quantile(0.95))
        fig_rate = px.scatter_mapbox(
            plot_df.sample(min(2000, len(plot_df)), random_state=1),
            lat='위도', lon='경도',
            color='화재율',
            color_continuous_scale='YlOrRd',
            range_color=[plot_df['화재율'].quantile(0.05), rate_max],
            size_max=7, mapbox_style='carto-positron',
            zoom=10.5, center=dict(lat=37.53, lon=126.99),
            opacity=0.75, labels={'화재율': '건물당 화재율'},
            title='실제 건물당 화재율 (offset 보정)',
        )
        fig_rate.update_layout(height=430, margin=dict(t=40, b=10))
        st.plotly_chart(fig_rate, use_container_width=True)

    # ── 구별 비교 ──
    st.subheader("④ 구별 AHP 위험점수 vs 화재수 비교")
    gu_ahp  = base.groupby('구')['위험점수_AHP'].mean().reset_index()
    gu_fire = base.groupby('구')['반경100m_화재수'].mean().reset_index()
    gu_both = pd.merge(gu_ahp, gu_fire, on='구')

    fig_gu = px.scatter(
        gu_both, x='위험점수_AHP', y='반경100m_화재수',
        text='구', size=[20]*len(gu_both),
        color='반경100m_화재수', color_continuous_scale=['#DBEAFE', '#1D4ED8'],
        labels={'위험점수_AHP': '평균 AHP 위험점수', '반경100m_화재수': '평균 화재수'},
        title='구별 평균 AHP 위험점수 vs 평균 화재수',
    )
    fig_gu.update_traces(textposition='top center')
    fig_gu.update_layout(height=420, plot_bgcolor='#FAFAFA', paper_bgcolor='white',
                         coloraxis_showscale=False, margin=dict(t=40, b=10))
    st.plotly_chart(fig_gu, use_container_width=True)

    st.info(
        f"**STEP 3 해석**\n\n"
        f"- AHP↔화재수 상관 r={corr_val:.3f}: 이론 위험지수가 높을수록 실제 화재도 증가하는 경향\n"
        f"- AHP↔화재율 상관 r={corr_rate:.3f}: offset 보정 후에도 경향 유지\n"
        f"- 두 지도의 고위험 지역이 공간적으로 일치 → **AHP 위험지수의 현실 타당성 확인**\n"
        f"- R²가 낮은 이유: 현실 화재는 우발적 요인 포함 — 이론과 완전 일치는 불가능"
    )

# ════════════════════════════════════
# TAB 6 │ 해석 가이드
# ════════════════════════════════════
with tab6:
    st.subheader("분석 결과 해석 가이드")
    st.markdown(f"""
#### 📌 모델 선택 근거 (Y=log(화재수+1) 기준)
| 모델 | R² | 판단 |
|---|---|---|
| OLS (6변수) | 0.032 | ❌ 공간 독립 가정 위반 |
| Spatial Lag | 0.610 | ✅ 인접 효과 반영 시 향상 (ρ=0.937) |
| Spatial Error | — | ✅ AIC 기준 우수 (λ=0.828) |
| **GWR (BW=68)** | **0.570** | ✅✅ 지역별 계수 적용 |
| **MGWR** | **0.364** | ✅ 주변건물수만 국소(BW=44) |

---
#### 📌 주변건물수 처리 방식
- **통제변수로 포함**: 주변건물수를 X에 넣으면 다른 계수들이 건물 수를 보정한 후의 순수 효과를 추정
- **결과**: MGWR에서 주변건물수 BW=44 (유일하게 국소) → 건물 밀집 효과만 공간적으로 이질적
- **나머지 위험 요인들은 전역 균일** → 구조노후도·단속위험도 등은 어느 지역이나 동일하게 작용

---
#### 📌 튜닝 결과 요약
| 항목 | 이전 (AHP 기준) | 최신 (화재수 기준) | 개선 |
|---|---|---|---|
| Y 변수 | AHP (순환 논리) | **log(화재수+1)** | 현실 타당성 |
| X 변수 | 6개 | **6개** (주변건물수=통제변수) | 건물수 효과 분리 |
| GWR BW | 245 NN | **68 NN** | 실제 데이터 기준 최적 |
| GWR R² | 0.859 (과장) | **0.570** | 정직한 설명력 |
| MGWR 국소 변수 | 없음(전역) | **주변건물수만** BW=44 | 의미 있는 발견 |

---
#### 📌 MGWR 핵심 발견
| 변수 | BW | 의미 |
|---|---|---|
| **주변건물수** | **44** | 골목 단위로 가장 국소적 — 밀도 효과 |
| 집중도 | 484 | 전역 균일 |
| 구조노후도 | 498 | 전역 균일 |
| 단속위험도 | 498 | 전역 균일 |
| 도로폭위험도 | 498 | 전역 균일 |
| 소방위험도_점수 | 498 | 전역 균일 |
    """)
