# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="GWR 공간분석", page_icon="🌡️", layout="wide")
st.title("🌡️ GWR 지역별 위험 계수 분석")
st.caption("Geographically Weighted Regression — 위치마다 다른 위험변수 효과 시각화")

@st.cache_data
def load_gwr():
    df = pd.read_csv('data/gwr_results.csv', encoding='utf-8-sig')
    return df

gwr = load_gwr()

# 변수 목록을 CSV에서 동적으로 추출
RISK_VARS = [c.replace('coef_', '') for c in gwr.columns if c.startswith('coef_')]

# ── 모델 요약 메트릭 ──
st.markdown("### 모델 성능 비교 (튜닝 완료 — 6변수 + BW 최적화)")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("GWR R²",        "0.8593", "adj 0.8483")
c2.metric("Bandwidth",     "245 NN", help="AICc 최소화 기준 최적 BW (기존 54 → 245)")
c3.metric("소방위험도 계수", "+13.8",  "최대 영향 변수")
c4.metric("Ridge R²",      "0.131",  delta_color="inverse")
c5.metric("VIF 최대",       "1.55",   help="모든 변수 VIF < 2 → 다중공선성 없음")

st.info(
    "**튜닝 결과 (6변수 + BW=245):**\n\n"
    "- 소방위험도_점수 추가 → 계수 평균 **+13.84** (6개 중 가장 큰 영향)\n"
    "- VIF 전 변수 < 2 → 상관성 낮아 6변수 전부 유효\n"
    "- Bandwidth: 54 → **245** (AICc 13386 → **13151**로 감소, 과적합 해소)\n"
    "- MGWR(변수별 BW): 계산량 과다 → 개념 적용(구조노후도=광역, 단속위험도=좁은 범위)"
)

st.markdown("---")

# ── 탭 구성 ──
tab1, tab2, tab3, tab4 = st.tabs(["📍 지역 R² 지도", "📊 계수 지도", "📈 계수 분포", "🔍 해석 가이드"])

with tab1:
    st.subheader("지역별 설명력 (Local R²)")
    st.caption("색이 진할수록 해당 위치에서 GWR 모델이 잘 맞는 지역")

    fig = px.scatter_mapbox(
        gwr, lat='위도', lon='경도',
        color='local_R2', size_max=8,
        color_continuous_scale='RdYlGn',
        range_color=[0, 1],
        mapbox_style='carto-positron',
        zoom=10.5, center=dict(lat=37.53, lon=126.99),
        opacity=0.75,
        labels={'local_R2': 'Local R²'},
        title='GWR 지역별 Local R²',
    )
    fig.update_layout(height=550, margin=dict(t=30, b=0))
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("평균 Local R²", f"{gwr['local_R2'].mean():.3f}")
        st.metric("최소 Local R²", f"{gwr['local_R2'].min():.3f}")
    with col2:
        st.metric("최대 Local R²", f"{gwr['local_R2'].max():.3f}")
        st.metric("R²>0.8 비율", f"{(gwr['local_R2']>0.8).mean()*100:.1f}%")

with tab2:
    st.subheader("변수별 GWR 계수 지도")
    st.caption("양수(붉은색) = 해당 지역에서 위험도 높이는 방향, 음수(파란색) = 낮추는 방향")

    sel_var = st.selectbox("변수 선택", RISK_VARS,
                           format_func=lambda x: {
                               '구조노후도':'구조노후도 (건물 노후화)',
                               '단속위험도':'단속위험도 (불법주정차)',
                               '도로폭위험도':'도로폭위험도 (좁은 도로)',
                               '집중도':'집중도 (주변 시설 밀집)',
                               '주변건물수':'주변건물수 (인근 건물 수)',
                           }[x])

    col_name = f'coef_{sel_var}'
    tval_col = f'tval_{sel_var}'
    abs_max  = gwr[col_name].abs().quantile(0.95)

    # 유의한 점만 크기 강조 (|t|>1.96)
    gwr_plot = gwr.copy()
    gwr_plot['유의'] = gwr_plot[tval_col].abs() > 1.96
    gwr_plot['size'] = gwr_plot['유의'].map({True: 8, False: 4})

    fig2 = px.scatter_mapbox(
        gwr_plot, lat='위도', lon='경도',
        color=col_name,
        color_continuous_scale='RdBu_r',
        range_color=[-abs_max, abs_max],
        size='size', size_max=10,
        mapbox_style='carto-positron',
        zoom=10.5, center=dict(lat=37.53, lon=126.99),
        opacity=0.8,
        labels={col_name: f'{sel_var} 계수'},
        title=f'GWR 계수 지도 — {sel_var}  (큰 점 = |t|>1.96 유의)',
    )
    fig2.update_layout(height=550, margin=dict(t=30, b=0))
    st.plotly_chart(fig2, use_container_width=True)

    sig_pct = (gwr_plot[tval_col].abs() > 1.96).mean() * 100
    st.caption(f"통계적으로 유의한 지점: {sig_pct:.1f}%  (|t| > 1.96)")

with tab3:
    st.subheader("변수별 계수 분포 비교")
    st.caption("계수의 지역 간 변동폭이 클수록 해당 변수의 효과가 지역마다 다름")

    rows = []
    for v in RISK_VARS:
        c = gwr[f'coef_{v}']
        rows.append({
            '변수': v, '평균': c.mean(), '중앙값': c.median(),
            '표준편차': c.std(), '최솟값': c.min(), '최댓값': c.max(),
            'IQR': c.quantile(0.75) - c.quantile(0.25),
        })
    stat_df = pd.DataFrame(rows).round(3)
    st.dataframe(stat_df.set_index('변수'), use_container_width=True)

    # 박스플롯
    melt = pd.DataFrame({v: gwr[f'coef_{v}'] for v in RISK_VARS}).melt(var_name='변수', value_name='계수')
    fig3 = px.violin(melt, x='변수', y='계수', box=True, points=False,
                     color='변수', color_discrete_sequence=px.colors.qualitative.Bold,
                     title='변수별 GWR 계수 분포 (바이올린)')
    fig3.add_hline(y=0, line_dash='dash', line_color='gray')
    fig3.update_layout(showlegend=False, xaxis_title=None)
    st.plotly_chart(fig3, use_container_width=True)

    # 계수 간 상관
    coef_cols = [f'coef_{v}' for v in RISK_VARS]
    corr = gwr[coef_cols].rename(columns={f'coef_{v}': v for v in RISK_VARS}).corr()
    fig4 = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r',
                     zmin=-1, zmax=1, title='GWR 계수 간 공간 상관관계')
    st.plotly_chart(fig4, use_container_width=True)

with tab4:
    st.subheader("분석 결과 해석 가이드")

    st.markdown("""
#### 📌 모델 선택 근거
| 모델 | R² | 판단 |
|---|---|---|
| Ridge/Lasso | 0.13 | ❌ 위험변수만으로 설명 불충분 |
| Spatial Lag | 0.77 | ✅ 인접 효과 반영 시 향상 (ρ=0.86) |
| Spatial Error | — | ✅ AIC 기준 Lag보다 659 낮음 → **우선 선택** |
| **GWR** | **0.80** | ✅✅ 지역별 계수 적용 시 최고 설명력 |

---
#### 📌 핵심 해석
- **Moran's I = 0.76** → 위험도는 공간적으로 강하게 뭉침.
  같은 구 안에서도 골목 단위로 위험 격차가 존재함.
- **Spatial Error λ = 0.89** → 모델에 포함 안 된 인접 요인(골목 환경, 주변 건물 용도 등)이
  위험도에 89% 수준으로 전이됨.
- **GWR 계수 변동** → 예를 들어 `구조노후도` 계수가 어떤 지역은 +28, 어떤 지역은 -9.
  즉, 건물 노후도가 위험도에 미치는 영향 자체가 위치에 따라 크게 다름.

---
#### 📌 튜닝 결과 요약
| 항목 | 기존 | 튜닝 후 | 개선 |
|---|---|---|---|
| 변수 수 | 5개 | **6개** (소방위험도_점수 추가) | 최대 영향 변수 추가 |
| GWR BW | 54 NN | **245 NN** (Sel_BW 자동) | AICc 13386→13151 ↓ |
| GWR R² | 0.80 | **0.86** | adj.R² 0.74→0.85 |
| 다중공선성 | 미확인 | VIF 최대 1.55 → **안전** | 6변수 전부 유효 |
| MGWR | 미실시 | BW 선택 완료 (변수별 상이) | fit은 계산량 제약 |

#### 📌 MGWR 개념 해석
- 구조노후도: **넓은 BW** → 광역 지역 특성 반영
- 단속위험도: **좁은 BW** → 단기·국소 효과 (골목 단위)
- 소방위험도_점수: **중간 BW** → 건물 자체 특성 (인근 소방서 기반)

---
#### 📌 앙상블 필요성
- **핵심 분석에는 불필요** — 클러스터 레이블을 같은 변수로 만들었기 때문에
  높은 정확도(95~98%)는 당연한 결과.
- **활용 가치**: 신규 시설이 들어올 때 어느 군집에 속할지 예측하는 **분류 도구**로는 유효.
- 논문/발표에서 강조할 핵심은 **GWR + Spatial Error** 체인.
    """)
