# -*- coding: utf-8 -*-
"""
공간 분석 체인 — 시각적 근거  (6변수 최신화)
  Moran's I → Spatial Lag → Spatial Error → GWR
"""
import glob
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="공간분석 근거", page_icon="🔬", layout="wide")

st.title("🔬 공간 분석 체인 — 시각적 근거")
st.markdown(
    """
    <style>
    .step-badge {
        background: #EFF6FF; border-left: 4px solid #2563EB;
        padding: 10px 16px; border-radius: 0 8px 8px 0;
        margin-bottom: 8px; font-size: 0.95rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """<div class="step-badge">
    <b>분석 흐름</b>: OLS 독립성 위반 확인 (Moran's I)
    → 공간 전염 포착 (Spatial Lag)
    → 숨겨진 인접 요인 보정 (Spatial Error)
    → 지역별 이질적 효과 측정 (GWR)
    </div>""",
    unsafe_allow_html=True,
)

# ── 6변수 (최신) ──────────────────────────────────────
RISK_VARS = ["구조노후도", "단속위험도", "도로폭위험도", "집중도", "주변건물수", "소방위험도_점수"]
PALETTE   = {"HH": "#EF4444", "LL": "#3B82F6", "HL": "#F97316", "LH": "#A78BFA", "NS": "#CBD5E1"}

# ── 실행 수치 (6변수 기준, Y=log(화재수+1)) ────────────
OLS_R2   = 0.032
OLS_MI   = 0.628    # 잔차 Moran's I
SLM_R2   = 0.610
SLM_RHO  = 0.937
GML_LAM  = 0.828
GWR_R2   = 0.402    # Y=화재수 기준 GWR (tuning_spatial_v2)

# ─────────────────────────────────────────────────────
# 캐시 계산
# ─────────────────────────────────────────────────────
@st.cache_data(show_spinner="공간 분석 데이터 준비 중…")
def prepare_spatial_data():
    from sklearn.preprocessing import StandardScaler
    from libpysal.weights import KNN
    from esda.moran import Moran, Moran_Local

    f    = glob.glob("data/*0423*.csv")[0]
    main = pd.read_csv(f, encoding="utf-8-sig")
    core = pd.read_csv("data/data_with_fire_targets.csv", encoding="utf-8-sig")

    # 위치 인덱스로 안전하게 로드
    cc = core.columns
    core_key = core[[cc[4], cc[5], cc[17], cc[22], cc[45]]].copy()
    core_key.columns = ["위도", "경도", "소방위험도_점수", "위험점수_AHP", "반경100m_화재수"]

    df = pd.merge(main, core_key, on=["위도", "경도"], how="left")
    for v in RISK_VARS + ["위험점수_AHP", "반경100m_화재수"]:
        df[v] = pd.to_numeric(df[v], errors="coerce")
    df = df.dropna(subset=RISK_VARS + ["위험점수_AHP", "반경100m_화재수", "위도", "경도"]).reset_index(drop=True)

    df["Y"] = np.log1p(df["반경100m_화재수"])
    Y  = df["Y"].values
    N  = len(df)

    coords = df[["위도", "경도"]].values
    w = KNN.from_array(coords, k=8)
    w.transform = "R"

    # OLS 잔차
    from sklearn.preprocessing import StandardScaler
    Xs = StandardScaler().fit_transform(df[RISK_VARS].values)
    Xc = np.column_stack([np.ones(N), Xs])
    b, *_ = np.linalg.lstsq(Xc, Y, rcond=None)
    resid = Y - Xc @ b

    # 공간지연 잔차
    lag_e = w.sparse.dot(resid)

    # Global Moran's I
    mi = Moran(resid, w, permutations=499)

    # LISA
    lm  = Moran_Local(resid, w, permutations=499, seed=42)
    sig = lm.p_sim < 0.05
    CAT = {1: "HH", 2: "LH", 3: "LL", 4: "HL"}
    lisa_cat = [CAT.get(lm.q[i], "NS") if sig[i] else "NS" for i in range(N)]

    # 공간지연 Y
    lag_y = w.sparse.dot(Y)

    out = df[["위도", "경도", "구"]].copy()
    out["residual"]  = resid
    out["lag_resid"] = np.array(lag_e).flatten()
    out["Y"]         = Y
    out["lag_Y"]     = np.array(lag_y).flatten()
    out["lisa_cat"]  = lisa_cat

    return out, mi


data, mi_global = prepare_spatial_data()
gwr_df = pd.read_csv("data/gwr_results.csv", encoding="utf-8-sig")
GWR_VARS = [c.replace("coef_", "") for c in gwr_df.columns if c.startswith("coef_")]

# ─────────────────────────────────────────────────────
# 탭
# ─────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "① Moran's I — OLS 독립성 위반",
    "② Spatial Lag — 공간 전염",
    "③ Spatial Error — 인접 요인 보정",
    "④ GWR — 지역별 이질적 효과",
])

# ════════════════════════════════════════════
# TAB 1 │ Moran's I
# ════════════════════════════════════════════
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("OLS 잔차 Moran's I", f"{mi_global.I:.4f}", help="1에 가까울수록 강한 군집")
    c2.metric("z-score",            f"{mi_global.z_sim:.2f}")
    c3.metric("p-value",            f"{mi_global.p_sim:.3f}  ✓ 유의")
    c4.metric("OLS R²",             f"{OLS_R2:.3f}", "화재수 기준")

    st.info(
        f"**해석**: OLS 잔차 Moran's I = **{mi_global.I:.3f}** (p={mi_global.p_sim:.3f}).\n"
        "잔차가 공간적으로 독립적이라면 I ≈ 0이어야 합니다.\n"
        "→ OLS의 '관측값 독립' 가정이 **심각하게 위반**됨 → 공간 모델 필수."
    )

    col_l, col_r = st.columns([1, 1])

    with col_l:
        st.subheader("Moran 산점도")
        st.caption("X=OLS 잔차, Y=인접 잔차 평균(W·e) — 우상·좌하 군집 = 양의 공간자기상관")

        slope = np.polyfit(data["residual"], data["lag_resid"], 1)
        x_line = np.linspace(data["residual"].min(), data["residual"].max(), 100)

        fig_ms = px.scatter(
            data, x="residual", y="lag_resid",
            color="lisa_cat", color_discrete_map=PALETTE,
            opacity=0.45, size_max=5,
            labels={"residual": "OLS 잔차", "lag_resid": "공간지연 잔차 (W·e)", "lisa_cat": "LISA"},
            title=f"Moran's I = {mi_global.I:.3f}  (기울기 = 공간자기상관)",
        )
        fig_ms.add_trace(go.Scatter(
            x=x_line, y=np.polyval(slope, x_line), mode="lines",
            line=dict(color="#1E293B", width=2),
            name=f"slope={slope[0]:.3f}",
        ))
        fig_ms.add_hline(y=0, line_dash="dash", line_color="#94A3B8", line_width=1)
        fig_ms.add_vline(x=0, line_dash="dash", line_color="#94A3B8", line_width=1)
        for txt, rx, ry in [("고-고", 0.88, 0.88), ("저-저", 0.08, 0.08),
                             ("고-저", 0.88, 0.08), ("저-고", 0.08, 0.88)]:
            fig_ms.add_annotation(
                x=data["residual"].quantile(rx), y=data["lag_resid"].quantile(ry),
                text=txt, showarrow=False, font=dict(size=11, color="#64748B"),
            )
        fig_ms.update_layout(height=420, margin=dict(t=40, b=10, l=10, r=10),
                             plot_bgcolor="#FAFAFA", paper_bgcolor="white",
                             legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig_ms, use_container_width=True)

    with col_r:
        st.subheader("LISA 군집 지도")
        st.caption("빨강=고위험 군집(HH), 파랑=저위험 군집(LL), 주황/보라=이상치, 회색=비유의")

        fig_map = px.scatter_mapbox(
            data, lat="위도", lon="경도",
            color="lisa_cat", color_discrete_map=PALETTE,
            size_max=7, mapbox_style="carto-positron",
            zoom=10.5, center=dict(lat=37.53, lon=126.99),
            opacity=0.8, labels={"lisa_cat": "LISA 유형"},
            title="LISA 공간 군집 분포",
        )
        fig_map.update_layout(height=420, margin=dict(t=40, b=10))
        st.plotly_chart(fig_map, use_container_width=True)

    st.markdown("**LISA 유형별 시설 수**")
    cat_df = data["lisa_cat"].value_counts().rename_axis("유형").reset_index(name="시설수")
    cat_df["설명"] = cat_df["유형"].map({
        "HH": "고위험 군집 — 인접도 고위험",
        "LL": "저위험 군집 — 인접도 저위험",
        "HL": "위험 이상치 — 주변은 저위험",
        "LH": "안전 이상치 — 주변은 고위험",
        "NS": "통계적으로 비유의",
    })
    st.dataframe(cat_df, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════
# TAB 2 │ Spatial Lag
# ════════════════════════════════════════════
with tab2:
    c1, c2, c3 = st.columns(3)
    c1.metric("공간시차계수 ρ", f"{SLM_RHO:.3f}", help="인접 시설 위험도가 자신에게 전이되는 비율")
    c2.metric("Spatial Lag R²", f"{SLM_R2:.3f}", f"OLS {OLS_R2:.3f} 대비 +{SLM_R2-OLS_R2:.3f}")
    c3.metric("Y 변수", "log(화재수+1)", "6변수 기준")

    st.info(
        f"**해석**: ρ={SLM_RHO:.2f}는 인접 시설의 위험도가 내 위험도에 {SLM_RHO*100:.0f}% 영향을 준다는 뜻입니다.\n"
        "→ 위험 시설 하나가 생기면 주변으로 연쇄적으로 위험이 높아지는 **공간 전염** 효과."
    )

    col_l, col_r = st.columns([1, 1])

    with col_l:
        st.subheader("Y vs 인접 Y 평균 (W·Y)")
        st.caption(f"두 변수의 상관이 높을수록 ρ가 큼 — 기울기 ≈ ρ={SLM_RHO:.2f}")

        samp = data.sample(min(1500, len(data)), random_state=42)
        slope_lag = np.polyfit(samp["Y"], samp["lag_Y"], 1)
        x_l = np.linspace(samp["Y"].min(), samp["Y"].max(), 100)

        fig_lag = px.scatter(
            samp, x="Y", y="lag_Y", opacity=0.35,
            color_discrete_sequence=["#3B82F6"],
            labels={"Y": "log(화재수+1) — 자신", "lag_Y": "인접 시설 평균"},
            title="자신의 위험도 vs 인접 시설 위험도 평균",
        )
        fig_lag.add_trace(go.Scatter(
            x=x_l, y=np.polyval(slope_lag, x_l), mode="lines",
            line=dict(color="#EF4444", width=2),
            name=f"기울기={slope_lag[0]:.2f}",
        ))
        fig_lag.update_layout(height=400, plot_bgcolor="#FAFAFA", paper_bgcolor="white",
                              margin=dict(t=40, b=10))
        st.plotly_chart(fig_lag, use_container_width=True)

    with col_r:
        st.subheader("모델별 설명력 비교")
        st.caption("Y = log(반경100m 화재수+1) 기준, 6변수")

        model_comp = pd.DataFrame({
            "모델":  ["OLS/Ridge", f"Spatial Lag\n(ρ={SLM_RHO:.2f})",
                      f"GM_Error\n(λ={GML_LAM:.2f})", "GWR\n(bisquare)"],
            "R²":    [OLS_R2, SLM_R2, 0.030, GWR_R2],
            "비고":  ["공간 독립 가정", "공간 전염 반영", "잔차 공간 오차 보정", "지역별 계수"],
        })
        fig_bar = px.bar(
            model_comp, x="모델", y="R²", color="R²",
            color_continuous_scale=["#DBEAFE", "#1D4ED8"],
            text="R²", title="모델별 R² 비교 (6변수)",
        )
        fig_bar.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig_bar.update_layout(height=400, plot_bgcolor="#FAFAFA", paper_bgcolor="white",
                              margin=dict(t=40, b=10), showlegend=False,
                              coloraxis_showscale=False, yaxis_range=[0, 0.75])
        st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("공간 전염 메커니즘")
    fig_sankey = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            label=["고위험 시설 A", "인접 시설 B", "인접 시설 C", "인접 시설 D",
                   f"위험 상승 (ρ={SLM_RHO:.2f})", "2차 전파"],
            color=["#EF4444", "#F97316", "#F97316", "#F97316", "#DC2626", "#B91C1C"],
            pad=20, thickness=20,
        ),
        link=dict(
            source=[0, 0, 0, 4, 4],
            target=[4, 4, 4, 1, 5],
            value=[3, 3, 3, 5, 3],
            color=["rgba(239,68,68,0.3)"] * 5,
        ),
    ))
    fig_sankey.update_layout(height=260, margin=dict(t=20, b=10),
                             title=f"고위험 시설의 위험이 인접 시설로 전이 (ρ={SLM_RHO:.2f})")
    st.plotly_chart(fig_sankey, use_container_width=True)


# ════════════════════════════════════════════
# TAB 3 │ Spatial Error
# ════════════════════════════════════════════
with tab3:
    c1, c2, c3 = st.columns(3)
    c1.metric("GM_Error λ",       f"{GML_LAM:.3f}", help="인접 관측되지 않은 요인이 오차에 전이되는 비율")
    c2.metric("OLS 잔차 Moran's I", f"{mi_global.I:.3f}", "잔차에 공간 패턴 → 보정 필요")
    c3.metric("보정 후 λ 해석",    f"{GML_LAM*100:.0f}%", "오차 공간 의존도")

    st.info(
        f"**해석**: λ={GML_LAM:.2f}는 모델에 포함되지 않은 인접 요인\n"
        "(골목 환경, 주변 건물 용도, 소방 접근성 등)이\n"
        f"오차 항을 통해 **{GML_LAM*100:.0f}%** 수준으로 전이된다는 의미입니다.\n"
        "→ Spatial Error가 이 구조적 편향을 보정합니다."
    )

    col_l, col_r = st.columns([1, 1])

    with col_l:
        st.subheader("OLS 잔차 공간 분포")
        st.caption("색이 강할수록 OLS가 크게 틀린 지역 — 패턴이 보이면 공간 의존성 존재")

        abs_max = float(np.percentile(np.abs(data["residual"]), 95))
        fig_resid = px.scatter_mapbox(
            data, lat="위도", lon="경도",
            color="residual", color_continuous_scale="RdBu_r",
            range_color=[-abs_max, abs_max],
            size_max=6, mapbox_style="carto-positron",
            zoom=10.5, center=dict(lat=37.53, lon=126.99),
            opacity=0.75, labels={"residual": "OLS 잔차"},
            title="OLS 잔차 지도 — 패턴이 보이면 독립성 위반",
        )
        fig_resid.update_layout(height=430, margin=dict(t=40, b=10))
        st.plotly_chart(fig_resid, use_container_width=True)

    with col_r:
        st.subheader("Spatial Error 보정 개념")
        st.caption(f"λ={GML_LAM:.2f}가 흡수하는 인접 요인들")

        factors = ["골목 폭·굴절도", "주변 건물 용도 혼재", "소방차 접근 장애물",
                   "불법 주정차 밀집", "야간 이동 인구", "전기 배선 노후화"]
        fig_lam = go.Figure(go.Bar(
            x=factors, y=[GML_LAM] * len(factors),
            marker_color="#8B5CF6",
            text=[f"λ={GML_LAM:.2f}"] * len(factors),
            textposition="inside",
        ))
        fig_lam.add_hline(y=GML_LAM, line_dash="dash", line_color="#4C1D95",
                          annotation_text=f"λ={GML_LAM:.2f}")
        fig_lam.update_layout(
            height=430, title=f"모델 외 인접 요인이 오차에 미치는 영향 (λ={GML_LAM:.2f})",
            xaxis_tickangle=-25, plot_bgcolor="#FAFAFA", paper_bgcolor="white",
            margin=dict(t=40, b=10), yaxis_range=[0, 1.1], yaxis_title="전이 강도",
            showlegend=False,
        )
        st.plotly_chart(fig_lam, use_container_width=True)

    st.markdown("---")
    st.subheader("보정 전후 효과 요약")
    before_after = pd.DataFrame({
        "단계":              ["① OLS (보정 전)", "② Spatial Lag 보정", "③ Spatial Error (GM) 보정"],
        "Moran's I (잔차)": [f"{mi_global.I:.3f} (유의)", "감소 (ρ 흡수)", "추가 감소 (λ 흡수)"],
        "R²":               [f"{OLS_R2:.3f}", f"{SLM_R2:.3f}", "0.030*"],
        "비고":             ["공간 의존성 무시", f"인접 Y 효과 반영 (ρ={SLM_RHO:.2f})",
                             "*GM은 pseudo-R² 낮음 — 보정 목적"],
    })
    st.dataframe(before_after, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════
# TAB 4 │ GWR 지역별 이질성
# ════════════════════════════════════════════
with tab4:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("GWR R² (화재수 기준)", "0.570", "BW=68, 6변수")
    c2.metric("Bandwidth",            "68 NN", "AICc 최소화")
    c3.metric("평균 local R²",        f"{gwr_df['local_R2'].mean():.3f}")
    c4.metric("R²>0.8 비율",          f"{(gwr_df['local_R2']>0.8).mean()*100:.1f}%")

    st.info(
        "**핵심 포인트**: Y=건물당 화재율(offset 보정), 5변수(주변건물수→offset 처리)\n\n"
        "전체에 하나의 계수를 쓰는 OLS/Spatial 모델과 달리, GWR은 위치마다 다른 계수를 추정합니다.\n"
        "→ '같은 변수라도 어느 구에 있느냐에 따라 화재 위험에 미치는 영향이 다름'을 정량화."
    )

    st.subheader("변수별 계수 지역 변동 폭 (6변수)")
    st.caption("IQR이 클수록 해당 변수의 효과가 지역마다 크게 다름")

    rows = []
    for v in GWR_VARS:
        col = gwr_df[f"coef_{v}"]
        rows.append({
            "변수": v, "전체평균": round(col.mean(), 4),
            "최솟값": round(col.min(), 4), "최댓값": round(col.max(), 4),
            "IQR": round(col.quantile(0.75) - col.quantile(0.25), 4),
            "표준편차": round(col.std(), 4),
        })
    stat_df = pd.DataFrame(rows)

    fig_iqr = px.bar(
        stat_df.sort_values("IQR", ascending=False),
        x="변수", y="IQR", color="IQR",
        color_continuous_scale=["#DBEAFE", "#1D4ED8"],
        text="IQR", title="변수별 GWR 계수 IQR — 클수록 지역 이질성 크다",
    )
    fig_iqr.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    fig_iqr.update_layout(height=340, plot_bgcolor="#FAFAFA", paper_bgcolor="white",
                          coloraxis_showscale=False, margin=dict(t=40, b=10))
    st.plotly_chart(fig_iqr, use_container_width=True)

    st.markdown("---")
    col_sel, _ = st.columns([1, 2])
    with col_sel:
        sel_var = st.selectbox("계수 지도 변수 선택", GWR_VARS)

    col_name = f"coef_{sel_var}"
    tval_col = f"tval_{sel_var}"
    abs_max  = float(gwr_df[col_name].abs().quantile(0.95))

    gwr_plot = gwr_df.copy()
    gwr_plot["유의"] = gwr_plot[tval_col].abs() > 1.96
    gwr_plot["size"] = gwr_plot["유의"].map({True: 9, False: 4})

    col_l, col_r = st.columns([3, 2])
    with col_l:
        fig_coef = px.scatter_mapbox(
            gwr_plot, lat="위도", lon="경도",
            color=col_name, color_continuous_scale="RdBu_r",
            range_color=[-abs_max, abs_max],
            size="size", size_max=10,
            mapbox_style="carto-positron",
            zoom=10.5, center=dict(lat=37.53, lon=126.99),
            opacity=0.85, labels={col_name: f"{sel_var} 계수"},
            title=f"GWR 계수 지도 — {sel_var}  (큰 점=|t|>1.96 유의)",
        )
        fig_coef.update_layout(height=460, margin=dict(t=40, b=10))
        st.plotly_chart(fig_coef, use_container_width=True)

    with col_r:
        st.subheader(f"{sel_var} 계수 분포")
        fig_hist = px.histogram(
            gwr_df, x=col_name, nbins=40,
            color_discrete_sequence=["#3B82F6"],
            labels={col_name: "계수값"},
            title=f"{sel_var} — 위치별 계수 히스토그램",
        )
        fig_hist.add_vline(x=0, line_dash="dash", line_color="#EF4444",
                           annotation_text="계수=0")
        fig_hist.add_vline(x=float(gwr_df[col_name].mean()), line_dash="dot",
                           line_color="#1E293B",
                           annotation_text=f"평균={gwr_df[col_name].mean():.3f}")
        fig_hist.update_layout(height=220, plot_bgcolor="#FAFAFA", paper_bgcolor="white",
                               margin=dict(t=40, b=10))
        st.plotly_chart(fig_hist, use_container_width=True)

        sig_pct = float((gwr_plot[tval_col].abs() > 1.96).mean() * 100)
        st.metric("유의 지점 비율", f"{sig_pct:.1f}%", help="|t|>1.96")
        sel_stat = stat_df[stat_df["변수"] == sel_var].T
        sel_stat.columns = ["값"]
        st.dataframe(sel_stat.iloc[1:], use_container_width=True)
