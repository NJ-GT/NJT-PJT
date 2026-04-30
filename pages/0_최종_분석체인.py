# -*- coding: utf-8 -*-
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(page_title="최종 분석 체인", page_icon="🧭", layout="wide")
st.title("최종 분석 체인")
st.caption("1. 업종별 군집화 → 2. Ridge/Lasso → 3. OLS+Moran's I → 4. Spatial Lag/Error → 5. GWR/MGWR → 6. 최종 위험시설 순위")

BASE = Path(__file__).resolve().parents[1]
PIPE = BASE / "data" / "final_spatial_pipeline"


@st.cache_data
def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(PIPE / name, encoding="utf-8-sig")


if not PIPE.exists():
    st.error("최종 분석 산출물이 없습니다. `python scripts/run_final_spatial_pipeline.py`를 먼저 실행하세요.")
    st.stop()

dataset = load_csv("analysis_dataset.csv")
cluster_summary = load_csv("step1_cluster_summary.csv")
ridge_lasso = load_csv("step2_ridge_lasso_coefficients.csv")
ols_moran = load_csv("step3_ols_moran.csv")
spatial = load_csv("step4_spatial_lag_error.csv")
gwr = load_csv("step5_gwr_mgwr_summary.csv")
rank = load_csv("step6_final_facility_rank.csv")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "1 업종별 군집화",
        "2 Ridge/Lasso",
        "3 OLS + Moran",
        "4 Spatial Lag/Error",
        "5 GWR/MGWR",
        "6 위험시설 순위",
    ]
)

with tab1:
    st.subheader("업종별 군집화")
    st.write(
        "전체 4천여 개를 한 번에 묶지 않고, 업종별로 따로 KMeans를 적용했습니다. "
        "군집은 위험요인 패턴을 나누는 용도이며, 기대피해액은 군집 해석 지표로 사용합니다."
    )
    metric_cols = st.columns(3)
    for i, group in enumerate(["관광숙박업", "숙박업", "외국인관광도시민박업"]):
        sub = cluster_summary[cluster_summary["업종"] == group]
        if len(sub):
            metric_cols[i].metric(group, f"{sub['시설수'].sum():,}개", f"K={int(sub['선택_K'].iloc[0])}")

    fig = px.bar(
        cluster_summary,
        x="업종",
        y="시설수",
        color="업종별_군집",
        barmode="stack",
        title="업종별 군집 구성",
        color_continuous_scale="Turbo",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(cluster_summary, hide_index=True, use_container_width=True)

with tab2:
    st.subheader("Ridge/Lasso 변수 선택")
    st.write("타깃은 `log1p(반경100m_화재수)`입니다. Lasso에서 0이 아닌 변수는 비교적 안정적으로 남은 변수로 봅니다.")
    selected = ridge_lasso.copy()
    selected["선택여부"] = selected["lasso_selected"].map({True: "선택", False: "제외"})
    fig = px.bar(
        selected,
        x="변수",
        y="lasso_coef",
        color="업종",
        barmode="group",
        title="Lasso 계수",
    )
    fig.update_layout(xaxis_title=None, yaxis_title="표준화 계수")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(selected, hide_index=True, use_container_width=True)

with tab3:
    st.subheader("OLS + Moran's I")
    st.write("OLS는 전역 유의성을 보고, Moran's I는 OLS 잔차에 공간자기상관이 남는지 확인합니다.")
    overall = ols_moran[ols_moran["업종"] == "전체"].copy()
    m1, m2, m3 = st.columns(3)
    m1.metric("전체 OLS R²", f"{overall['ols_r2'].iloc[0]:.3f}")
    m2.metric("잔차 Moran's I", f"{overall['moran_I_residual'].iloc[0]:.3f}")
    m3.metric("Moran p-value", f"{overall['moran_p_sim'].iloc[0]:.3f}")

    sig = ols_moran[(ols_moran["term"] != "const") & (ols_moran["significant_0_05"])]
    fig = px.bar(
        sig,
        x="term",
        y="coef",
        color="업종",
        barmode="group",
        title="OLS 유의 변수 계수(p<0.05)",
    )
    fig.update_layout(xaxis_title=None, yaxis_title="표준화 계수")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(ols_moran, hide_index=True, use_container_width=True)

with tab4:
    st.subheader("Spatial Lag / Spatial Error")
    st.write(
        "OLS 잔차의 공간자기상관이 유의하므로 공간회귀를 적용했습니다. "
        "표본이 큰 경우 대시보드 산출 안정성을 위해 일부 샘플링된 요약값입니다."
    )
    fig = px.bar(
        spatial,
        x="업종",
        y="pseudo_r2",
        color="model",
        barmode="group",
        title="Spatial Lag/Error pseudo R²",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(spatial, hide_index=True, use_container_width=True)

with tab5:
    st.subheader("GWR/MGWR")
    st.write(
        "기존 GWR 결과 파일을 연결해 지역별 계수 분포를 요약했습니다. "
        "MGWR은 변수별 공간 스케일을 분리해 보는 확장 단계로, 최종 발표에서는 GWR 결과를 중심으로 제시하는 편이 안전합니다."
    )
    if "metric" in gwr.columns:
        local_r2 = gwr[gwr["metric"].astype(str).str.contains("local_R2", na=False)]
        if len(local_r2):
            c1, c2, c3 = st.columns(3)
            c1.metric("local R² 평균", f"{local_r2['mean'].iloc[0]:.3f}")
            c2.metric("local R² 최소", f"{local_r2['min'].iloc[0]:.3f}")
            c3.metric("local R² 최대", f"{local_r2['max'].iloc[0]:.3f}")
    st.dataframe(gwr, hide_index=True, use_container_width=True)

with tab6:
    st.subheader("최종 위험시설 순위")
    st.write(
        "기대피해액은 절대 금액 예측보다 상대 위험 순위로 해석합니다. "
        "AHP 위험점수와 팀 공식의 사각지대 위험도는 서로 다른 관점의 보조 지표로 함께 봅니다."
    )
    top_n = st.slider("상위 시설 수", 10, 100, 30, step=10)
    table_cols = [
        "최종위험순위",
        "AHP위험순위",
        "사각지대_위험순위",
        "기대피해액순위",
        "구",
        "동",
        "숙소명",
        "업종",
        "업종별_군집명",
        "주변건물수_검증상태",
        "주변건물수_보정출처",
        "기대피해액_백만원",
        "예상_화재발생확률",
        "조건부_예상피해액_백만원",
        "위험점수_AHP",
        "사각지대_위험도점수",
        "소방위험도_점수",
        "도로폭위험도",
        "단속위험도",
    ]
    table_cols = [c for c in table_cols if c in rank.columns]
    st.dataframe(rank[table_cols].head(top_n), hide_index=True, use_container_width=True)

    chart = rank.head(top_n).copy()
    chart["시설라벨"] = chart["구"] + " " + chart["숙소명"].astype(str)
    fig = px.bar(
        chart.sort_values("최종위험순위", ascending=False),
        x="기대피해액_백만원",
        y="시설라벨",
        color="업종",
        orientation="h",
        title=f"최종 위험시설 상위 {top_n}개",
        hover_data=["예상_화재발생확률", "위험점수_AHP", "사각지대_위험도점수", "업종별_군집명"],
    )
    fig.update_layout(xaxis_title="기대피해액(백만원)", yaxis_title=None, height=720)
    st.plotly_chart(fig, use_container_width=True)
