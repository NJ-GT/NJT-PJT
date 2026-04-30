# -*- coding: utf-8 -*-
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(page_title="팀 모델 검증", layout="wide")

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "data" / "team_pipeline_validation"


@st.cache_data
def load_data():
    summary = pd.read_json(OUT / "validation_summary.json", typ="series")
    scored = pd.read_csv(OUT / "team_pipeline_scored_dataset.csv", encoding="utf-8-sig")
    cluster = pd.read_csv(OUT / "01_cluster_fire_summary.csv", encoding="utf-8-sig")
    lasso = pd.read_csv(OUT / "02_lasso_coefficients.csv", encoding="utf-8-sig")
    top30 = pd.read_csv(OUT / "03_team_risk_top30.csv", encoding="utf-8-sig")
    moran = pd.read_csv(OUT / "04_moran_results.csv", encoding="utf-8-sig")
    rf = pd.read_csv(OUT / "06_rf_validation_metrics.csv", encoding="utf-8-sig")
    imp = pd.read_csv(OUT / "06_rf_circular_feature_importance.csv", encoding="utf-8-sig")
    ols = pd.read_csv(OUT / "07_ols_sanity.csv", encoding="utf-8-sig")
    return summary, scored, cluster, lasso, top30, moran, rf, imp, ols


if not OUT.exists() or not (OUT / "validation_summary.json").exists():
    st.error("검증 산출물이 없습니다. `python scripts/validate_team_pipeline.py`를 먼저 실행하세요.")
    st.stop()

summary, scored, cluster, lasso, top30, moran, rf, imp, ols = load_data()

st.title("팀 모델 검증 대시보드")
st.caption("군집화 → Lasso → 사각지대 위험도 → Moran's I → GWR → Random Forest 흐름을 실제로 검증")

st.warning(
    "핵심 결론: 최종 Random Forest의 높은 R²는 실제 화재나 피해액을 잘 맞힌다는 뜻이 아닙니다. "
    "우리가 만든 `위험도점수`를 그 점수를 만든 재료로 다시 맞힌 결과라서 높게 나온 것입니다."
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("숙박시설", f"{int(summary['rows']):,}개")
c2.metric("2021-2024 화재", f"{int(summary['fire_rows_2021_2024']):,}건")
c3.metric("150m 내 화재 매칭", f"{int(summary['facilities_with_fire_150m']):,}개")
c4.metric("평균 150m 화재수", f"{float(summary['mean_fire_count_150m']):.2f}건")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["한눈 결론", "군집화", "Lasso/위험점수", "공간분석", "GWR/RF 검증", "흠잡을 곳"]
)

with tab1:
    st.subheader("왜 성능이 좋아 보였나")
    st.write(
        "예를 들어 `국어점수 = 읽기점수 + 쓰기점수`로 만들어 놓고, 다시 읽기점수와 쓰기점수로 국어점수를 예측하면 "
        "당연히 성능이 높게 나옵니다. 여기서도 비슷합니다. `위험도점수`는 주변건물수, 집중도, 도로폭위험도, "
        "구조노후도, 단속위험도로 만든 점수입니다. 그런데 Random Forest가 다시 그 변수들로 `위험도점수`를 맞혔습니다."
    )
    display_rf = rf.copy()
    display_rf["R²"] = display_rf["r2"].round(3)
    display_rf["MAE"] = display_rf["mae"].round(3)
    fig = px.bar(
        display_rf,
        x="검증",
        y="R²",
        color="target",
        text="R²",
        title="좋아 보이는 검증 vs 진짜 검증",
    )
    fig.update_layout(xaxis_title=None, yaxis_range=[0, 1], height=520)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(display_rf[["검증", "target", "rows", "R²", "MAE"]], hide_index=True, use_container_width=True)

    st.info(
        "발표 문장 추천: '위험도점수 재현 모델의 설명력은 높았으나, 이는 산식 기반 점수를 재학습한 결과이다. "
        "실제 화재수와 피해액 예측 성능은 별도로 검증했으며, 예측보다는 위험요인 해석과 우선순위 도출에 활용한다.'"
    )

with tab2:
    st.subheader("군집화 결과")
    st.write(
        "군집화는 비슷한 성격의 숙소를 4묶음으로 나누는 작업입니다. "
        "학교에서 키, 몸무게, 나이로 비슷한 학생끼리 조를 나누는 것과 같습니다."
    )
    c1, c2 = st.columns([1.1, 1])
    with c1:
        st.image(str(OUT / "01_elbow_method.png"), caption="Elbow Method")
    with c2:
        st.image(str(OUT / "01_cluster_fire_boxplot.png"), caption="군집별 150m 화재수 분포")
    st.dataframe(cluster, hide_index=True, use_container_width=True)
    st.caption("주의: 군집은 원인 분석이 아니라 '비슷한 시설끼리 묶기'입니다. 군집번호 자체가 위험 원인은 아닙니다.")

with tab3:
    st.subheader("Lasso와 사각지대 위험도")
    st.write(
        "Lasso는 여러 변수 중에서 타깃과 관련이 약한 변수의 계수를 0에 가깝게 줄이는 방법입니다. "
        "다만 여기서는 타깃을 150m 내 재산피해액으로 잡았기 때문에, 계수는 '실제 피해액과의 관련성' 관점입니다."
    )
    left, right = st.columns(2)
    with left:
        fig = px.bar(lasso, x="변수", y="lasso_cv_coef", color="lasso_cv_coef", title="LassoCV 계수")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(lasso, hide_index=True, use_container_width=True)
    with right:
        st.write("사각지대 위험도 상위 30개")
        st.dataframe(
            top30[["구", "동", "숙소명", "업종", "주변건물수", "집중도", "위험도점수"]].head(30),
            hide_index=True,
            use_container_width=True,
        )
    st.warning(
        "중요: '주변건물수와 집중도가 낮을수록 위험'이라는 방향은 정책적 해석입니다. "
        "항상 실제 화재/피해액 예측 성능과 분리해서 말해야 합니다."
    )

with tab4:
    st.subheader("공간분석 Moran's I")
    st.write(
        "Moran's I는 비슷한 값이 가까운 곳끼리 모여 있는지 보는 지표입니다. "
        "반 친구들이 조용한 학생끼리 한쪽에 몰려 있으면 공간적으로 모여 있다고 보는 것과 비슷합니다."
    )
    st.dataframe(moran, hide_index=True, use_container_width=True)
    fig = px.bar(moran, x="좌표", y="moran_I", color="p_value", text=moran["moran_I"].round(3), title="Moran's I 비교")
    st.plotly_chart(fig, use_container_width=True)
    st.info("EPSG:5181 평면좌표 기준 Moran's I도 유의하므로, 위험도점수는 공간적으로 군집되어 있습니다.")

with tab5:
    st.subheader("GWR + Random Forest 검증")
    st.write(
        "GWR은 지역마다 변수 효과가 달라지는지 보는 모델입니다. 예를 들어 같은 도로폭이라도 강남과 마포에서 "
        "위험도에 미치는 영향이 다를 수 있다고 보는 방식입니다."
    )
    g1, g2, g3 = st.columns(3)
    g1.metric("GWR 표본", f"{int(summary['gwr_rows_sampled']):,}개")
    g2.metric("GWR bandwidth", f"{float(summary['gwr_bandwidth']):.0f}")
    g3.metric("GWR R²", f"{float(summary['gwr_r2']):.3f}")
    st.caption("GWR은 계산량 때문에 검증용 표본으로 실행했습니다.")

    fig = px.bar(imp, x="변수", y="importance", color="importance", title="순환 RF 모델 변수 중요도")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(imp, hide_index=True, use_container_width=True)

    st.error(
        "여기서 Random Forest R²가 높다고 해서 '실제 화재피해를 잘 예측한다'고 말하면 안 됩니다. "
        "타깃이 실제 피해액이 아니라 우리가 만든 위험도점수이기 때문입니다."
    )

with tab6:
    st.subheader("논리 검증 체크리스트")
    checks = pd.DataFrame(
        [
            ["군집화", "가능", "비슷한 숙소 유형을 나누는 탐색 분석으로는 적절함"],
            ["Lasso 계수", "부분 가능", "타깃을 실제 피해액으로 둘 때만 변수 관련성 해석 가능"],
            ["위험도점수", "가능", "정책 점수/우선순위 지표로 사용 가능. 예측값이라고 말하면 안 됨"],
            ["Moran's I", "가능", "위험도점수의 공간 군집성 근거로 사용 가능"],
            ["GWR", "조심", "위험도점수를 종속변수로 쓰면 산식의 지역별 재표현에 가까움"],
            ["RF 높은 R²", "위험", "위험도점수를 만든 변수로 다시 맞힌 순환 검증"],
            ["실제 화재수 예측", "보통", "R²가 낮아졌지만 진짜 검증이라 더 정직함"],
            ["실제 피해액 예측", "약함", "피해액은 이상치와 우연성이 커서 별도 모델링 필요"],
        ],
        columns=["항목", "판정", "이유"],
    )
    st.dataframe(checks, hide_index=True, use_container_width=True)

    st.markdown(
        """
**최종 판단**

이 분석은 “위험시설 우선순위와 위험요인 설명” 용도로는 쓸 수 있습니다.  
하지만 “우리 모델이 실제 재산피해액을 아주 정확히 예측한다”는 주장에는 아직 부족합니다.

**가장 안전한 발표 문장**

본 연구는 숙박시설의 구조·입지·접근성 변수를 기반으로 사각지대형 위험도 점수를 구성하고,  
공간자기상관 및 지역별 효과 차이를 검토하여 우선 점검 대상 시설을 도출하였다.  
단, Random Forest의 높은 설명력은 산식 기반 위험도점수의 재현 성능이며,  
실제 화재수 및 피해액 예측 성능은 별도 검증 지표로 제한적으로 해석하였다.
"""
    )
