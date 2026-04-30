# -*- coding: utf-8 -*-
from pathlib import Path

import json
import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(page_title="마포구 민박 위험도", layout="wide")

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "data" / "mapo_foreign_lodging_risk"


@st.cache_data
def load_data():
    summary = json.loads((OUT / "mapo_summary.json").read_text(encoding="utf-8"))
    gu = pd.read_csv(OUT / "gu_risk_summary.csv", encoding="utf-8-sig")
    upjong = pd.read_csv(OUT / "gu_upjong_risk_summary.csv", encoding="utf-8-sig")
    trend = pd.read_csv(OUT / "foreign_lodging_license_trend_2020_2025.csv", encoding="utf-8-sig")
    top = pd.read_csv(OUT / "mapo_foreign_lodging_top_risk.csv", encoding="utf-8-sig")
    return summary, gu, upjong, trend, top


if not OUT.exists():
    st.error("마포구 위험도 산출물이 없습니다. `python scripts/analyze_mapo_foreign_lodging_risk.py`를 먼저 실행하세요.")
    st.stop()

summary, gu, upjong, trend, top = load_data()

st.title("마포구 외국인관광도시민박업 위험도 진단")
st.caption("신규 인허가 급증이 실제 위험과 어떻게 연결되는지 확인")

st.info(
    "결론: 마포구 외국인관광도시민박업은 시설 수와 신규 인허가 증가가 압도적이고, "
    "실제 150m 내 화재수 기준으로도 10개구 중 상위권입니다. 다만 평균 피해액은 강남구·중구 등보다 낮아, "
    "'피해액 최고 위험지역'보다는 '시설 밀집과 화재 노출이 큰 관리 우선지역'으로 해석하는 것이 안전합니다."
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("마포구 전체 숙박시설", f"{summary['mapo_total_facilities']:,}개")
c2.metric("마포구 외국인민박", f"{summary['mapo_foreign_lodging_count']:,}개")
c3.metric("마포구 내 외국인민박 비중", f"{summary['mapo_foreign_share_in_mapo']*100:.1f}%")
c4.metric("150m 화재 경험률", f"{summary['mapo_foreign_fire_exists_rate']*100:.1f}%")

tab1, tab2, tab3, tab4 = st.tabs(["신규 증가", "구별 위험 비교", "마포구 민박 상세", "해석 문장"])

with tab1:
    st.subheader("외국인관광도시민박업 신규 인허가 추이")
    focus = trend[trend["구"].isin(["마포구", "종로구", "용산구", "송파구", "강남구", "중구"])].copy()
    fig = px.line(
        focus,
        x="인허가연도",
        y="신규인허가",
        color="구",
        markers=True,
        title="주요 구 외국인관광도시민박업 신규 인허가",
    )
    fig.update_layout(xaxis_title="연도", yaxis_title="신규 인허가")
    st.plotly_chart(fig, use_container_width=True)

    mapo = trend[trend["구"].eq("마포구")].copy()
    st.dataframe(mapo, hide_index=True, use_container_width=True)
    st.warning("마포구는 2023년 210건, 2024년 495건, 2025년 459건으로 급증했습니다.")

with tab2:
    st.subheader("구별 실제 위험 비교")
    cols = [
        "구",
        "시설수",
        "외국인민박수",
        "위험도점수_평균",
        "fire_count_150m_평균",
        "fire_exists_150m_평균",
        "target_damage_sum_천원_평균",
        "위험점수_AHP_평균",
    ]
    show = gu[cols].copy()
    show["화재경험률_%"] = show["fire_exists_150m_평균"] * 100
    show = show.sort_values("fire_count_150m_평균", ascending=False)
    st.dataframe(show, hide_index=True, use_container_width=True)

    fig = px.bar(
        show,
        x="구",
        y="fire_count_150m_평균",
        color="외국인민박수",
        text=show["fire_count_150m_평균"].round(2),
        title="구별 평균 150m 내 화재수",
    )
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.scatter(
        show,
        x="외국인민박수",
        y="fire_count_150m_평균",
        size="시설수",
        color="구",
        text="구",
        title="외국인민박 수와 평균 화재 노출",
    )
    fig2.update_traces(textposition="top center")
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.subheader("마포구 외국인관광도시민박업만 보기")
    foreign = upjong[upjong["업종"].eq("외국인관광도시민박업")].copy()
    foreign["화재경험률_%"] = foreign["fire_exists_150m_평균"] * 100
    foreign = foreign.sort_values("fire_count_150m_평균", ascending=False)
    st.dataframe(
        foreign[
            [
                "구",
                "시설수",
                "위험도점수_평균",
                "fire_count_150m_평균",
                "화재경험률_%",
                "target_damage_sum_천원_평균",
                "위험점수_AHP_평균",
            ]
        ],
        hide_index=True,
        use_container_width=True,
    )

    fig = px.bar(
        foreign,
        x="구",
        y="fire_count_150m_평균",
        color="시설수",
        text=foreign["fire_count_150m_평균"].round(2),
        title="외국인관광도시민박업 기준 구별 평균 화재수",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.write("마포구 외국인민박 위험도 상위 시설")
    table_cols = [
        "동",
        "숙소명",
        "승인연도",
        "위험도점수",
        "위험점수_AHP",
        "사각지대_위험도점수",
        "fire_count_150m",
        "target_damage_sum_천원",
        "주변건물수",
        "집중도",
        "도로폭위험도",
    ]
    table_cols = [c for c in table_cols if c in top.columns]
    st.dataframe(top[table_cols].head(30), hide_index=True, use_container_width=True)

with tab4:
    st.subheader("발표용 해석")
    st.markdown(
        """
**핵심 해석**

마포구는 외국인관광도시민박업 신규 인허가가 2023년 이후 급격히 증가했습니다.  
현재 분석대상 숙박시설 중 마포구 외국인민박은 1,070개이며, 마포구 숙박시설의 약 92.0%를 차지합니다.

실제 화재 데이터와 결합한 결과, 마포구 외국인민박의 150m 내 화재 경험률은 약 92.1%입니다.  
평균 150m 내 화재수는 6.27건으로, 외국인민박 업종 기준 10개구 중 강남구 다음의 상위권입니다.

다만 평균 재산피해액은 강남구·중구·영등포구보다 낮게 나타납니다.  
따라서 마포구는 '피해액이 가장 큰 지역'이라기보다,  
**신규 시설 증가와 실제 화재 노출이 동시에 큰 관리 우선지역**으로 보는 것이 타당합니다.

**안전한 결론 문장**

마포구 외국인관광도시민박업은 신규 인허가 증가세가 뚜렷하고, 실제 화재 발생 노출도도 높은 편이다.  
따라서 정밀 피해액 예측보다는 신규 민박 밀집지역을 중심으로 사전 점검, 소방 접근성 확인,  
불법 주정차 및 도로폭 취약지 관리를 강화할 필요가 있다.
"""
    )
