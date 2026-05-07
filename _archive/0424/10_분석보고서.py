# -*- coding: utf-8 -*-
import streamlit as st
from PIL import Image
import os

st.set_page_config(page_title="분석 보고서 아카이브", page_icon="📑", layout="wide")
st.title("📑 분석 보고서 — 시각화 아카이브")
st.caption("스크립트로 생성한 분석 이미지 전체 모음 — 클릭 시 확대")

def show_img(path, caption="", width=None):
    if os.path.exists(path):
        img = Image.open(path)
        st.image(img, caption=caption or os.path.basename(path), use_container_width=(width is None))
    else:
        st.warning(f"이미지 없음: {path}")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍 탐색적 분석",
    "🔥 화재 분석",
    "📐 회귀·진단",
    "🗺️ 공간 분석",
    "📊 군집·위험도",
])

# ─────────────────────────────────────────────
with tab1:
    st.subheader("탐색적 데이터 분석 (EDA)")

    c1, c2 = st.columns(2)
    with c1:
        show_img("data/viz_all/01_위치_산포도.png", "시설 위치 산포도")
        show_img("data/viz_all/07_상관_히트맵.png", "변수 간 상관 히트맵")
        show_img("data/pairplot.png", "주요 변수 Pair Plot")
    with c2:
        show_img("data/viz_all/02_건물나이_바이올린.png", "건물나이 분포 (바이올린)")
        show_img("data/corr_heatmap.png", "상관관계 히트맵 (상세)")
        show_img("data/vif_table.png", "VIF 다중공선성 진단")

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        show_img("data/viz_all/08_건물나이_구조노후도_hexbin.png", "건물나이 × 구조노후도 Hexbin")
        show_img("data/viz_all/11_층수_분포.png", "층수 분포")
    with c2:
        show_img("data/viz_all/13_연면적_분포.png", "연면적 분포")
        show_img("data/structure_pie.png", "건물구조 비율 (파이)")

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        show_img("data/viz_all/03_연도별_신규_스택.png", "연도별 신규 인허가 (스택)")
        show_img("data/viz_10gu_license.png", "10개구 인허가 추이")
    with c2:
        show_img("data/viz_all/10_시설군_구성.png", "업종 구성")
        show_img("data/viz_mapo_license.png", "마포구 인허가 추이")
        show_img("data/mapo_lodging_trend.png", "마포구 숙박업 트렌드")

    st.markdown("---")
    show_img("data/구동별_노후도_단속위험도.png", "구·동별 노후도 × 단속위험도")

# ─────────────────────────────────────────────
with tab2:
    st.subheader("화재 발생 분석")
    st.caption("reports/fire_visualizations_png/ — 화재출동 데이터 기반 종합 분석")

    FIRE_DIR = "reports/fire_visualizations_png"
    c1, c2 = st.columns(2)
    with c1:
        show_img(f"{FIRE_DIR}/01_year_month_fire_trend.png",   "연도·월별 화재 추이")
        show_img(f"{FIRE_DIR}/03_district_fire_map.png",       "구별 화재 발생 지도")
        show_img(f"{FIRE_DIR}/05_response_time_vs_distance.png","출동시간 vs 거리")
    with c2:
        show_img(f"{FIRE_DIR}/02_weekday_hour_fire_heatmap.png","요일×시간 화재 히트맵")
        show_img(f"{FIRE_DIR}/04_top10_fire_causes.png",        "상위 10대 발화 원인")
        show_img(f"{FIRE_DIR}/06_high_damage_fire_map_and_ranking.png","고피해 화재 지도·순위")

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        show_img("data/gu_fire_count.png", "구별 화재 건수")
    with c2:
        show_img("data/fire_kde_map.png", "화재 KDE 밀도")
    with c3:
        show_img("data/damage_distribution_2224.png", "피해 분포 (22–24년)")

    show_img("data/top10_damage_by_place.png", "발화장소별 상위 10 피해")
    show_img("data/viz_all/04_소방_골든타임.png", "소방 골든타임 분석")

# ─────────────────────────────────────────────
with tab3:
    st.subheader("회귀 분석 & 모델 진단")

    st.markdown("#### OLS 회귀 결과")
    c1, c2 = st.columns(2)
    with c1:
        show_img("data/ols_result.png",          "OLS (AHP 위험점수)")
        show_img("data/ols_100m_result.png",     "OLS (반경100m 화재수)")
    with c2:
        show_img("data/ols_count_result.png",    "OLS (화재수 카운트)")
        show_img("data/ols_improved_result.png", "OLS 개선 버전")

    st.markdown("---")
    st.markdown("#### 변수 중요도 & PCA")
    c1, c2 = st.columns(2)
    with c1:
        show_img("data/viz_all/05_PCA_AHP_산점도.png", "PCA 주성분 × AHP 산점도")
    with c2:
        show_img("data/elbow_0423.png", "K-Means 엘보우 곡선")
        show_img("data/silhouette_comparison.png", "실루엣 점수 비교")

# ─────────────────────────────────────────────
with tab4:
    st.subheader("공간 분석")

    c1, c2 = st.columns(2)
    with c1:
        show_img("data/lisa_map.png", "LISA 지역 공간자기상관 지도")
        show_img("data/fire_kde_map.png", "화재 KDE 밀도 지도")
    with c2:
        show_img("data/Map_Seoul10_Firestation.png", "서울 10개구 소방서 위치")

    st.markdown("---")
    st.markdown("#### 인터랙티브 지도 (별도 탭에서 확인)")
    st.info(
        "• **GWR 계수 지도** → 9_GWR_공간분석 페이지\n"
        "• **3D 위험도 지도** → 1_지도 페이지\n"
        "• **소방서·숙박 통합 / KDE / 버퍼** → 7_지도_아카이브 페이지"
    )

    # 위험점수 히트맵 HTML (인터랙티브)
    heatmap_html = "data/viz_all/12_위험점수_히트맵.html"
    if os.path.exists(heatmap_html):
        st.markdown("#### 위험점수 인터랙티브 히트맵")
        import streamlit.components.v1 as components
        size_mb = os.path.getsize(heatmap_html) / 1024 / 1024
        st.caption(f"파일: {heatmap_html}  ({size_mb:.1f} MB)")
        with open(heatmap_html, 'r', encoding='utf-8') as f:
            components.html(f.read(), height=600, scrolling=False)

    # 건물상가밀집도 HTML
    density_html = "data/건물상가밀집도_10개구_0417/서울10개구_건물상가_50,000sqm_밀도지도.html"
    if os.path.exists(density_html):
        st.markdown("#### 건물·상가 밀집도 지도 (50,000㎡ 단위)")
        size_mb = os.path.getsize(density_html) / 1024 / 1024
        st.caption(f"파일: {density_html}  ({size_mb:.1f} MB)")
        with open(density_html, 'r', encoding='utf-8') as f:
            components.html(f.read(), height=600, scrolling=False)

# ─────────────────────────────────────────────
with tab5:
    st.subheader("군집 분석 & 위험도 시각화")

    c1, c2 = st.columns(2)
    with c1:
        show_img("data/viz_all/06_군집_레이더.png", "군집별 레이더 차트")
        show_img("data/viz_all/09_구별_AHP위험점수.png", "구별 AHP 위험점수 분포")
    with c2:
        show_img("data/viz_all/05_PCA_AHP_산점도.png", "군집 × PCA 산점도")

    st.markdown("---")
    st.markdown("#### 인터랙티브 AHP 히트맵")
    ahp_html = "data/viz_all/12_위험점수_히트맵.html"
    if os.path.exists(ahp_html):
        import streamlit.components.v1 as components
        with open(ahp_html, 'r', encoding='utf-8') as f:
            components.html(f.read(), height=550, scrolling=False)
    else:
        st.warning("data/viz_all/12_위험점수_히트맵.html 파일 없음")
